from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from pocketlab.explorations import exploration_ids_for_sensors
from pocketlab.schemas import AccelerationSample, PhyphoxBufferMapping, SensorKind
from pocketlab.sensor_models import (
    PhyphoxBufferAlignmentReceipt,
    PhyphoxSensorProfile,
    SensorChannelDefinition,
    SensorProvenance,
    SensorRecordingUpload,
    SensorSample,
)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_SAMPLES = 60_000
ALLOWED_PORTS = {80, 8080}
BUFFER_READ_ATTEMPTS = 3
BUFFER_SETTLE_DELAY_S = 0.12
MAX_BOUNDED_TAIL_SAMPLES = 2
MAX_BOUNDED_TAIL_RATIO = 0.02
MICROPHONE_BUFFER_READ_ATTEMPTS = 12
MICROPHONE_BUFFER_SETTLE_DELAY_S = 0.15
MAX_MICROPHONE_DERIVED_TAIL_SAMPLES = 8
MAX_MICROPHONE_DERIVED_TAIL_RATIO = 0.20
MAX_MICROPHONE_DERIVED_TAIL_DURATION_S = 1.0


class PhyphoxError(RuntimeError):
    """Base error for a phyphox remote-interface operation."""


class PhyphoxUrlError(PhyphoxError):
    """The supplied phone URL is outside the intentionally narrow LAN boundary."""


@dataclass(frozen=True)
class PhyphoxProbe:
    base_url: str
    experiment_title: str
    remote_session: str | None
    measuring: bool
    compatible: bool
    buffer_mapping: PhyphoxBufferMapping
    available_buffers: list[str]
    missing_buffers: list[str]
    detected_sensors: list[SensorKind] = field(default_factory=list)
    export_buffers: list[str] = field(default_factory=list)
    exploration_matches: list[str] = field(default_factory=list)
    config_sha256: str = ""
    sensor_profiles: dict[SensorKind, PhyphoxSensorProfile] = field(default_factory=dict)


@dataclass(frozen=True)
class PhyphoxCapture:
    probe: PhyphoxProbe
    requested_duration_s: float
    actual_duration_s: float
    samples: list[AccelerationSample]
    buffer_receipt: PhyphoxBufferAlignmentReceipt | None = None


@dataclass(frozen=True)
class PhyphoxSensorCapture:
    probe: PhyphoxProbe
    profile: PhyphoxSensorProfile
    requested_duration_s: float
    actual_duration_s: float
    recording: SensorRecordingUpload


@dataclass(frozen=True)
class PhyphoxSynchronizedCapture:
    probe: PhyphoxProbe
    requested_duration_s: float
    actual_duration_s: dict[SensorKind, float]
    recordings: dict[SensorKind, SensorRecordingUpload]
    capture_group_id: str
    clock_id: str
    maximum_alignment_error_ms: float


@dataclass(frozen=True)
class _AlignedBufferSet:
    values: dict[str, list[float]]
    receipt: PhyphoxBufferAlignmentReceipt


def normalize_phyphox_base_url(value: str) -> str:
    """Accept only an IP-literal phyphox URL on its documented LAN ports.

    Keeping the bridge IP-only, HTTP-only and limited to ports 80/8080 avoids DNS
    rebinding and prevents this local helper from becoming a general URL fetcher.
    """

    raw = value.strip().rstrip("/")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise PhyphoxUrlError("phyphox 地址格式无效。") from exc

    if parsed.scheme != "http":
        raise PhyphoxUrlError("phyphox 远程接口应使用手机显示的 http:// 地址。")
    if not parsed.hostname or parsed.username or parsed.password:
        raise PhyphoxUrlError("phyphox 地址必须只包含手机 IP 和端口。")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise PhyphoxUrlError("请填写 phyphox 根地址，不要附加路径、查询参数或片段。")

    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise PhyphoxUrlError("请直接填写 phyphox 显示的局域网 IP，不要使用域名。") from exc
    if address.is_unspecified or address.is_multicast:
        raise PhyphoxUrlError("该 IP 不能作为 phyphox 手机地址。")
    if not (address.is_private or address.is_link_local or address.is_loopback):
        raise PhyphoxUrlError("只允许连接同一局域网内的手机 IP。")

    effective_port = port or 80
    if effective_port not in ALLOWED_PORTS:
        raise PhyphoxUrlError("只允许 phyphox 官方远程接口端口 80 或 8080。")

    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    suffix = "" if effective_port == 80 else f":{effective_port}"
    return f"http://{host}{suffix}"


class PhyphoxClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_phyphox_base_url(base_url)
        self._transport = transport

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=f"{self.base_url}/",
            follow_redirects=False,
            timeout=httpx.Timeout(10.0, connect=4.0),
            trust_env=False,
            transport=self._transport,
        )

    async def probe(self) -> PhyphoxProbe:
        async with self._http_client() as client:
            return await self._probe_with_client(client)

    async def capture(
        self,
        duration_s: float,
        requested_mapping: PhyphoxBufferMapping,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> PhyphoxCapture:
        async with self._http_client() as client:
            probe = await self._probe_with_client(client, requested_mapping)
            if not probe.compatible:
                missing = "、".join(probe.missing_buffers)
                raise PhyphoxError(f"当前 phyphox 实验缺少加速度缓冲区：{missing}")

            await self._control(client, "clear")
            await self._control(client, "start")
            started = True
            try:
                await sleep(duration_s)
                await self._control(client, "stop")
                started = False
            finally:
                if started:
                    with suppress(PhyphoxError):
                        await self._control(client, "stop")

            mapping = probe.buffer_mapping
            role_buffers = {
                "timestamp": mapping.timestamp,
                "x": mapping.x,
                "y": mapping.y,
                "z": mapping.z,
            }
            payload, aligned_sets = await self._read_stable_full_buffer_sets(
                client,
                {"accelerometer": role_buffers},
                expected_session=probe.remote_session,
                sleep=sleep,
            )
            source_session = _read_remote_session(payload)
            aligned = aligned_sets["accelerometer"]
            samples, actual_duration_s = _samples_from_raw(
                aligned.values,
                duration_s,
            )
            return PhyphoxCapture(
                probe=PhyphoxProbe(
                    base_url=probe.base_url,
                    experiment_title=probe.experiment_title,
                    remote_session=source_session or probe.remote_session,
                    measuring=False,
                    compatible=True,
                    buffer_mapping=mapping,
                    available_buffers=probe.available_buffers,
                    missing_buffers=[],
                    detected_sensors=probe.detected_sensors,
                    export_buffers=probe.export_buffers,
                    exploration_matches=probe.exploration_matches,
                    config_sha256=probe.config_sha256,
                    sensor_profiles=probe.sensor_profiles,
                ),
                requested_duration_s=duration_s,
                actual_duration_s=actual_duration_s,
                samples=samples,
                buffer_receipt=aligned.receipt,
            )

    async def capture_sensor(
        self,
        sensor: SensorKind,
        duration_s: float,
        *,
        label: str,
        notes: str = "",
        privacy_acknowledged: bool = False,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> PhyphoxSensorCapture:
        """Capture a typed numeric series using a profile derived from `/config`.

        Every built-in phone sensor with a verified profile uses this contract,
        including accelerometer. Bluetooth remains separate because its channel
        semantics and units depend on the external device protocol.
        """

        if sensor == "bluetooth":
            raise PhyphoxError(
                "Bluetooth 需先注册设备专用通道、单位与采样时钟协议；"
                "当前只能执行连接与能力检查，不能把未知数值作为物理证据。"
            )
        async with self._http_client() as client:
            probe = await self._probe_with_client(client)
            if probe.measuring:
                raise PhyphoxError(
                    "phyphox 当前已有测量正在运行；PocketLab 不会静默清空它，"
                    "请先在手机停止并确认可覆盖的数据。"
                )
            profile = probe.sensor_profiles.get(sensor)
            if profile is None:
                raise PhyphoxError(
                    f"当前 phyphox 实验没有可验证的 {sensor} 输入/缓冲区映射。"
                )
            if sensor == "location" and not privacy_acknowledged:
                raise PhyphoxError("位置采集需要先确认轨迹隐私提示。")

            await self._control(client, "clear")
            await self._control(client, "start")
            started = True
            try:
                await sleep(duration_s)
                await self._control(client, "stop")
                started = False
            finally:
                if started:
                    with suppress(PhyphoxError):
                        await self._control(client, "stop")

            role_buffers = {"timestamp": profile.timestamp_buffer, **profile.channel_buffers}
            payload, aligned_sets = await self._read_stable_full_buffer_sets(
                client,
                {sensor: role_buffers},
                expected_session=probe.remote_session,
                sleep=sleep,
            )
            source_session = _read_remote_session(payload)

            config_after = await self._get_json(client, "config")
            if _config_sha256(config_after) != probe.config_sha256:
                raise PhyphoxError("采集期间 phyphox 实验配置发生变化，请重新探测后重测。")

            aligned = aligned_sets[sensor]
            samples, actual_duration_s = _sensor_samples_from_raw(
                aligned.values,
                profile,
                duration_s,
            )
            recording = SensorRecordingUpload(
                label=label,
                device=f"phyphox · {probe.experiment_title}"[:120],
                sensor=sensor,
                notes=notes,
                channels={
                    channel: SensorChannelDefinition(unit=profile.channel_units[channel])
                    for channel in profile.channel_buffers
                },
                samples=samples,
                provenance=SensorProvenance(
                    source="phyphox_remote",
                    experiment_title=probe.experiment_title,
                    remote_session=source_session or probe.remote_session,
                    config_sha256=probe.config_sha256,
                    channel_mapping={
                        "timestamp": profile.timestamp_buffer,
                        **profile.channel_buffers,
                    },
                    privacy_acknowledged=privacy_acknowledged,
                    phyphox_buffer_receipt=aligned.receipt,
                ),
            )
            return PhyphoxSensorCapture(
                probe=probe,
                profile=profile,
                requested_duration_s=duration_s,
                actual_duration_s=actual_duration_s,
                recording=recording,
            )

    async def capture_sensors(
        self,
        sensors: tuple[SensorKind, ...],
        duration_s: float,
        *,
        label: str,
        notes: str = "",
        privacy_acknowledged: bool = False,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> PhyphoxSynchronizedCapture:
        """Capture 2-3 sensor profiles under one phyphox start/stop and clock."""

        if not 2 <= len(sensors) <= 3 or len(sensors) != len(set(sensors)):
            raise PhyphoxError("同步采集要求 2 到 3 个互不重复的传感器。")
        if "bluetooth" in sensors:
            raise PhyphoxError("Bluetooth 没有可验证的同步数值采集协议。")
        if any(sensor in {"microphone", "location"} for sensor in sensors) and not privacy_acknowledged:
            raise PhyphoxError("麦克风或位置同步采集需要先确认隐私提示。")

        async with self._http_client() as client:
            probe = await self._probe_with_client(client)
            if probe.measuring:
                raise PhyphoxError(
                    "phyphox 当前已有测量正在运行；PocketLab 不会静默清空它，"
                    "请先在手机停止并确认可覆盖的数据。"
                )
            missing = [sensor for sensor in sensors if sensor not in probe.sensor_profiles]
            if missing:
                raise PhyphoxError(
                    "当前 phyphox 实验没有同时暴露全部传感器 profile："
                    + "、".join(missing)
                )
            profiles = {sensor: probe.sensor_profiles[sensor] for sensor in sensors}

            await self._control(client, "clear")
            await self._control(client, "start")
            started = True
            try:
                await sleep(duration_s)
                await self._control(client, "stop")
                started = False
            finally:
                if started:
                    with suppress(PhyphoxError):
                        await self._control(client, "stop")

            role_buffer_groups: dict[str, dict[str, str]] = {}
            for profile in profiles.values():
                role_buffer_groups[profile.sensor] = {
                    "timestamp": profile.timestamp_buffer,
                    **profile.channel_buffers,
                }
            payload, aligned_sets = await self._read_stable_full_buffer_sets(
                client,
                role_buffer_groups,
                expected_session=probe.remote_session,
                sleep=sleep,
            )
            source_session = _read_remote_session(payload)
            config_after = await self._get_json(client, "config")
            if _config_sha256(config_after) != probe.config_sha256:
                raise PhyphoxError("采集期间 phyphox 实验配置发生变化，请重新探测后重测。")

            timestamp_series = {
                sensor: aligned_sets[sensor].values["timestamp"] for sensor in profiles
            }
            alignment_error_ms = _window_alignment_error_ms(timestamp_series)
            if alignment_error_ms > 250:
                raise PhyphoxError(
                    f"多传感器原始时间窗偏差为 {alignment_error_ms:.1f} ms，"
                    "超过 250 ms 的可信同步上限。"
                )

            capture_group_id = f"phyphox-sync-{uuid4().hex[:16]}"
            clock_id = f"phyphox-clock-{probe.config_sha256[:16]}"
            recordings: dict[SensorKind, SensorRecordingUpload] = {}
            actual_durations: dict[SensorKind, float] = {}
            for sensor, profile in profiles.items():
                aligned = aligned_sets[sensor]
                samples, actual_duration_s = _sensor_samples_from_raw(
                    aligned.values,
                    profile,
                    duration_s,
                )
                actual_durations[sensor] = actual_duration_s
                recordings[sensor] = SensorRecordingUpload(
                    label=f"{label} · {sensor}"[:80],
                    device=f"phyphox · {probe.experiment_title}"[:120],
                    sensor=sensor,
                    notes=notes,
                    channels={
                        channel: SensorChannelDefinition(unit=profile.channel_units[channel])
                        for channel in profile.channel_buffers
                    },
                    samples=samples,
                    provenance=SensorProvenance(
                        source="phyphox_remote",
                        experiment_title=probe.experiment_title,
                        remote_session=source_session or probe.remote_session,
                        config_sha256=probe.config_sha256,
                        channel_mapping={
                            "timestamp": profile.timestamp_buffer,
                            **profile.channel_buffers,
                        },
                        privacy_acknowledged=privacy_acknowledged,
                        capture_group_id=capture_group_id,
                        clock_id=clock_id,
                        maximum_alignment_error_ms=alignment_error_ms,
                        alignment_method="shared_monotonic_clock",
                        phyphox_buffer_receipt=aligned.receipt,
                    ),
                )
            return PhyphoxSynchronizedCapture(
                probe=probe,
                requested_duration_s=duration_s,
                actual_duration_s=actual_durations,
                recordings=recordings,
                capture_group_id=capture_group_id,
                clock_id=clock_id,
                maximum_alignment_error_ms=alignment_error_ms,
            )

    async def _read_stable_full_buffer_sets(
        self,
        client: httpx.AsyncClient,
        role_buffer_groups: dict[str, dict[str, str]],
        *,
        expected_session: str | None,
        sleep: Callable[[float], Awaitable[None]],
    ) -> tuple[dict[str, Any], dict[str, _AlignedBufferSet]]:
        """Read index-paired buffers, retrying only an apparent append-tail race.

        Each sensor group is aligned independently.  This deliberately does not
        resample different sensors onto a shared rate or interpolate missing points.
        """

        requested_buffers = list(
            dict.fromkeys(
                buffer_name
                for role_buffers in role_buffer_groups.values()
                for buffer_name in role_buffers.values()
            )
        )
        observed_session = expected_session
        microphone_derived_groups = {
            group
            for group, role_buffers in role_buffer_groups.items()
            if _is_microphone_derived_role_set(group, set(role_buffers))
        }
        read_attempts = (
            MICROPHONE_BUFFER_READ_ATTEMPTS
            if microphone_derived_groups
            else BUFFER_READ_ATTEMPTS
        )
        settle_delay_s = (
            MICROPHONE_BUFFER_SETTLE_DELAY_S
            if microphone_derived_groups
            else BUFFER_SETTLE_DELAY_S
        )
        append_only_observed = {group: True for group in microphone_derived_groups}
        previous_raw_sets: dict[str, dict[str, list[float]]] = {}
        # Some phyphox experiments finish a derived analysis append immediately
        # after stop. The official Audio amplitude experiment writes `time` before
        # its derived `dB` output, so that profile gets a longer bounded settle
        # window. Native sensor buffers keep the original fast, strict policy.
        await sleep(settle_delay_s)
        for attempt in range(1, read_attempts + 1):
            payload = await self._get_json(
                client,
                "get",
                params=[(name, "full") for name in requested_buffers],
            )
            source_session = _read_remote_session(payload)
            if observed_session and source_session and source_session != observed_session:
                raise PhyphoxError("采集期间 phyphox 切换了实验，请重新连接后重测。")
            observed_session = source_session or observed_session

            buffers = payload.get("buffer")
            if not isinstance(buffers, dict):
                raise PhyphoxError("phyphox /get 没有返回测量缓冲区。")
            raw_sets = {
                group: {
                    role: _buffer_values(buffers, name, require_full=True)
                    for role, name in role_buffers.items()
                }
                for group, role_buffers in role_buffer_groups.items()
            }
            for group in microphone_derived_groups:
                previous = previous_raw_sets.get(group)
                if previous is not None and not _buffer_set_is_append_only(
                    previous,
                    raw_sets[group],
                ):
                    append_only_observed[group] = False
                previous_raw_sets[group] = raw_sets[group]
            all_exact = all(
                len({len(values) for values in raw.values()}) == 1
                for raw in raw_sets.values()
            )
            if all_exact or attempt == read_attempts:
                return payload, {
                    group: _align_buffer_set(
                        raw,
                        read_attempts=attempt,
                        allow_microphone_derived_tail=(
                            group in microphone_derived_groups
                            and append_only_observed[group]
                        ),
                    )
                    for group, raw in raw_sets.items()
                }
            await sleep(settle_delay_s)

        raise AssertionError("unreachable phyphox buffer read loop")

    async def _probe_with_client(
        self,
        client: httpx.AsyncClient,
        requested_mapping: PhyphoxBufferMapping | None = None,
    ) -> PhyphoxProbe:
        config = await self._get_json(client, "config")
        available = _available_buffers(config)
        # iOS phyphox may reject a bare /get. Request one known buffer without
        # relying on it as evidence; the capture path later requests full data.
        status_payload = await self._get_json(client, "get", params={available[0]: ""})
        mapping, missing = _resolve_mapping(available, requested_mapping)
        sensors = _detected_sensors(config, available)
        profiles = _resolve_sensor_profiles(config, available)
        title = str(config.get("localTitle") or config.get("title") or "phyphox experiment")
        status = status_payload.get("status")
        measuring = bool(status.get("measuring")) if isinstance(status, dict) else False
        return PhyphoxProbe(
            base_url=self.base_url,
            experiment_title=title[:120],
            remote_session=_read_remote_session(status_payload),
            measuring=measuring,
            compatible=not missing,
            buffer_mapping=mapping,
            available_buffers=available,
            missing_buffers=missing,
            detected_sensors=sensors,
            export_buffers=_export_buffers(config),
            exploration_matches=exploration_ids_for_sensors(sensors),
            config_sha256=_config_sha256(config),
            sensor_profiles=profiles,
        )

    async def _control(self, client: httpx.AsyncClient, command: str) -> None:
        payload = await self._get_json(client, "control", params={"cmd": command})
        if payload.get("result") is not True:
            raise PhyphoxError(f"phyphox 未能执行 {command} 命令。")

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        path: str,
        params: Any = None,
    ) -> dict[str, Any]:
        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise PhyphoxError("连接 phyphox 超时，请确认手机与电脑在同一网络。") from exc
        except httpx.HTTPError as exc:
            raise PhyphoxError(f"phyphox 请求失败：{exc}") from exc
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise PhyphoxError("phyphox 返回数据超过 8 MB，请缩短采集时长。")
        try:
            payload = response.json()
        except ValueError as exc:
            raise PhyphoxError("phyphox 返回了无法解析的响应。") from exc
        if not isinstance(payload, dict):
            raise PhyphoxError("phyphox 响应结构无效。")
        return payload


async def probe_phyphox(
    base_url: str,
    requested_mapping: PhyphoxBufferMapping | None = None,
) -> PhyphoxProbe:
    client = PhyphoxClient(base_url)
    if requested_mapping is None:
        return await client.probe()
    async with client._http_client() as http_client:
        return await client._probe_with_client(http_client, requested_mapping)


async def capture_phyphox_acceleration(
    base_url: str,
    duration_s: float,
    buffer_mapping: PhyphoxBufferMapping,
) -> PhyphoxCapture:
    return await PhyphoxClient(base_url).capture(duration_s, buffer_mapping)


async def capture_phyphox_sensor(
    base_url: str,
    sensor: SensorKind,
    duration_s: float,
    *,
    label: str,
    notes: str = "",
    privacy_acknowledged: bool = False,
) -> PhyphoxSensorCapture:
    return await PhyphoxClient(base_url).capture_sensor(
        sensor,
        duration_s,
        label=label,
        notes=notes,
        privacy_acknowledged=privacy_acknowledged,
    )


async def capture_phyphox_sensors(
    base_url: str,
    sensors: tuple[SensorKind, ...],
    duration_s: float,
    *,
    label: str,
    notes: str = "",
    privacy_acknowledged: bool = False,
) -> PhyphoxSynchronizedCapture:
    return await PhyphoxClient(base_url).capture_sensors(
        sensors,
        duration_s,
        label=label,
        notes=notes,
        privacy_acknowledged=privacy_acknowledged,
    )


def _available_buffers(config: dict[str, Any]) -> list[str]:
    buffers = config.get("buffers")
    if not isinstance(buffers, list):
        raise PhyphoxError("phyphox /config 没有返回缓冲区列表。")
    names = []
    for entry in buffers:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(entry["name"])
    if not names:
        raise PhyphoxError("当前 phyphox 实验没有可读取的缓冲区。")
    return sorted(set(names), key=str.casefold)


_SENSOR_ALIASES: dict[SensorKind, tuple[str, ...]] = {
    "accelerometer": ("accelerometer", "acceleration", "linear_acceleration", "accx"),
    "gyroscope": ("gyroscope", "angular_velocity", "gyr"),
    "magnetometer": ("magnetometer", "magnetic_field", "magnetic"),
    "light": ("light", "illuminance", "lux"),
    "pressure": ("pressure", "barometer", "barometric"),
    "proximity": ("proximity", "distance"),
    "microphone": ("microphone", "audio", "sound"),
    "location": ("location", "gps", "latitude", "longitude"),
    "bluetooth": ("bluetooth", "ble"),
}

_SOURCE_TO_SENSOR: dict[str, SensorKind] = {
    "accelerometer": "accelerometer",
    # The official "Acceleration (without g)" experiment exposes the Android
    # TYPE_LINEAR_ACCELERATION input under this source name while retaining the
    # standard acc_time/accX/accY/accZ output contract.
    "linear_acceleration": "accelerometer",
    "gyroscope": "gyroscope",
    "magnetic_field": "magnetometer",
    "magnetometer": "magnetometer",
    "light": "light",
    "pressure": "pressure",
    "proximity": "proximity",
    "audio": "microphone",
    "microphone": "microphone",
    "location": "location",
    "bluetooth": "bluetooth",
}

_PROFILE_SPECS: dict[SensorKind, dict[str, Any]] = {
    "accelerometer": {
        "required": ("x", "y", "z"),
        "units": {"x": "m/s^2", "y": "m/s^2", "z": "m/s^2"},
        "aliases": {
            "t": ("acc_time", "time"),
            "x": ("accX",),
            "y": ("accY",),
            "z": ("accZ",),
        },
    },
    "gyroscope": {
        "required": ("x", "y", "z"),
        "units": {"x": "rad/s", "y": "rad/s", "z": "rad/s"},
        "aliases": {
            "t": ("gyr_time", "time"),
            "x": ("gyrX",),
            "y": ("gyrY",),
            "z": ("gyrZ",),
        },
    },
    "magnetometer": {
        "required": ("x", "y", "z"),
        "optional": ("accuracy",),
        "units": {"x": "uT", "y": "uT", "z": "uT", "accuracy": "state"},
        "aliases": {
            "t": ("mag_time", "time"),
            "x": ("magX",),
            "y": ("magY",),
            "z": ("magZ",),
            "accuracy": ("magAccuracy",),
        },
    },
    "light": {
        "required": ("illuminance",),
        "units": {"illuminance": "lx"},
        "component_aliases": {"illuminance": ("x", "value")},
        "aliases": {
            "t": ("illum_time", "time"),
            "illuminance": ("illum", "light"),
        },
    },
    "pressure": {
        "required": ("pressure",),
        "units": {"pressure": "hPa"},
        "component_aliases": {"pressure": ("x", "value")},
        "aliases": {"t": ("p_time", "time"), "pressure": ("pressure",)},
    },
    "proximity": {
        "required": ("distance",),
        "units": {"distance": "cm"},
        "component_aliases": {"distance": ("x", "value")},
        "aliases": {"t": ("traw", "time"), "distance": ("amplitude", "proximity")},
    },
    "microphone": {
        "required_any": ("level_db", "amplitude"),
        "units": {"level_db": "dB_relative", "amplitude": "a.u."},
        "aliases": {
            "t": ("time", "audio_time"),
            "level_db": ("dB", "db"),
            "amplitude": ("amplitude", "audio_amplitude"),
        },
    },
    "location": {
        "required": ("lat", "lon"),
        "optional": ("accuracy", "speed", "status", "altitude", "vertical_accuracy"),
        "units": {
            "lat": "deg",
            "lon": "deg",
            "accuracy": "m",
            "speed": "m/s",
            "status": "state",
            "altitude": "m",
            "vertical_accuracy": "m",
        },
        "aliases": {
            "t": ("t", "time"),
            "lat": ("lat", "latitude"),
            "lon": ("lon", "longitude"),
            "accuracy": ("accuracy",),
            "speed": ("v", "speed"),
            "status": ("status",),
            "altitude": ("z", "zwgs84", "altitude"),
            "vertical_accuracy": ("zAccuracy", "verticalAccuracy"),
        },
    },
}


def _detected_sensors(
    config: dict[str, Any],
    available_buffers: list[str],
) -> list[SensorKind]:
    """Normalize phyphox input declarations without depending on one app version."""

    inputs = config.get("inputs", [])
    tokens = _all_string_values(inputs)
    tokens.extend(available_buffers)
    title = config.get("localTitle") or config.get("title")
    if isinstance(title, str):
        tokens.append(title)
    haystack = " ".join(tokens).casefold().replace("-", "_").replace(" ", "_")
    return [
        sensor
        for sensor, aliases in _SENSOR_ALIASES.items()
        if any(alias in haystack for alias in aliases)
    ]


def _all_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            result.append(str(key))
            result.extend(_all_string_values(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_all_string_values(child))
        return result
    return []


def _export_buffers(config: dict[str, Any]) -> list[str]:
    names: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.casefold() in {"buffer", "source"} and isinstance(child, str):
                    names.append(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config.get("export", []))
    return sorted(set(names), key=str.casefold)


def _resolve_mapping(
    available: list[str],
    requested: PhyphoxBufferMapping | None,
) -> tuple[PhyphoxBufferMapping, list[str]]:
    default = requested or PhyphoxBufferMapping()
    casefolded = {item.casefold(): item for item in available}
    candidates = {
        "timestamp": [default.timestamp, "acc_time", "time"],
        "x": [default.x, "accX", "x"],
        "y": [default.y, "accY", "y"],
        "z": [default.z, "accZ", "z"],
    }
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for field_name, names in candidates.items():
        match = next((casefolded[name.casefold()] for name in names if name.casefold() in casefolded), None)
        if match is None:
            resolved[field_name] = names[0]
            missing.append(names[0])
        else:
            resolved[field_name] = match
    return PhyphoxBufferMapping(**resolved), missing


def _config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolve_sensor_profiles(
    config: dict[str, Any],
    available: list[str],
) -> dict[SensorKind, PhyphoxSensorProfile]:
    casefolded = {name.casefold(): name for name in available}
    input_mappings: dict[SensorKind, dict[str, str]] = {}
    input_buffer_owners: dict[str, set[SensorKind]] = {}
    inputs = config.get("inputs")
    if isinstance(inputs, list):
        for entry in inputs:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("source", "")).casefold()
            sensor = _SOURCE_TO_SENSOR.get(source)
            if sensor is None or sensor == "bluetooth":
                continue
            mapped = input_mappings.setdefault(sensor, {})
            outputs = entry.get("outputs")
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                for component, buffer_name in output.items():
                    if isinstance(buffer_name, str) and buffer_name.casefold() in casefolded:
                        resolved_buffer = casefolded[buffer_name.casefold()]
                        mapped[str(component)] = resolved_buffer
                        input_buffer_owners.setdefault(resolved_buffer, set()).add(sensor)

    profiles: dict[SensorKind, PhyphoxSensorProfile] = {}
    for sensor, spec in _PROFILE_SPECS.items():
        # A familiar buffer name is discovery evidence only. Authorize a capture
        # profile only when `/config.inputs[*].source` identifies the sensor.
        if sensor not in input_mappings:
            continue
        components = input_mappings.get(sensor, {})
        resolved: dict[str, str] = {}
        used_input_mapping = False
        component_aliases = spec.get("component_aliases", {})
        all_roles = set(spec.get("required", ())) | set(spec.get("optional", ()))
        all_roles |= set(spec.get("required_any", ()))
        all_roles.add("t")
        for role in all_roles:
            component_names = (role, *component_aliases.get(role, ()))
            input_match = next(
                (components[name] for name in component_names if name in components),
                None,
            )
            if input_match is not None and (
                sensor != "microphone" or role not in {"level_db", "amplitude"}
            ):
                # Raw audio input is deliberately not a supported evidence channel.
                resolved[role] = input_match
                used_input_mapping = True
                continue
            aliases = spec.get("aliases", {}).get(role, ())
            alias_match = next(
                (
                    casefolded[name.casefold()]
                    for name in aliases
                    if name.casefold() in casefolded
                    and (
                        not input_buffer_owners.get(casefolded[name.casefold()])
                        or sensor in input_buffer_owners[casefolded[name.casefold()]]
                    )
                ),
                None,
            )
            if alias_match is not None:
                resolved[role] = alias_match

        required = set(spec.get("required", ()))
        required_any = set(spec.get("required_any", ()))
        if "t" not in resolved or not required.issubset(resolved):
            continue
        if required_any and not required_any.intersection(resolved):
            continue
        channel_buffers = {role: name for role, name in resolved.items() if role != "t"}
        profiles[sensor] = PhyphoxSensorProfile(
            sensor=sensor,
            timestamp_buffer=resolved["t"],
            channel_buffers=channel_buffers,
            channel_units={role: spec["units"][role] for role in channel_buffers},
            resolution_source=(
                "input_mapping" if used_input_mapping else "official_raw_alias"
            ),
        )
    return profiles


def _read_remote_session(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if not isinstance(status, dict):
        return None
    session = status.get("session")
    return str(session)[:120] if session is not None else None


def _sensor_samples_from_payload(
    payload: dict[str, Any],
    profile: PhyphoxSensorProfile,
    requested_duration_s: float,
) -> tuple[list[SensorSample], float]:
    buffers = payload.get("buffer")
    if not isinstance(buffers, dict):
        raise PhyphoxError("phyphox /get 没有返回测量缓冲区。")

    names = {"timestamp": profile.timestamp_buffer, **profile.channel_buffers}
    raw = {
        role: _buffer_values(buffers, name, require_full=True) for role, name in names.items()
    }
    aligned = _align_buffer_set(raw, read_attempts=1)
    return _sensor_samples_from_raw(aligned.values, profile, requested_duration_s)


def _sensor_samples_from_raw(
    raw: dict[str, list[float]],
    profile: PhyphoxSensorProfile,
    requested_duration_s: float,
) -> tuple[list[SensorSample], float]:
    sample_count = len(raw["timestamp"])
    if sample_count < 2:
        raise PhyphoxError("phyphox 至少需要返回两个带时间的采样点。")
    if sample_count > 120_000:
        raise PhyphoxError("phyphox 采样点超过 120000，请缩短采集时长。")
    timestamps = raw["timestamp"]
    if any(right <= left for left, right in pairwise(timestamps)):
        raise PhyphoxError("phyphox 时间缓冲区不是严格递增的。")
    actual_duration_s = timestamps[-1] - timestamps[0]
    if actual_duration_s < max(0.25, requested_duration_s * 0.7):
        raise PhyphoxError(
            f"有效记录只有 {actual_duration_s:.2f} 秒，明显短于请求的 {requested_duration_s:.2f} 秒。"
        )
    origin = timestamps[0]
    samples = [
        SensorSample(
            timestamp_ms=(timestamp - origin) * 1000.0,
            values={role: raw[role][index] for role in profile.channel_buffers},
        )
        for index, timestamp in enumerate(timestamps)
    ]
    return samples, actual_duration_s


def _window_alignment_error_ms(
    timestamp_series: dict[SensorKind, list[float]],
) -> float:
    if not timestamp_series:
        raise PhyphoxError("同步采集没有可验证的时间缓冲区。")
    starts = []
    ends = []
    for timestamps in timestamp_series.values():
        if len(timestamps) < 2 or any(
            right <= left for left, right in pairwise(timestamps)
        ):
            raise PhyphoxError("同步采集时间缓冲区必须至少两点且严格递增。")
        starts.append(timestamps[0])
        ends.append(timestamps[-1])
    if min(ends) <= max(starts):
        raise PhyphoxError("多传感器采集时间窗没有共同重叠区间。")
    return round(max(max(starts) - min(starts), max(ends) - min(ends)) * 1000.0, 6)


def _samples_from_payload(
    payload: dict[str, Any],
    mapping: PhyphoxBufferMapping,
    requested_duration_s: float,
) -> tuple[list[AccelerationSample], float]:
    buffers = payload.get("buffer")
    if not isinstance(buffers, dict):
        raise PhyphoxError("phyphox /get 没有返回测量缓冲区。")

    raw = {
        "timestamp": _buffer_values(buffers, mapping.timestamp),
        "x": _buffer_values(buffers, mapping.x),
        "y": _buffer_values(buffers, mapping.y),
        "z": _buffer_values(buffers, mapping.z),
    }
    aligned = _align_buffer_set(raw, read_attempts=1)
    return _samples_from_raw(aligned.values, requested_duration_s)


def _samples_from_raw(
    raw: dict[str, list[float]],
    requested_duration_s: float,
) -> tuple[list[AccelerationSample], float]:
    sample_count = len(raw["timestamp"])
    if sample_count < 64:
        raise PhyphoxError(f"phyphox 只返回了 {sample_count} 个采样点，至少需要 64 个。")
    if sample_count > MAX_SAMPLES:
        raise PhyphoxError("phyphox 采样点超过 60000，请缩短采集时长。")

    timestamps = raw["timestamp"]
    if any(right <= left for left, right in pairwise(timestamps)):
        raise PhyphoxError("phyphox 时间缓冲区不是严格递增的。")
    actual_duration_s = timestamps[-1] - timestamps[0]
    if actual_duration_s < max(0.25, requested_duration_s * 0.7):
        raise PhyphoxError(
            f"有效记录只有 {actual_duration_s:.2f} 秒，明显短于请求的 {requested_duration_s:.2f} 秒。"
        )

    origin = timestamps[0]
    samples = [
        AccelerationSample(
            timestamp_ms=(timestamp - origin) * 1000.0,
            x=x,
            y=y,
            z=z,
        )
        for timestamp, x, y, z in zip(
            timestamps,
            raw["x"],
            raw["y"],
            raw["z"],
            strict=True,
        )
    ]
    return samples, actual_duration_s


def _align_buffer_set(
    raw: dict[str, list[float]],
    *,
    read_attempts: int,
    allow_microphone_derived_tail: bool = False,
) -> _AlignedBufferSet:
    """Keep an original common prefix when only a bounded append tail differs."""

    lengths = {role: len(values) for role, values in raw.items()}
    if not lengths:
        raise PhyphoxError("phyphox 没有返回可对齐的缓冲区。")
    shortest = min(lengths.values())
    longest = max(lengths.values())
    if shortest < 2:
        raise PhyphoxError("phyphox 至少需要返回两个可配对的采样点。")

    gap = longest - shortest
    if gap == 0:
        method = "exact"
    else:
        within_absolute_bound = gap <= MAX_BOUNDED_TAIL_SAMPLES
        within_relative_bound = gap == 1 or gap / longest <= MAX_BOUNDED_TAIL_RATIO
        generic_tail_is_bounded = within_absolute_bound and within_relative_bound
        microphone_tail_is_bounded = (
            allow_microphone_derived_tail
            and read_attempts >= MICROPHONE_BUFFER_READ_ATTEMPTS
            and _is_bounded_microphone_derived_tail(raw, shortest, longest)
        )
        if not generic_tail_is_bounded and not microphone_tail_is_bounded:
            details = ", ".join(f"{role}={length}" for role, length in lengths.items())
            raise PhyphoxError(
                "phyphox 缓冲区长度差异无法确认只是末尾写入竞态；"
                f"不会插值、抽样或静默截断：{details}"
            )
        method = "bounded_common_prefix"

    aligned_values = {role: values[:shortest] for role, values in raw.items()}
    discarded = {
        role: length - shortest for role, length in lengths.items() if length > shortest
    }
    receipt = PhyphoxBufferAlignmentReceipt(
        read_attempts=read_attempts,
        alignment_method=method,
        original_lengths=lengths,
        aligned_sample_count=shortest,
        discarded_tail_samples=discarded,
    )
    return _AlignedBufferSet(values=aligned_values, receipt=receipt)


def _is_microphone_derived_role_set(group: str, roles: set[str]) -> bool:
    channels = roles - {"timestamp"}
    return (
        group == "microphone"
        and "timestamp" in roles
        and bool(channels)
        and channels <= {"level_db", "amplitude"}
    )


def _buffer_set_is_append_only(
    previous: dict[str, list[float]],
    current: dict[str, list[float]],
) -> bool:
    if set(previous) != set(current):
        return False
    return all(
        len(current[role]) >= len(previous_values)
        and current[role][: len(previous_values)] == previous_values
        for role, previous_values in previous.items()
    )


def _is_bounded_microphone_derived_tail(
    raw: dict[str, list[float]],
    shortest: int,
    longest: int,
) -> bool:
    """Authorize only a short unpaired timestamp tail from derived audio history.

    This is narrower than generic truncation: the value channels must have one
    common length, only the monotonic timestamp buffer may be longer, and the
    discarded end of the time axis must cover at most one second. The caller also
    proves that repeated snapshots were append-only across the full settle window.
    """

    if not _is_microphone_derived_role_set("microphone", set(raw)):
        return False
    timestamps = raw["timestamp"]
    channel_lengths = {
        len(values) for role, values in raw.items() if role != "timestamp"
    }
    gap = longest - shortest
    if (
        len(timestamps) != longest
        or channel_lengths != {shortest}
        or gap > MAX_MICROPHONE_DERIVED_TAIL_SAMPLES
        or gap / longest > MAX_MICROPHONE_DERIVED_TAIL_RATIO
        or any(right <= left for left, right in pairwise(timestamps))
    ):
        return False
    unmatched_tail_duration_s = timestamps[-1] - timestamps[shortest - 1]
    return unmatched_tail_duration_s <= MAX_MICROPHONE_DERIVED_TAIL_DURATION_S


def _buffer_values(
    buffers: dict[str, Any],
    name: str,
    *,
    require_full: bool = False,
) -> list[float]:
    entry = buffers.get(name)
    values = entry.get("buffer") if isinstance(entry, dict) else None
    if not isinstance(values, list):
        raise PhyphoxError(f"phyphox 未返回缓冲区 {name}。")
    if require_full and isinstance(entry, dict):
        update_mode = entry.get("updateMode")
        if update_mode is not None and update_mode != "full":
            raise PhyphoxError(f"phyphox 缓冲区 {name} 不是 full 模式响应。")
    converted: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PhyphoxError(f"phyphox 缓冲区 {name} 含有非数值数据。") from exc
        if not math.isfinite(number):
            raise PhyphoxError(f"phyphox 缓冲区 {name} 含有非有限值。")
        converted.append(number)
    return converted
