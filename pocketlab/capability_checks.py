from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pocketlab.phyphox import PhyphoxProbe
from pocketlab.sensor_models import (
    PhyphoxSensorProfile,
    SensorCapability,
    SensorKind,
)


class SensorCapabilityCheck(BaseModel):
    """A redacted, non-scientific inspection of one phyphox input capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    sensor: SensorKind
    status: Literal["profile_ready", "detected", "not_detected"]
    experiment_title: str = Field(min_length=1, max_length=120)
    measuring: bool
    detected_sensors: list[SensorKind] = Field(default_factory=list)
    available_buffers: list[str] = Field(default_factory=list, max_length=128)
    export_buffers: list[str] = Field(default_factory=list, max_length=128)
    profile: PhyphoxSensorProfile | None = None
    analyzer_id: str | None = None
    analyzer_maturity: str
    can_capture: bool
    can_analyze: bool
    can_start_bounded_agent: bool = False
    blockers: list[str] = Field(default_factory=list, max_length=12)
    next_steps: list[str] = Field(default_factory=list, max_length=12)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    privacy_statement: str


def build_sensor_capability_check(
    probe: PhyphoxProbe,
    sensor: SensorKind,
    capabilities: list[SensorCapability],
) -> SensorCapabilityCheck:
    """Describe what the current experiment proves without authorizing new claims."""

    capability = next((item for item in capabilities if item.sensor == sensor), None)
    maturity = capability.maturity if capability is not None else "detectable"
    analyzer_id = capability.analyzer_id if capability is not None else None
    profile = probe.sensor_profiles.get(sensor)
    detected = sensor in probe.detected_sensors
    can_analyze = maturity in {"analysis_ready", "agent_ready", "release_candidate"}
    can_capture = profile is not None and sensor != "bluetooth"

    blockers: list[str] = []
    next_steps: list[str] = []
    if not detected:
        blockers.append("当前 phyphox 实验没有识别到所选传感器输入。")
        next_steps.append("在手机 phyphox 中打开对应实验并重新检测。")
    elif profile is None:
        if sensor == "bluetooth":
            blockers.extend(
                [
                    "BLE 输入只被识别，尚未建立设备专用通道、单位与采样时钟协议。",
                    "没有可信通道协议时不能采集、分析或启动受限 Agent。",
                ]
            )
            next_steps.extend(
                [
                    "先在 phyphox 中确认 BLE 实验能稳定输出数值缓冲区。",
                    "登记设备型号、通道语义、单位、采样率和可复现零点后再接入。",
                ]
            )
        else:
            blockers.append("已发现传感器线索，但 /config 尚不足以解析可信通道映射。")
            next_steps.append("切换到 phyphox 官方原始传感器实验并重新检测。")
    else:
        next_steps.append("通道与单位 Profile 已解析，可进入确定性传感器实验台。")
        if can_analyze:
            next_steps.append("分析器可处理单条记录；实验 Agent 仍需专用协议与 Harness。")

    if capability is not None:
        blockers.extend(item for item in capability.limitations if item not in blockers)

    return SensorCapabilityCheck(
        sensor=sensor,
        status="profile_ready" if profile is not None else "detected" if detected else "not_detected",
        experiment_title=probe.experiment_title,
        measuring=probe.measuring,
        detected_sensors=list(probe.detected_sensors),
        available_buffers=list(probe.available_buffers),
        export_buffers=list(probe.export_buffers),
        profile=profile,
        analyzer_id=analyzer_id,
        analyzer_maturity=maturity,
        can_capture=can_capture,
        can_analyze=can_analyze,
        blockers=blockers,
        next_steps=next_steps,
        config_sha256=probe.config_sha256,
        privacy_statement=(
            "报告不返回手机 IP、BLE 地址、配对标识或原始数值；缓冲区名称只用于本次能力检查。"
        ),
    )
