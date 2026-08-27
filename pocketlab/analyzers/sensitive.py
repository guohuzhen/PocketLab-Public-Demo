from __future__ import annotations

import numpy as np

from pocketlab.analyzers.base import (
    channel_array,
    confidence_from_quality,
    time_series_quality,
)
from pocketlab.sensor_models import (
    AnalysisMetric,
    SensorAnalysis,
    SensorCapability,
    SensorRecordingUpload,
)

_EARTH_MEAN_RADIUS_M = 6_371_008.8
# PocketLab's supported real-world exploration envelope. This is deliberately an
# application safety gate, not a phyphox accuracy or platform guarantee.
_MAX_SUPPORTED_LOCATION_SPEED_M_S = 350.0


def _cap_confidence(confidence: str, maximum: str) -> str:
    ranks = {"low": 0, "medium": 1, "high": 2}
    return confidence if ranks[confidence] <= ranks[maximum] else maximum


def _has_repeated_upper_plateau(values: np.ndarray, *, minimum_run: int = 3) -> bool:
    """Flag a repeated derived upper plateau without claiming waveform clipping."""

    if values.size < minimum_run or float(np.ptp(values)) == 0.0:
        return False
    at_maximum = np.isclose(values, float(np.max(values)), rtol=1e-9, atol=1e-12)
    longest_run = 0
    current_run = 0
    for is_maximum in at_maximum:
        current_run = current_run + 1 if bool(is_maximum) else 0
        longest_run = max(longest_run, current_run)
    return longest_run >= minimum_run


def _haversine_distances_m(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    delta_lat = np.diff(lat)
    delta_lon = np.diff(lon)
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(delta_lon / 2.0) ** 2
    )
    central_angle = 2.0 * np.arctan2(
        np.sqrt(np.clip(haversine, 0.0, 1.0)),
        np.sqrt(np.clip(1.0 - haversine, 0.0, 1.0)),
    )
    return _EARTH_MEAN_RADIUS_M * central_angle


def _haversine_displacement_m(lat_deg: np.ndarray, lon_deg: np.ndarray) -> float:
    endpoint_lat = np.asarray([lat_deg[0], lat_deg[-1]], dtype=np.float64)
    endpoint_lon = np.asarray([lon_deg[0], lon_deg[-1]], dtype=np.float64)
    return float(_haversine_distances_m(endpoint_lat, endpoint_lon)[0])


class MicrophoneAnalyzer:
    """Analyze privacy-reduced microphone features, never raw audio waveforms."""

    sensor = "microphone"
    analyzer_id = "pocketlab.microphone.derived.v1"
    analyzer_version = "1.0.0"

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=["level_db", "amplitude"],
            accepted_units={"level_db": ["dB_relative"], "amplitude": ["a.u."]},
            limitations=[
                "只接受单通道派生级别或振幅序列，不接受或保留原始音频。",
                "dB_relative 不是校准声压级，不能解释为 SPL。",
                "派生序列只能提示疑似上限平台，不能确认原始波形削顶。",
                "分析前需要明确的隐私确认。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        if upload.sensor != self.sensor:
            raise ValueError(f"{self.analyzer_id} cannot analyze {upload.sensor}")
        if not upload.provenance.privacy_acknowledged:
            raise ValueError("microphone recordings require privacy_acknowledged=true")

        channel_names = set(upload.channels)
        if channel_names == {"level_db"}:
            channel_name = "level_db"
            unit = "dB_relative"
        elif channel_names == {"amplitude"}:
            channel_name = "amplitude"
            unit = "a.u."
        else:
            raise ValueError(
                "microphone recording must contain exactly one derived channel: "
                "level_db or amplitude; raw audio channels are forbidden"
            )

        quality = time_series_quality(upload, minimum_samples=4, minimum_duration_s=0.05)
        values = channel_array(upload, channel_name, unit)
        if channel_name == "amplitude" and np.any(values < 0):
            raise ValueError("microphone.amplitude must be non-negative")

        warnings: list[str] = []
        confidence = confidence_from_quality(quality, warnings)
        if _has_repeated_upper_plateau(values):
            warnings.append(
                "派生级别出现重复上限平台，标记为疑似饱和；"
                "仅凭派生数据无法确认原始波形削顶。"
            )
            confidence = _cap_confidence(confidence, "medium")

        if channel_name == "level_db":
            metrics = [
                AnalysisMetric(
                    key="mean_relative_level_db",
                    label="平均相对级别",
                    value=float(np.mean(values)),
                    unit="dB_relative",
                ),
                AnalysisMetric(
                    key="peak_relative_level_db",
                    label="峰值相对级别",
                    value=float(np.max(values)),
                    unit="dB_relative",
                ),
                AnalysisMetric(
                    key="relative_level_span_db",
                    label="相对级别范围",
                    value=float(np.ptp(values)),
                    unit="dB_relative",
                ),
            ]
        else:
            rms = float(np.sqrt(np.mean(np.square(values))))
            peak = float(np.max(values))
            metrics = [
                AnalysisMetric(
                    key="rms_amplitude_au",
                    label="均方根派生振幅",
                    value=rms,
                    unit="a.u.",
                ),
                AnalysisMetric(
                    key="peak_amplitude_au",
                    label="峰值派生振幅",
                    value=peak,
                    unit="a.u.",
                ),
                AnalysisMetric(
                    key="crest_factor",
                    label="峰值因数",
                    value=peak / rms if rms > 0.0 else 0.0,
                    unit="1",
                ),
            ]

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=len(upload.samples),
            duration_s=quality.duration_s,
            sampling_rate_hz=quality.sampling_rate_hz,
            sampling_jitter_ratio=quality.jitter_ratio,
            max_sampling_gap_ratio=quality.max_gap_ratio,
            confidence=confidence,
            warnings=warnings,
            metrics=metrics,
        )


class LocationAnalyzer:
    """Analyze a consented trajectory while never returning absolute coordinates."""

    sensor = "location"
    analyzer_id = "pocketlab.location.haversine.v1"
    analyzer_version = "1.0.0"

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=[
                "lat",
                "lon",
                "accuracy",
                "speed",
                "status",
                "altitude",
                "vertical_accuracy",
            ],
            accepted_units={
                "lat": ["deg"],
                "lon": ["deg"],
                "accuracy": ["m"],
                "speed": ["m/s"],
                "status": ["state"],
                "altitude": ["m"],
                "vertical_accuracy": ["m"],
            },
            limitations=[
                "需要 lat/lon 和隐私确认；不输出绝对坐标指标。",
                "status 与 accuracy 可选；缺失时置信度不会评为 high。",
                "350 m/s 是 PocketLab 支持范围门禁，不是 phyphox 精度保证。",
                "轨迹指标不等同于测绘级距离。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        if upload.sensor != self.sensor:
            raise ValueError(f"{self.analyzer_id} cannot analyze {upload.sensor}")
        if not upload.provenance.privacy_acknowledged:
            raise ValueError("location recordings require privacy_acknowledged=true")

        required_channels = {"lat", "lon"}
        optional_channels = {
            "accuracy",
            "speed",
            "status",
            "altitude",
            "vertical_accuracy",
        }
        channel_names = set(upload.channels)
        if not required_channels.issubset(channel_names):
            raise ValueError("location recording requires lat and lon channels")
        unexpected = channel_names - required_channels - optional_channels
        if unexpected:
            raise ValueError(f"unsupported location channels: {sorted(unexpected)}")

        quality = time_series_quality(upload, minimum_samples=2, minimum_duration_s=0.05)
        lat = channel_array(upload, "lat", "deg")
        lon = channel_array(upload, "lon", "deg")
        if np.any((lat < -90.0) | (lat > 90.0)):
            raise ValueError("location.lat must be within [-90, 90] degrees")
        if np.any((lon < -180.0) | (lon > 180.0)):
            raise ValueError("location.lon must be within [-180, 180] degrees")

        accuracy: np.ndarray | None = None
        if "accuracy" in channel_names:
            accuracy = channel_array(upload, "accuracy", "m")
            if np.any(accuracy <= 0.0):
                raise ValueError("location.accuracy must be positive")

        speed: np.ndarray | None = None
        if "speed" in channel_names:
            speed = channel_array(upload, "speed", "m/s")
            if np.any(speed < 0.0):
                raise ValueError("location.speed must be non-negative")
            if np.any(speed > _MAX_SUPPORTED_LOCATION_SPEED_M_S):
                raise ValueError("reported location speed exceeds PocketLab's supported range")

        if "status" in channel_names:
            status = channel_array(upload, "status", "state")
            if not np.all(np.isin(status, [1.0, 2.0])):
                raise ValueError(
                    "location.status must be active (1 or 2) for every accepted sample"
                )

        altitude: np.ndarray | None = None
        if "altitude" in channel_names:
            altitude = channel_array(upload, "altitude", "m")
        vertical_accuracy: np.ndarray | None = None
        if "vertical_accuracy" in channel_names:
            vertical_accuracy = channel_array(upload, "vertical_accuracy", "m")
            if np.any(vertical_accuracy <= 0.0):
                raise ValueError("location.vertical_accuracy must be positive")

        segment_distances = _haversine_distances_m(lat, lon)
        segment_durations = np.diff(quality.timestamps_s)
        uncertainty = (
            accuracy[:-1] + accuracy[1:]
            if accuracy is not None
            else np.zeros_like(segment_distances)
        )
        uncertainty_adjusted_speed = (
            np.maximum(segment_distances - uncertainty, 0.0) / segment_durations
        )
        if np.any(uncertainty_adjusted_speed > _MAX_SUPPORTED_LOCATION_SPEED_M_S):
            raise ValueError("location trajectory contains an anomalous coordinate jump")

        trajectory_distance = float(np.sum(segment_distances))
        displacement = _haversine_displacement_m(lat, lon)
        warnings: list[str] = []
        confidence = confidence_from_quality(quality, warnings)

        if "status" not in channel_names:
            warnings.append("记录未提供定位 status，无法验证所有点均来自活跃定位状态。")
            confidence = _cap_confidence(confidence, "medium")
        elif np.any(status == 2.0):
            warnings.append(
                "部分定位点的 status=2；高度可能采用 WGS84 椭球基准，"
                "不可与海拔或其他高度基准直接混用。"
            )
        if accuracy is None:
            warnings.append("记录未提供水平 accuracy，无法量化轨迹位置不确定性。")
            confidence = _cap_confidence(confidence, "medium")
        else:
            if displacement <= float(accuracy[0] + accuracy[-1]):
                warnings.append("端点位移未超过端点水平不确定性之和，不宜宣称发生了净移动。")
                confidence = "low"
            if np.all(segment_distances <= uncertainty):
                warnings.append("各段轨迹变化均未超过相邻点水平不确定性之和。")
                confidence = "low"

        metrics = [
            AnalysisMetric(
                key="trajectory_distance_m",
                label="轨迹长度",
                value=trajectory_distance,
                unit="m",
            ),
            AnalysisMetric(
                key="displacement_m",
                label="端点位移",
                value=displacement,
                unit="m",
            ),
            AnalysisMetric(
                key="average_path_speed_m_s",
                label="按轨迹计算的平均速率",
                value=trajectory_distance / quality.duration_s,
                unit="m/s",
            ),
            AnalysisMetric(
                key="path_efficiency_ratio",
                label="端点位移与轨迹长度比",
                value=displacement / trajectory_distance if trajectory_distance > 0.0 else 0.0,
                unit="1",
            ),
        ]
        if accuracy is not None:
            metrics.extend(
                [
                    AnalysisMetric(
                        key="median_horizontal_accuracy_m",
                        label="水平不确定性中位数",
                        value=float(np.median(accuracy)),
                        unit="m",
                    ),
                    AnalysisMetric(
                        key="max_horizontal_accuracy_m",
                        label="最大水平不确定性",
                        value=float(np.max(accuracy)),
                        unit="m",
                    ),
                ]
            )
        if speed is not None:
            metrics.append(
                AnalysisMetric(
                    key="mean_reported_speed_m_s",
                    label="设备报告平均速率",
                    value=float(np.mean(speed)),
                    unit="m/s",
                )
            )
        if altitude is not None:
            metrics.extend(
                [
                    AnalysisMetric(
                        key="altitude_change_m",
                        label="记录端点高度变化",
                        value=float(altitude[-1] - altitude[0]),
                        unit="m",
                    ),
                    AnalysisMetric(
                        key="altitude_span_m",
                        label="记录高度范围",
                        value=float(np.ptp(altitude)),
                        unit="m",
                    ),
                ]
            )
            warnings.append(
                "定位高度只做同一记录内的描述；基准与垂直误差未核验时不得当作绝对海拔。"
            )
            confidence = _cap_confidence(confidence, "medium")
        if vertical_accuracy is not None:
            metrics.append(
                AnalysisMetric(
                    key="median_vertical_accuracy_m",
                    label="垂直不确定性中位数",
                    value=float(np.median(vertical_accuracy)),
                    unit="m",
                )
            )

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=len(upload.samples),
            duration_s=quality.duration_s,
            sampling_rate_hz=quality.sampling_rate_hz,
            sampling_jitter_ratio=quality.jitter_ratio,
            max_sampling_gap_ratio=quality.max_gap_ratio,
            confidence=confidence,
            warnings=warnings,
            metrics=metrics,
        )
