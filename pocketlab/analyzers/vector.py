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

_MINIMUM_SAMPLES = 20
_MINIMUM_DURATION_S = 1.0


def _metric(key: str, label: str, value: float, unit: str) -> AnalysisMetric:
    return AnalysisMetric(
        key=key,
        label=label,
        value=round(float(value), 6),
        unit=unit,
    )


def _vector_channels(
    upload: SensorRecordingUpload,
    *,
    expected_sensor: str,
    unit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if upload.sensor != expected_sensor:
        raise ValueError(
            f"{expected_sensor} analyzer cannot analyze {upload.sensor} recordings"
        )
    return (
        channel_array(upload, "x", unit),
        channel_array(upload, "y", unit),
        channel_array(upload, "z", unit),
    )


class GyroscopeAnalyzer:
    sensor = "gyroscope"
    analyzer_id = "pocketlab.gyroscope.v1"
    analyzer_version = "1.0.0"

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=["x", "y", "z"],
            accepted_units={
                "x": ["rad/s"],
                "y": ["rad/s"],
                "z": ["rad/s"],
            },
            limitations=[
                "只接受 phyphox 原生 rad/s，不自动猜测或换算 deg/s。",
                "稳定非零均值只是静止零偏候选；未提供静止基线时不能解释为角位移。",
                "当前不对角速度积分，避免在缺少零偏校正时累积漂移。",
                "尚未通过真实手机 3 场景 × 2 条件 × 3 重复的 Agent 门禁。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        quality = time_series_quality(
            upload,
            minimum_samples=_MINIMUM_SAMPLES,
            minimum_duration_s=_MINIMUM_DURATION_S,
        )
        x, y, z = _vector_channels(
            upload,
            expected_sensor=self.sensor,
            unit="rad/s",
        )
        vectors = np.column_stack((x, y, z))
        angular_speed = np.linalg.norm(vectors, axis=1)
        mean_vector = np.mean(vectors, axis=0)
        component_std = np.std(vectors, axis=0)
        bias_candidate = float(np.linalg.norm(mean_vector))
        component_noise = float(np.linalg.norm(component_std))

        warnings: list[str] = []
        confidence = confidence_from_quality(quality, warnings)
        stable_nonzero_mean = bias_candidate >= 0.02 and component_noise <= 0.01
        if stable_nonzero_mean:
            warnings.append(
                "记录呈稳定非零角速度；若本次是静止基线，该值是陀螺仪零偏候选，"
                "在校正前不得积分为角位移。"
            )
            if confidence == "high":
                confidence = "medium"

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=len(upload.samples),
            duration_s=round(quality.duration_s, 6),
            sampling_rate_hz=round(quality.sampling_rate_hz, 6),
            sampling_jitter_ratio=round(quality.jitter_ratio, 6),
            max_sampling_gap_ratio=round(quality.max_gap_ratio, 6),
            confidence=confidence,
            warnings=warnings,
            metrics=[
                _metric(
                    "mean_angular_speed_rad_s",
                    "平均角速度模长",
                    np.mean(angular_speed),
                    "rad/s",
                ),
                _metric(
                    "angular_speed_std_rad_s",
                    "角速度模长标准差",
                    np.std(angular_speed),
                    "rad/s",
                ),
                _metric(
                    "peak_angular_speed_rad_s",
                    "峰值角速度模长",
                    np.max(angular_speed),
                    "rad/s",
                ),
                _metric(
                    "stationary_bias_candidate_rad_s",
                    "静止零偏候选",
                    bias_candidate,
                    "rad/s",
                ),
                _metric("mean_x_rad_s", "X 轴平均角速度", mean_vector[0], "rad/s"),
                _metric("mean_y_rad_s", "Y 轴平均角速度", mean_vector[1], "rad/s"),
                _metric("mean_z_rad_s", "Z 轴平均角速度", mean_vector[2], "rad/s"),
            ],
        )


class MagnetometerAnalyzer:
    sensor = "magnetometer"
    analyzer_id = "pocketlab.magnetometer.v1"
    analyzer_version = "1.0.0"

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=["x", "y", "z", "accuracy"],
            accepted_units={
                "x": ["uT"],
                "y": ["uT"],
                "z": ["uT"],
                "accuracy": ["state"],
            },
            limitations=[
                "只接受 ASCII 单位 uT；不自动猜测或换算 T、mT、µT 或 μT。",
                "未记录手机朝向和磁力计校准状态时，只能做同设备同方向的相对比较。",
                "时间序列突变表示局部磁场变化候选，不能单独识别物体或证明因果。",
                "尚未通过真实手机 3 场景 × 2 条件 × 3 重复的 Agent 门禁。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        quality = time_series_quality(
            upload,
            minimum_samples=_MINIMUM_SAMPLES,
            minimum_duration_s=_MINIMUM_DURATION_S,
        )
        x, y, z = _vector_channels(
            upload,
            expected_sensor=self.sensor,
            unit="uT",
        )
        vectors = np.column_stack((x, y, z))
        field_magnitude = np.linalg.norm(vectors, axis=1)
        mean_vector = np.mean(vectors, axis=0)
        median_magnitude = float(np.median(field_magnitude))
        max_deviation = float(np.max(np.abs(field_magnitude - median_magnitude)))
        anomaly_threshold = max(10.0, abs(median_magnitude) * 0.25)

        warnings: list[str] = []
        confidence = confidence_from_quality(quality, warnings)
        if "accuracy" in upload.channels:
            accuracy = channel_array(upload, "accuracy", "state")
            if np.any(~np.isin(accuracy, [-1.0, 0.0, 1.0, 2.0, 3.0])):
                raise ValueError("magnetometer.accuracy contains an unknown state code")
            if np.any(accuracy < 2.0):
                warnings.append(
                    "磁力计 accuracy 状态低于 medium；绝对场强与方向结论不可靠，建议校准后重测。"
                )
                confidence = "low"
        else:
            warnings.append("记录未提供磁力计 accuracy 状态，只能做有边界的相对变化分析。")
            if confidence == "high":
                confidence = "medium"
        if max_deviation >= anomaly_threshold:
            warnings.append(
                "磁场模长相对中位数出现明显突变，记录中存在局部磁场异常候选；"
                "请保持手机方向不变并重复扫描。"
            )

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=len(upload.samples),
            duration_s=round(quality.duration_s, 6),
            sampling_rate_hz=round(quality.sampling_rate_hz, 6),
            sampling_jitter_ratio=round(quality.jitter_ratio, 6),
            max_sampling_gap_ratio=round(quality.max_gap_ratio, 6),
            confidence=confidence,
            warnings=warnings,
            metrics=[
                _metric(
                    "mean_field_magnitude_ut",
                    "平均磁场模长",
                    np.mean(field_magnitude),
                    "uT",
                ),
                _metric(
                    "field_magnitude_std_ut",
                    "磁场模长标准差",
                    np.std(field_magnitude),
                    "uT",
                ),
                _metric(
                    "field_peak_to_peak_ut",
                    "磁场模长峰峰值",
                    np.ptp(field_magnitude),
                    "uT",
                ),
                _metric(
                    "max_field_deviation_ut",
                    "相对中位数最大偏差",
                    max_deviation,
                    "uT",
                ),
                _metric("mean_x_ut", "X 轴平均磁场", mean_vector[0], "uT"),
                _metric("mean_y_ut", "Y 轴平均磁场", mean_vector[1], "uT"),
                _metric("mean_z_ut", "Z 轴平均磁场", mean_vector[2], "uT"),
            ],
        )
