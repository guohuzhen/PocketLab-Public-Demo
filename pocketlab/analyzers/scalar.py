from __future__ import annotations

from typing import Literal

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
    SensorKind,
    SensorRecordingUpload,
)

Confidence = Literal["low", "medium", "high"]


def _require_sensor(upload: SensorRecordingUpload, expected: SensorKind) -> None:
    if upload.sensor != expected:
        raise ValueError(f"{expected} analyzer cannot analyze {upload.sensor} data")


def _lower_confidence(confidence: Confidence, ceiling: Confidence) -> Confidence:
    rank = {"low": 0, "medium": 1, "high": 2}
    return confidence if rank[confidence] <= rank[ceiling] else ceiling


def _metric(key: str, label: str, value: float, unit: str) -> AnalysisMetric:
    return AnalysisMetric(key=key, label=label, value=float(value), unit=unit)


def _upper_plateau_fraction(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    tolerance = max(abs(maximum) * 1e-6, 1e-9)
    return float(np.mean(np.isclose(values, maximum, rtol=0.0, atol=tolerance)))


def _minimum_positive_step(values: np.ndarray) -> float:
    unique = np.unique(values)
    if unique.size < 2:
        return 0.0
    positive_steps = np.diff(unique)
    return float(np.min(positive_steps[positive_steps > 0]))


class LightAnalyzer:
    sensor: SensorKind = "light"
    analyzer_id = "pocketlab.light.v2"
    analyzer_version = "2.0.0"

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=["illuminance"],
            accepted_units={"illuminance": ["lx"]},
            limitations=[
                "没有参考照度计时只报告相对照度和变化，不宣称绝对校准准确度。",
                "观测上限平台只能提示可能饱和；数据契约目前不包含设备量程。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        _require_sensor(upload, self.sensor)
        quality = time_series_quality(upload, minimum_samples=8, minimum_duration_s=1.0)
        values = channel_array(upload, "illuminance", "lx")
        if np.any(values < 0):
            raise ValueError("light.illuminance cannot be negative")

        warnings: list[str] = []
        confidence: Confidence = confidence_from_quality(quality, warnings)  # type: ignore[assignment]
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        mean = float(np.mean(values))
        coefficient_of_variation = float(np.std(values) / mean) if mean > 0 else 0.0
        plateau_fraction = _upper_plateau_fraction(values)
        span = float(np.ptp(values))

        if span == 0:
            warnings.append("照度记录完全不变；只能报告该平台值，不能验证变化响应。")
            confidence = _lower_confidence(confidence, "medium")
        elif plateau_fraction >= 0.50:
            warnings.append(
                "大量样本停在观测上限，可能是传感器饱和或量化平台；缺少设备量程时不能确认。"
            )
            confidence = _lower_confidence(confidence, "medium")
        if float(np.max(values)) > 1_000_000:
            warnings.append("观测照度异常高，请核对量程、单位和传感器是否被强光直射。")
            confidence = _lower_confidence(confidence, "medium")

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=len(values),
            duration_s=quality.duration_s,
            sampling_rate_hz=quality.sampling_rate_hz,
            sampling_jitter_ratio=quality.jitter_ratio,
            max_sampling_gap_ratio=quality.max_gap_ratio,
            confidence=confidence,
            warnings=warnings,
            metrics=[
                _metric("median_illuminance_lx", "照度中位数", median, "lx"),
                _metric("minimum_illuminance_lx", "最低照度", np.min(values), "lx"),
                _metric("maximum_illuminance_lx", "最高照度", np.max(values), "lx"),
                _metric("illuminance_iqr_lx", "照度四分位距", q75 - q25, "lx"),
                _metric(
                    "coefficient_of_variation_ratio",
                    "照度变异系数",
                    coefficient_of_variation,
                    "ratio",
                ),
                _metric(
                    "minimum_quantization_step_lx",
                    "最小观测量化步长",
                    _minimum_positive_step(values),
                    "lx",
                ),
                _metric(
                    "upper_plateau_fraction",
                    "观测上限平台比例",
                    plateau_fraction,
                    "ratio",
                ),
            ],
        )


class PressureAnalyzer:
    sensor: SensorKind = "pressure"
    analyzer_id = "pocketlab.pressure.v2"
    analyzer_version = "2.0.0"

    # International standard-atmosphere approximation used by phyphox's elevator experiment.
    _HEIGHT_SCALE_M = 44_330.0
    _PRESSURE_EXPONENT = 1.0 / 5.255

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=["pressure"],
            accepted_units={"pressure": ["hPa"]},
            limitations=[
                "相对高度采用标准大气近地面近似，短时温度、天气、HVAC 和传感器漂移仍会影响结果。",
                "不提供未经校准的绝对海拔，也不把压力趋势单独归因为真实垂直运动。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        _require_sensor(upload, self.sensor)
        quality = time_series_quality(upload, minimum_samples=8, minimum_duration_s=2.0)
        values = channel_array(upload, "pressure", "hPa")
        if np.any((values < 100.0) | (values > 1_200.0)):
            raise ValueError(
                "pressure.pressure is outside the supported 100-1200 hPa range; "
                "check whether Pa was supplied instead of hPa"
            )

        warnings = [
            "相对高度采用标准大气近地面近似；它不是绝对海拔，且需用受控对照排除天气、HVAC 与传感器漂移。"
        ]
        confidence: Confidence = confidence_from_quality(quality, warnings)  # type: ignore[assignment]
        window = max(1, min(len(values) // 5, 10))
        reference_pressure = float(np.median(values[:window]))
        end_pressure = float(np.median(values[-window:]))
        relative_height = self._HEIGHT_SCALE_M * (
            1.0 - (end_pressure / reference_pressure) ** self._PRESSURE_EXPONENT
        )
        slope_hpa_per_s = float(
            np.polyfit(quality.timestamps_s, values, deg=1)[0]
        )
        trend_hpa_per_min = slope_hpa_per_s * 60.0

        if abs(trend_hpa_per_min) >= 0.15:
            warnings.append(
                "记录存在明显线性压力趋势；它可能来自高度变化，也可能来自环境或传感器漂移，不能单独归因。"
            )
            confidence = _lower_confidence(confidence, "medium")
        if float(np.ptp(values)) < 1e-4:
            warnings.append("压力记录几乎完全不变；只能作为静止平台或分辨率检查。")
            confidence = _lower_confidence(confidence, "medium")

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=len(values),
            duration_s=quality.duration_s,
            sampling_rate_hz=quality.sampling_rate_hz,
            sampling_jitter_ratio=quality.jitter_ratio,
            max_sampling_gap_ratio=quality.max_gap_ratio,
            confidence=confidence,
            warnings=warnings,
            metrics=[
                _metric("reference_pressure_hpa", "起始参考压力", reference_pressure, "hPa"),
                _metric("end_pressure_hpa", "结束参考压力", end_pressure, "hPa"),
                _metric(
                    "pressure_change_hpa",
                    "末段相对起始压力变化",
                    end_pressure - reference_pressure,
                    "hPa",
                ),
                _metric(
                    "relative_height_change_m",
                    "标准大气近似相对高度变化",
                    relative_height,
                    "m",
                ),
                _metric(
                    "pressure_trend_hpa_per_min",
                    "全程线性压力趋势",
                    trend_hpa_per_min,
                    "hPa/min",
                ),
                _metric("pressure_mad_hpa", "压力中位绝对偏差", _mad(values), "hPa"),
            ],
        )


class ProximityAnalyzer:
    sensor: SensorKind = "proximity"
    analyzer_id = "pocketlab.proximity.v2"
    analyzer_version = "2.0.0"

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=["distance"],
            accepted_units={"distance": ["cm"]},
            limitations=[
                "自动识别常量、二态和多级输出；二态 near/far 绝不解释为连续厘米距离。",
                "连续数值仍受目标材质、入射角和设备实现影响，不能当作通用测距仪。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        _require_sensor(upload, self.sensor)
        quality = time_series_quality(upload, minimum_samples=4, minimum_duration_s=0.2)
        values = channel_array(upload, "distance", "cm")
        if np.any(values < 0):
            raise ValueError("proximity.distance cannot be negative")

        warnings: list[str] = []
        confidence: Confidence = confidence_from_quality(quality, warnings)  # type: ignore[assignment]
        rounded = np.round(values, decimals=6)
        levels = np.unique(rounded)
        transitions = int(np.count_nonzero(np.diff(rounded)))
        is_binary = levels.size == 2
        is_continuous = levels.size > 2
        mode_code = 1.0 if is_binary else 2.0 if is_continuous else 0.0

        metrics = [
            _metric("observed_level_count", "观测离散级数", levels.size, "count"),
            _metric("signal_mode_code", "信号模式（0常量/1二态/2多级）", mode_code, "code"),
            _metric("transition_count", "状态或数值转换次数", transitions, "count"),
            _metric("minimum_observed_cm", "最小观测值", np.min(values), "cm"),
            _metric("maximum_observed_cm", "最大观测值", np.max(values), "cm"),
        ]

        if levels.size == 1:
            warnings.append("记录只覆盖一个接近状态；无法判断传感器是二态还是连续输出。")
            confidence = _lower_confidence(confidence, "low")
        elif is_binary:
            near = float(levels[0])
            far = float(levels[1])
            threshold = (near + far) / 2.0
            near_fraction = float(np.mean(values < threshold))
            warnings.append(
                "该记录呈二态 near/far 输出；两个 cm 数值是状态编码，不得解释为连续物体距离。"
            )
            metrics.extend(
                [
                    _metric("near_state_value_cm", "near 状态编码值", near, "cm"),
                    _metric("far_state_value_cm", "far 状态编码值", far, "cm"),
                    _metric("near_state_fraction", "near 状态占比", near_fraction, "ratio"),
                ]
            )
        else:
            warnings.append(
                "该记录具有多个距离级，但连续数值仍需按设备、目标材质和角度验证，不能泛化为精密测距。"
            )
            metrics.extend(
                [
                    _metric("median_distance_cm", "观测距离中位数", np.median(values), "cm"),
                    _metric("distance_iqr_cm", "观测距离四分位距", _iqr(values), "cm"),
                ]
            )

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=len(values),
            duration_s=quality.duration_s,
            sampling_rate_hz=quality.sampling_rate_hz,
            sampling_jitter_ratio=quality.jitter_ratio,
            max_sampling_gap_ratio=quality.max_gap_ratio,
            confidence=confidence,
            warnings=warnings,
            metrics=metrics,
        )


def _mad(values: np.ndarray) -> float:
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _iqr(values: np.ndarray) -> float:
    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)
