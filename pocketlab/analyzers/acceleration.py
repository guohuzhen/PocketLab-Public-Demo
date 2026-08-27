from __future__ import annotations

from pocketlab.analyzers.base import channel_array
from pocketlab.schemas import AccelerationSample
from pocketlab.sensor_models import (
    AnalysisMetric,
    SensorAnalysis,
    SensorCapability,
    SensorRecordingUpload,
)
from pocketlab.signal_processing import analyze_acceleration


def _metric(key: str, label: str, value: float, unit: str) -> AnalysisMetric:
    return AnalysisMetric(key=key, label=label, value=float(value), unit=unit)


class AccelerationAnalyzer:
    """Strict v2 adapter around the established vibration signal pipeline.

    This analyzer deliberately remains a single-record descriptive analyzer.  Cadence,
    elevator-phase segmentation and causal vibration diagnosis each require their own
    evidence protocol and must not be inferred from these generic metrics alone.
    """

    sensor = "accelerometer"
    analyzer_id = "pocketlab.acceleration.v2"
    analyzer_version = "2.0.0"

    def capability(self) -> SensorCapability:
        return SensorCapability(
            sensor=self.sensor,
            maturity="analysis_ready",
            analyzer_id=self.analyzer_id,
            accepted_channels=["x", "y", "z"],
            accepted_units={
                "x": ["m/s^2"],
                "y": ["m/s^2"],
                "z": ["m/s^2"],
            },
            limitations=[
                "只接受三轴 m/s^2；不自动猜测或换算 g、Gal 等单位。",
                "通用频谱分析只描述单条记录，不能单独识别振动原因、路面或运动阶段。",
                "步频和电梯阶段需要各自的受限实验协议、质量门与跨记录比较工具。",
                "尚未通过真实手机 3 场景 × 2 条件 × 3 重复的统一 Agent 门禁。",
            ],
        )

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis:
        if upload.sensor != self.sensor:
            raise ValueError(
                f"{self.sensor} analyzer cannot analyze {upload.sensor} recordings"
            )
        if set(upload.channels) != {"x", "y", "z"}:
            raise ValueError("accelerometer recording must contain exactly x, y and z")

        x = channel_array(upload, "x", "m/s^2")
        y = channel_array(upload, "y", "m/s^2")
        z = channel_array(upload, "z", "m/s^2")
        legacy = analyze_acceleration(
            [
                AccelerationSample(
                    timestamp_ms=sample.timestamp_ms,
                    x=float(x[index]),
                    y=float(y[index]),
                    z=float(z[index]),
                )
                for index, sample in enumerate(upload.samples)
            ]
        )
        warnings = [
            (
                f"频谱使用动态方差最大的 {legacy.selected_axis.upper()} 轴；"
                "该选择不是手机空间姿态或竖直方向的证明。"
            ),
            *legacy.warnings,
        ]

        return SensorAnalysis(
            sensor=self.sensor,
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            sample_count=legacy.sample_count,
            duration_s=legacy.duration_s,
            sampling_rate_hz=legacy.sampling_rate_hz,
            sampling_jitter_ratio=legacy.sampling_jitter_ratio,
            max_sampling_gap_ratio=legacy.max_sampling_gap_ratio,
            confidence=legacy.confidence,
            warnings=warnings,
            metrics=[
                _metric(
                    "selected_axis_rms_m_s2",
                    "主动态轴 RMS",
                    legacy.rms_acceleration_m_s2,
                    "m/s^2",
                ),
                _metric(
                    "selected_axis_peak_to_peak_m_s2",
                    "主动态轴峰峰值",
                    legacy.peak_to_peak_m_s2,
                    "m/s^2",
                ),
                _metric(
                    "dominant_frequency_hz",
                    "主频",
                    legacy.dominant_frequency_hz,
                    "Hz",
                ),
                _metric(
                    "spectral_snr_db",
                    "频谱信噪比",
                    legacy.spectral_snr_db,
                    "dB",
                ),
                _metric(
                    "nyquist_frequency_hz",
                    "奈奎斯特频率",
                    legacy.nyquist_frequency_hz,
                    "Hz",
                ),
                _metric(
                    "recommended_frequency_limit_hz",
                    "建议可信频率上限",
                    legacy.recommended_frequency_limit_hz,
                    "Hz",
                ),
            ],
        )
