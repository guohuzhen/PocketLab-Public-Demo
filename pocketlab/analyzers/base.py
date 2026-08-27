from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pocketlab.sensor_models import SensorAnalysis, SensorKind, SensorRecordingUpload


@dataclass(frozen=True)
class TimeSeriesQuality:
    timestamps_s: np.ndarray
    duration_s: float
    sampling_rate_hz: float
    jitter_ratio: float
    max_gap_ratio: float


class SensorAnalyzer(Protocol):
    sensor: SensorKind
    analyzer_id: str
    analyzer_version: str

    def analyze(self, upload: SensorRecordingUpload) -> SensorAnalysis: ...


def time_series_quality(
    upload: SensorRecordingUpload,
    *,
    minimum_samples: int,
    minimum_duration_s: float,
) -> TimeSeriesQuality:
    if len(upload.samples) < minimum_samples:
        raise ValueError(
            f"{upload.sensor} requires at least {minimum_samples} samples; "
            f"received {len(upload.samples)}"
        )
    timestamps = np.asarray([sample.timestamp_ms for sample in upload.samples], dtype=np.float64)
    elapsed = (timestamps - timestamps[0]) / 1000.0
    deltas = np.diff(elapsed)
    if not np.isfinite(elapsed).all() or np.any(deltas <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    duration = float(elapsed[-1])
    if duration < minimum_duration_s:
        raise ValueError(
            f"{upload.sensor} recording is too short; need {minimum_duration_s:.2f} seconds"
        )
    median_dt = float(np.median(deltas))
    return TimeSeriesQuality(
        timestamps_s=elapsed,
        duration_s=duration,
        sampling_rate_hz=1.0 / median_dt,
        jitter_ratio=float(np.std(deltas) / median_dt),
        max_gap_ratio=float(np.max(deltas) / median_dt),
    )


def channel_array(upload: SensorRecordingUpload, channel: str, unit: str) -> np.ndarray:
    definition = upload.channels.get(channel)
    if definition is None:
        raise ValueError(f"{upload.sensor} recording is missing channel {channel!r}")
    if definition.unit != unit:
        raise ValueError(
            f"{upload.sensor}.{channel} must use {unit!r}; received {definition.unit!r}"
        )
    values = np.asarray([sample.values[channel] for sample in upload.samples], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{upload.sensor}.{channel} contains non-finite values")
    return values


def confidence_from_quality(
    quality: TimeSeriesQuality,
    warnings: list[str],
    *,
    high_jitter: float = 0.20,
    severe_gap: float = 3.0,
) -> str:
    if quality.max_gap_ratio > severe_gap:
        warnings.append("记录中存在明显采样断流，时间相关指标不可靠。")
        return "low"
    if quality.jitter_ratio > high_jitter:
        warnings.append("采样时间间隔波动较大。")
        return "medium"
    return "high"
