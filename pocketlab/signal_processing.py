from __future__ import annotations

import math

import numpy as np

from pocketlab.schemas import AccelerationSample, VibrationAnalysis


def analyze_acceleration(samples: list[AccelerationSample]) -> VibrationAnalysis:
    """Analyze a short stationary-phone vibration recording.

    Timestamps may be mildly irregular. Each axis is interpolated onto a uniform grid before
    applying a Hann-window FFT. The highest-variance axis is used instead of vector magnitude so
    gravity on another axis does not suppress or double the measured vibration frequency.
    """

    if len(samples) < 64:
        raise ValueError("At least 64 acceleration samples are required")

    timestamps_ms = np.asarray([sample.timestamp_ms for sample in samples], dtype=np.float64)
    axes = np.asarray([[sample.x, sample.y, sample.z] for sample in samples], dtype=np.float64)
    if not np.isfinite(timestamps_ms).all() or not np.isfinite(axes).all():
        raise ValueError("Samples contain non-finite values")

    elapsed_s = (timestamps_ms - timestamps_ms[0]) / 1000.0
    deltas = np.diff(elapsed_s)
    if np.any(deltas <= 0):
        raise ValueError("Timestamps must be strictly increasing")

    median_dt = float(np.median(deltas))
    sampling_rate = 1.0 / median_dt
    max_gap_ratio = float(np.max(deltas) / median_dt)
    duration = float(elapsed_s[-1])
    if duration <= 0.25:
        raise ValueError("Recording is too short for vibration analysis")
    if sampling_rate < 8:
        raise ValueError("Sampling rate is too low; at least 8 Hz is required")

    jitter_ratio = float(np.std(deltas) / median_dt)
    uniform_t = np.linspace(0.0, duration, len(samples), dtype=np.float64)
    uniform_axes = np.column_stack(
        [np.interp(uniform_t, elapsed_s, axes[:, index]) for index in range(3)]
    )
    centered = uniform_axes - np.mean(uniform_axes, axis=0, keepdims=True)

    axis_index = int(np.argmax(np.std(centered, axis=0)))
    axis_names = ("x", "y", "z")
    signal = centered[:, axis_index]
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak_to_peak = float(np.ptp(signal))

    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window))
    frequencies = np.fft.rfftfreq(len(signal), d=duration / (len(signal) - 1))

    minimum_frequency = max(0.5, 2.0 / duration)
    eligible = np.flatnonzero(frequencies >= minimum_frequency)
    if eligible.size == 0 or float(np.max(spectrum[eligible])) <= 1e-12:
        dominant_frequency = 0.0
        snr_db = 0.0
    else:
        peak_index = int(eligible[np.argmax(spectrum[eligible])])
        dominant_frequency = float(frequencies[peak_index])
        noise_bins = spectrum[eligible].copy()
        local_peak = np.flatnonzero(np.abs(eligible - peak_index) <= 1)
        noise_bins[local_peak] = np.nan
        noise_floor = float(np.nanmedian(noise_bins))
        peak_amplitude = float(spectrum[peak_index])
        snr_db = 20.0 * math.log10(peak_amplitude / max(noise_floor, 1e-12))

    warnings: list[str] = []
    if jitter_ratio > 0.20:
        warnings.append("采样时间间隔波动较大，频率估计可信度下降。")
    if max_gap_ratio > 2.5:
        warnings.append("记录中存在明显采样断流，建议固定手机并重新采集。")
    if duration < 3.0:
        warnings.append("记录短于 3 秒，低频分辨率有限。")
    if rms < 0.01:
        warnings.append("振动幅值接近手机传感器噪声区间。")
    nyquist = sampling_rate / 2.0
    recommended_frequency_limit = sampling_rate / 5.0
    aliasing_risk = dominant_frequency > recommended_frequency_limit
    if aliasing_risk:
        warnings.append(
            "主频超过采样率的 1/5，无法可靠排除高频混叠；请提高采样率或使用抗混叠滤波。"
        )

    if snr_db >= 18 and jitter_ratio <= 0.12 and duration >= 3.0:
        confidence = "high"
    elif snr_db >= 9 and jitter_ratio <= 0.25:
        confidence = "medium"
    else:
        confidence = "low"
    if aliasing_risk and confidence == "high":
        confidence = "medium"
    if max_gap_ratio > 2.5:
        confidence = "low"

    return VibrationAnalysis(
        sample_count=len(samples),
        duration_s=round(duration, 4),
        sampling_rate_hz=round(sampling_rate, 3),
        sampling_jitter_ratio=round(jitter_ratio, 4),
        max_sampling_gap_ratio=round(max_gap_ratio, 4),
        nyquist_frequency_hz=round(nyquist, 3),
        recommended_frequency_limit_hz=round(recommended_frequency_limit, 3),
        selected_axis=axis_names[axis_index],
        rms_acceleration_m_s2=round(rms, 6),
        peak_to_peak_m_s2=round(peak_to_peak, 6),
        dominant_frequency_hz=round(dominant_frequency, 4),
        spectral_snr_db=round(snr_db, 3),
        confidence=confidence,
        warnings=warnings,
    )
