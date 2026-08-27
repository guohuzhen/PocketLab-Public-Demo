from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from pocketlab.analyzers.registry import analyze_sensor_recording
from pocketlab.public_replay_dataset import (
    load_public_replay_dataset,
    read_public_replay_recording,
    verify_public_source_files,
)
from pocketlab.public_sensor_agent_models import (
    PublicSensorComparison,
    PublicSensorComparisonMetric,
    PublicSensorEvidenceSnapshot,
    PublicSensorToolExecution,
)
from pocketlab.sensor_models import SensorRecordingUpload

DATASET_ID: Final = "accelerometer-nist-perfloc-pixel-20180516-v1"
CADENCE_PROTOCOL_ID: Final = "walking-cadence-public-exploration.v1"
ELEVATOR_PROTOCOL_ID: Final = "elevator-motion-public-exploration.v1"
VIBRATION_PROTOCOL_ID: Final = "vibration-response-public-exploration.v1"

CADENCE_RECORDING_IDS: Final = (
    "as4-stair-ascent-lower",
    "as4-stair-ascent-middle",
    "as4-stair-ascent-upper",
)
ELEVATOR_RECORDING_IDS: Final = (
    "as5-elevator-ascent-full",
    "as5-elevator-ascent-lower-half",
    "as5-elevator-ascent-upper-half",
)
VIBRATION_RECORDING_IDS: Final = (
    "as7-floor-stationary-anchor",
    "as7-handheld-transition",
)

ANALYZE_TOOL_ID: Final = "analyze_accelerometer_recording"
COMPARE_CADENCE_TOOL_ID: Final = "compare_stair_cadence_repeats"
SEGMENT_ELEVATOR_TOOL_ID: Final = "segment_elevator_motion_phases"
COMPARE_ELEVATOR_TOOL_ID: Final = "compare_elevator_phase_sequences"
COMPARE_VIBRATION_TOOL_ID: Final = "compare_acceleration_motion_states"

MIN_CADENCE_HZ: Final = 1.2
MAX_CADENCE_HZ: Final = 2.5
MIN_CADENCE_SNR_DB: Final = 18.0
MAX_CADENCE_CV: Final = 0.08

ELEVATOR_SMOOTHING_S: Final = 0.75
ELEVATOR_THRESHOLD_M_S2: Final = 0.25
ELEVATOR_MIN_PHASE_S: Final = 0.40
ELEVATOR_MIN_CRUISE_S: Final = 1.0
ELEVATOR_MIN_EXCURSION_M_S2: Final = 0.40

MAX_STATIONARY_RMS_M_S2: Final = 0.03
MIN_HANDHELD_RMS_M_S2: Final = 1.0
MIN_HANDHELD_PEAK_TO_PEAK_M_S2: Final = 8.0
MIN_MOTION_SEPARATION_RATIO: Final = 50.0

ProtocolId = Literal[
    "walking-cadence-public-exploration.v1",
    "elevator-motion-public-exploration.v1",
    "vibration-response-public-exploration.v1",
]


@dataclass(frozen=True)
class ElevatorPhaseSummary:
    recording_id: str
    baseline_m_s2: float
    acceleration_start_s: float
    acceleration_end_s: float
    acceleration_mean_excursion_m_s2: float
    cruise_duration_s: float
    deceleration_start_s: float
    deceleration_end_s: float
    deceleration_mean_excursion_m_s2: float
    quality_passed: bool


def _confidence_passed(value: str) -> bool:
    return value in {"medium", "high"}


def _allowed_ids(protocol_id: ProtocolId) -> tuple[str, ...]:
    return {
        CADENCE_PROTOCOL_ID: CADENCE_RECORDING_IDS,
        ELEVATOR_PROTOCOL_ID: ELEVATOR_RECORDING_IDS,
        VIBRATION_PROTOCOL_ID: VIBRATION_RECORDING_IDS,
    }[protocol_id]


def load_public_accelerometer_evidence(
    pack_dir: Path,
    protocol_id: ProtocolId,
    recording_ids: tuple[str, ...],
) -> tuple[
    tuple[PublicSensorEvidenceSnapshot, ...],
    PublicSensorComparison,
    tuple[PublicSensorToolExecution, ...],
]:
    allowed = set(_allowed_ids(protocol_id))
    if (
        not recording_ids
        or len(recording_ids) != len(set(recording_ids))
        or not set(recording_ids).issubset(allowed)
    ):
        raise ValueError("accelerometer route is outside the frozen protocol")

    manifest = load_public_replay_dataset(pack_dir)
    if (
        manifest.dataset_id != DATASET_ID
        or manifest.sensor != "accelerometer"
        or manifest.analyzer_id != "pocketlab.acceleration.v2"
        or manifest.analyzer_version != "2.0.0"
        or manifest.public_replay_status != "source_validated"
        or manifest.agent_ready
    ):
        raise ValueError("accelerometer public replay manifest contract changed")
    if manifest.source.doi is None:
        raise ValueError("accelerometer source DOI is required")
    verify_public_source_files(pack_dir, manifest)

    by_id = {item.recording_id: item for item in manifest.recordings}
    snapshots: list[PublicSensorEvidenceSnapshot] = []
    uploads: dict[str, SensorRecordingUpload] = {}
    tool_trace: list[PublicSensorToolExecution] = []
    for recording_id in recording_ids:
        recording = by_id.get(recording_id)
        if recording is None:
            raise ValueError("accelerometer recording is missing from the reviewed manifest")
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        analysis = analyze_sensor_recording(upload)
        if analysis != recording.reference_analysis:
            raise ValueError("accelerometer analyzer drifted from the frozen reference")
        uploads[recording_id] = upload
        evidence_id = f"accelerometer-{recording_id}"
        snapshots.append(
            PublicSensorEvidenceSnapshot(
                evidence_id=evidence_id,
                dataset_id=manifest.dataset_id,
                recording_id=recording_id,
                sensor="accelerometer",
                data_class=manifest.data_class,
                condition_label=recording.label,
                device_scope=(
                    f"{recording.device_alias}; public Android SensorEvent capture, "
                    "not phyphox"
                ),
                source_title=manifest.source.title,
                source_url=manifest.source.record_url,
                doi=manifest.source.doi,
                license_spdx=manifest.source.license_spdx,
                analysis=analysis,
                processing_disclosures=tuple(recording.processing_disclosures),
                claim_boundary=tuple(manifest.claim_boundary),
            )
        )
        tool_trace.append(
            PublicSensorToolExecution(
                sequence=len(tool_trace) + 1,
                tool_id=ANALYZE_TOOL_ID,
                evidence_ids=(evidence_id,),
                result_codes=(
                    f"confidence_{analysis.confidence}",
                    "analysis_reference_matched",
                ),
            )
        )

    evidence = tuple(snapshots)
    if protocol_id == CADENCE_PROTOCOL_ID:
        comparison = compare_public_cadence(evidence)
    elif protocol_id == ELEVATOR_PROTOCOL_ID:
        phase_summaries = tuple(
            segment_elevator_motion_phases(recording_id, uploads[recording_id])
            for recording_id in recording_ids
        )
        for snapshot, summary in zip(evidence, phase_summaries, strict=True):
            tool_trace.append(
                PublicSensorToolExecution(
                    sequence=len(tool_trace) + 1,
                    tool_id=SEGMENT_ELEVATOR_TOOL_ID,
                    evidence_ids=(snapshot.evidence_id,),
                    result_codes=(
                        "phase_sequence_passed"
                        if summary.quality_passed
                        else "phase_sequence_failed",
                    ),
                )
            )
        comparison = compare_public_elevator_phases(evidence, phase_summaries)
    else:
        comparison = compare_public_vibration_states(evidence)
    compare_tool_id = {
        CADENCE_PROTOCOL_ID: COMPARE_CADENCE_TOOL_ID,
        ELEVATOR_PROTOCOL_ID: COMPARE_ELEVATOR_TOOL_ID,
        VIBRATION_PROTOCOL_ID: COMPARE_VIBRATION_TOOL_ID,
    }[protocol_id]
    tool_trace.append(
        PublicSensorToolExecution(
            sequence=len(tool_trace) + 1,
            tool_id=compare_tool_id,
            evidence_ids=comparison.evidence_ids,
            result_codes=comparison.result_codes,
        )
    )
    return evidence, comparison, tuple(tool_trace)


def compare_public_cadence(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
) -> PublicSensorComparison:
    if not evidence or len(evidence) != len({item.evidence_id for item in evidence}):
        raise ValueError("cadence comparison requires unique evidence")
    allowed_sets = tuple({item} for item in CADENCE_RECORDING_IDS) + (
        set(CADENCE_RECORDING_IDS),
    )
    if {item.recording_id for item in evidence} not in allowed_sets:
        raise ValueError("cadence comparison route is outside the frozen repeats")
    if any(item.sensor != "accelerometer" or item.gate_c_eligible for item in evidence):
        raise ValueError("cadence comparison accepts public acceleration evidence only")

    frequencies = [item.analysis.metric_value("dominant_frequency_hz") for item in evidence]
    snr_values = [item.analysis.metric_value("spectral_snr_db") for item in evidence]
    rms_values = [item.analysis.metric_value("selected_axis_rms_m_s2") for item in evidence]
    record_quality = all(
        _confidence_passed(item.analysis.confidence)
        and MIN_CADENCE_HZ <= frequency <= MAX_CADENCE_HZ
        and snr >= MIN_CADENCE_SNR_DB
        for item, frequency, snr in zip(evidence, frequencies, snr_values, strict=True)
    )
    mean_frequency = float(np.mean(frequencies))
    cadence_cv = (
        float(np.std(frequencies) / mean_frequency) if len(frequencies) >= 2 else 0.0
    )
    repeat_quality = len(frequencies) < 2 or cadence_cv <= MAX_CADENCE_CV
    quality_passed = record_quality and repeat_quality
    metrics: list[PublicSensorComparisonMetric] = []
    for index, (frequency, snr, rms) in enumerate(
        zip(frequencies, snr_values, rms_values, strict=True), start=1
    ):
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key=f"repeat_{index}_cadence_hz",
                    label=f"重复 {index} 主频",
                    value=frequency,
                    unit="Hz",
                ),
                PublicSensorComparisonMetric(
                    key=f"repeat_{index}_rms_m_s2",
                    label=f"重复 {index} 动态轴 RMS",
                    value=rms,
                    unit="m/s^2",
                ),
                PublicSensorComparisonMetric(
                    key=f"repeat_{index}_snr_db",
                    label=f"重复 {index} 频谱信噪比",
                    value=snr,
                    unit="dB",
                ),
            )
        )
    if len(frequencies) >= 2:
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="mean_cadence_hz",
                    label="平均步频候选",
                    value=mean_frequency,
                    unit="Hz",
                ),
                PublicSensorComparisonMetric(
                    key="mean_cadence_steps_min",
                    label="平均步频候选",
                    value=mean_frequency * 60.0,
                    unit="steps/min",
                ),
                PublicSensorComparisonMetric(
                    key="cadence_cv_ratio",
                    label="三段步频变异系数",
                    value=cadence_cv,
                    unit="ratio",
                ),
            )
        )
    codes = [
        "cadence_band_passed" if record_quality else "cadence_band_failed",
        "cadence_repeatability_passed"
        if repeat_quality
        else "cadence_repeatability_failed",
    ]
    interpretation = (
        "公开 AS4 楼梯上行记录的周期峰与频谱信噪比通过预注册门；重复段步频候选一致。"
        if quality_passed and len(evidence) >= 2
        else "公开楼梯单段通过步频候选门；跨路面效应仍需用户真机成组对照。"
        if quality_passed
        else "公开楼梯记录未通过步频频带、信噪比或重复性门。"
    )
    return PublicSensorComparison(
        comparison_id=(
            "accelerometer-stair-cadence-repeatability"
            if len(evidence) >= 2
            else "accelerometer-stair-cadence-single"
        ),
        sensor="accelerometer",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=tuple(codes),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=tuple(metrics),
        interpretation=interpretation,
    )


def _segments(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
) -> list[tuple[int, int]]:
    mask = values >= threshold
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end > start
        and timestamps_s[end - 1] - timestamps_s[start] >= ELEVATOR_MIN_PHASE_S
    ]


def segment_elevator_motion_phases(
    recording_id: str,
    upload: SensorRecordingUpload,
) -> ElevatorPhaseSummary:
    if recording_id not in ELEVATOR_RECORDING_IDS or upload.sensor != "accelerometer":
        raise ValueError("elevator segmenter accepts only frozen AS5 acceleration records")
    timestamps = np.asarray(
        [sample.timestamp_ms for sample in upload.samples], dtype=np.float64
    )
    timestamps_s = (timestamps - timestamps[0]) / 1000.0
    xyz = np.asarray(
        [[sample.values[axis] for axis in ("x", "y", "z")] for sample in upload.samples],
        dtype=np.float64,
    )
    deltas = np.diff(timestamps_s)
    if len(timestamps_s) < 64 or np.any(deltas <= 0) or not np.isfinite(xyz).all():
        raise ValueError("elevator segmenter requires a finite monotonic acceleration series")
    sampling_rate_hz = 1.0 / float(np.median(deltas))
    window = max(3, round(sampling_rate_hz * ELEVATOR_SMOOTHING_S))
    if window % 2 == 0:
        window += 1
    magnitude = np.linalg.norm(xyz, axis=1)
    smooth = np.convolve(magnitude, np.ones(window) / window, mode="same")
    padding = window // 2
    interior = smooth[padding:-padding]
    if len(interior) < 20:
        raise ValueError("elevator series is too short after smoothing")
    baseline = float(np.median(interior))
    excursion = smooth - baseline
    positive = [
        item
        for item in _segments(
            timestamps_s,
            excursion,
            threshold=ELEVATOR_THRESHOLD_M_S2,
        )
        if item[0] >= padding and item[1] <= len(smooth) - padding
    ]
    negative = [
        item
        for item in _segments(
            timestamps_s,
            -excursion,
            threshold=ELEVATOR_THRESHOLD_M_S2,
        )
        if item[0] >= padding and item[1] <= len(smooth) - padding
    ]
    positive = sorted(
        positive,
        key=lambda item: float(np.mean(excursion[item[0] : item[1]])),
        reverse=True,
    )
    if not positive:
        raise ValueError("no sustained positive elevator phase was found")
    acceleration = positive[0]
    later_negative = [item for item in negative if item[0] > acceleration[1]]
    later_negative = sorted(
        later_negative,
        key=lambda item: float(np.mean(-excursion[item[0] : item[1]])),
        reverse=True,
    )
    if not later_negative:
        raise ValueError("no sustained deceleration phase follows acceleration")
    deceleration = later_negative[0]
    acceleration_mean = float(np.mean(excursion[acceleration[0] : acceleration[1]]))
    deceleration_mean = float(np.mean(excursion[deceleration[0] : deceleration[1]]))
    cruise_duration = float(
        timestamps_s[deceleration[0]] - timestamps_s[acceleration[1] - 1]
    )
    quality_passed = (
        acceleration_mean >= ELEVATOR_MIN_EXCURSION_M_S2
        and deceleration_mean <= -ELEVATOR_MIN_EXCURSION_M_S2
        and cruise_duration >= ELEVATOR_MIN_CRUISE_S
    )
    return ElevatorPhaseSummary(
        recording_id=recording_id,
        baseline_m_s2=baseline,
        acceleration_start_s=float(timestamps_s[acceleration[0]]),
        acceleration_end_s=float(timestamps_s[acceleration[1] - 1]),
        acceleration_mean_excursion_m_s2=acceleration_mean,
        cruise_duration_s=cruise_duration,
        deceleration_start_s=float(timestamps_s[deceleration[0]]),
        deceleration_end_s=float(timestamps_s[deceleration[1] - 1]),
        deceleration_mean_excursion_m_s2=deceleration_mean,
        quality_passed=quality_passed,
    )


def compare_public_elevator_phases(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
    phases: tuple[ElevatorPhaseSummary, ...],
) -> PublicSensorComparison:
    if not evidence or len(evidence) != len(phases):
        raise ValueError("elevator comparison requires one phase result per evidence")
    allowed_sets = tuple({item} for item in ELEVATOR_RECORDING_IDS) + (
        set(ELEVATOR_RECORDING_IDS[1:]),
        set(ELEVATOR_RECORDING_IDS),
    )
    if {item.recording_id for item in evidence} not in allowed_sets:
        raise ValueError("elevator comparison route is outside frozen AS5 records")
    if tuple(item.recording_id for item in evidence) != tuple(
        item.recording_id for item in phases
    ):
        raise ValueError("elevator phase lineage does not match evidence order")
    quality_passed = all(
        phase.quality_passed and _confidence_passed(item.analysis.confidence)
        for item, phase in zip(evidence, phases, strict=True)
    )
    metrics = (
        PublicSensorComparisonMetric(
            key="phase_sequences_detected",
            label="通过阶段序列数",
            value=float(sum(item.quality_passed for item in phases)),
            unit="records",
        ),
        PublicSensorComparisonMetric(
            key="mean_acceleration_start_s",
            label="平均加速开始时刻",
            value=float(np.mean([item.acceleration_start_s for item in phases])),
            unit="s",
        ),
        PublicSensorComparisonMetric(
            key="mean_positive_excursion_m_s2",
            label="平均正向加速度偏移",
            value=float(
                np.mean([item.acceleration_mean_excursion_m_s2 for item in phases])
            ),
            unit="m/s^2",
        ),
        PublicSensorComparisonMetric(
            key="mean_acceleration_duration_s",
            label="平均加速段时长",
            value=float(
                np.mean(
                    [
                        item.acceleration_end_s - item.acceleration_start_s
                        for item in phases
                    ]
                )
            ),
            unit="s",
        ),
        PublicSensorComparisonMetric(
            key="mean_cruise_duration_s",
            label="平均中间稳定段时长",
            value=float(np.mean([item.cruise_duration_s for item in phases])),
            unit="s",
        ),
        PublicSensorComparisonMetric(
            key="mean_deceleration_start_s",
            label="平均减速开始时刻",
            value=float(np.mean([item.deceleration_start_s for item in phases])),
            unit="s",
        ),
        PublicSensorComparisonMetric(
            key="mean_negative_excursion_m_s2",
            label="平均反向加速度偏移",
            value=float(
                np.mean([item.deceleration_mean_excursion_m_s2 for item in phases])
            ),
            unit="m/s^2",
        ),
        PublicSensorComparisonMetric(
            key="mean_deceleration_duration_s",
            label="平均减速段时长",
            value=float(
                np.mean(
                    [
                        item.deceleration_end_s - item.deceleration_start_s
                        for item in phases
                    ]
                )
            ),
            unit="s",
        ),
    )
    return PublicSensorComparison(
        comparison_id="accelerometer-elevator-phase-sequence",
        sensor="accelerometer",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=(
            "acceleration_cruise_deceleration_passed"
            if quality_passed
            else "acceleration_cruise_deceleration_failed",
            "no_displacement_integration_claim",
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=metrics,
        interpretation=(
            "公开 AS5 记录均检测到正向加速、稳定中段和反向减速的有序序列；"
            "结果不积分为位移或楼层。"
            if quality_passed
            else "至少一条公开 AS5 记录未通过阶段持续时间、幅值或顺序门。"
        ),
    )


def compare_public_vibration_states(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
) -> PublicSensorComparison:
    if not evidence or len(evidence) != len({item.evidence_id for item in evidence}):
        raise ValueError("vibration response comparison requires unique evidence")
    allowed_sets = tuple({item} for item in VIBRATION_RECORDING_IDS) + (
        set(VIBRATION_RECORDING_IDS),
    )
    by_recording = {item.recording_id: item for item in evidence}
    if set(by_recording) not in allowed_sets:
        raise ValueError("vibration route is outside the frozen AS7 pair")

    metrics: list[PublicSensorComparisonMetric] = []
    codes: list[str] = []
    stationary = by_recording.get(VIBRATION_RECORDING_IDS[0])
    handheld = by_recording.get(VIBRATION_RECORDING_IDS[1])
    stationary_passed: bool | None = None
    handheld_passed: bool | None = None
    if stationary is not None:
        rms = stationary.analysis.metric_value("selected_axis_rms_m_s2")
        stationary_passed = (
            _confidence_passed(stationary.analysis.confidence)
            and rms <= MAX_STATIONARY_RMS_M_S2
        )
        metrics.append(
            PublicSensorComparisonMetric(
                key="stationary_rms_m_s2",
                label="静止动态轴 RMS",
                value=rms,
                unit="m/s^2",
            )
        )
        codes.append(
            "stationary_noise_gate_passed"
            if stationary_passed
            else "stationary_noise_gate_failed"
        )
    if handheld is not None:
        rms = handheld.analysis.metric_value("selected_axis_rms_m_s2")
        peak = handheld.analysis.metric_value("selected_axis_peak_to_peak_m_s2")
        handheld_passed = (
            _confidence_passed(handheld.analysis.confidence)
            and rms >= MIN_HANDHELD_RMS_M_S2
            and peak >= MIN_HANDHELD_PEAK_TO_PEAK_M_S2
        )
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="handheld_rms_m_s2",
                    label="手持动态轴 RMS",
                    value=rms,
                    unit="m/s^2",
                ),
                PublicSensorComparisonMetric(
                    key="handheld_peak_to_peak_m_s2",
                    label="手持动态轴峰峰值",
                    value=peak,
                    unit="m/s^2",
                ),
            )
        )
        codes.append(
            "handheld_response_gate_passed"
            if handheld_passed
            else "handheld_response_gate_failed"
        )
    if stationary is not None and handheld is not None:
        stationary_rms = stationary.analysis.metric_value("selected_axis_rms_m_s2")
        handheld_rms = handheld.analysis.metric_value("selected_axis_rms_m_s2")
        ratio = handheld_rms / max(stationary_rms, 1e-12)
        ratio_passed = ratio >= MIN_MOTION_SEPARATION_RATIO
        metrics.append(
            PublicSensorComparisonMetric(
                key="motion_to_stationary_rms_ratio",
                label="运动与静止 RMS 比",
                value=ratio,
                unit="ratio",
            )
        )
        codes.append(
            "motion_state_separation_passed"
            if ratio_passed
            else "motion_state_separation_failed"
        )
        quality_passed = bool(stationary_passed and handheld_passed and ratio_passed)
    else:
        quality_passed = bool(
            stationary_passed if stationary is not None else handheld_passed
        )
    return PublicSensorComparison(
        comparison_id=(
            "accelerometer-stationary-handheld-contrast"
            if len(evidence) == 2
            else "accelerometer-single-motion-state"
        ),
        sensor="accelerometer",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=tuple(codes),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=tuple(metrics),
        interpretation=(
            "公开 AS7 静止与手持窗口通过响应分离门，只证明加速度链能区分这两种运动状态；"
            "不能据此识别偏载、松动、传振或结构放大。"
            if quality_passed and len(evidence) == 2
            else "公开单状态记录通过预注册响应门；设备故障原因仍需真机对照。"
            if quality_passed
            else "公开 AS7 记录未通过静止噪声、运动响应或状态分离门。"
        ),
    )
