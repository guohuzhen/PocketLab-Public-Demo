from __future__ import annotations

import math
from pathlib import Path
from typing import Final

from pocketlab.analyzers.registry import analyze_sensor_recording
from pocketlab.public_replay_dataset import (
    load_public_replay_dataset,
    read_public_replay_recording,
)
from pocketlab.public_sensor_agent_models import (
    PublicSensorComparison,
    PublicSensorComparisonMetric,
    PublicSensorEvidenceSnapshot,
    PublicSensorToolExecution,
)

DATASET_ID: Final = "microphone-noisecapture-andorra-odbl-v1"
EARLY_RECORDING_ID: Final = "andorra-track-early-window"
LATE_RECORDING_ID: Final = "andorra-track-late-window"

ANALYZE_TOOL_ID: Final = "analyze_microphone_relative_window"
COMPARE_TOOL_ID: Final = "compare_microphone_chronological_windows"

EXPECTED_METRICS: Final = {
    EARLY_RECORDING_ID: {
        "mean_relative_level_db": 0.1651969,
        "peak_relative_level_db": 2.2467,
        "relative_level_span_db": 5.190689,
    },
    LATE_RECORDING_ID: {
        "mean_relative_level_db": 7.11797965,
        "peak_relative_level_db": 16.068635,
        "relative_level_span_db": 15.670842,
    },
}

MINIMUM_MEAN_CONTRAST_DB: Final = 6.0
MINIMUM_PEAK_CONTRAST_DB: Final = 10.0
MINIMUM_SPAN_CONTRAST_DB: Final = 8.0


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def _recording_gate(evidence: PublicSensorEvidenceSnapshot) -> bool:
    analysis = evidence.analysis
    expected = EXPECTED_METRICS[evidence.recording_id]
    return (
        analysis.sample_count == 20
        and _close(analysis.duration_s, 19.0)
        and _close(analysis.sampling_rate_hz, 1.0)
        and _close(analysis.sampling_jitter_ratio, 0.0)
        and _close(analysis.max_sampling_gap_ratio, 1.0)
        and analysis.confidence == "medium"
        and all(_close(analysis.metric_value(key), value) for key, value in expected.items())
        and any("No raw audio" in warning for warning in analysis.warnings)
        and any("not a phyphox" in warning for warning in analysis.warnings)
    )


def load_public_microphone_evidence(
    pack_dir: Path,
    recording_ids: tuple[str, ...],
) -> tuple[
    tuple[PublicSensorEvidenceSnapshot, ...],
    PublicSensorComparison,
    tuple[PublicSensorToolExecution, ...],
]:
    """Load only the reviewed, privacy-minimized relative-level windows."""

    allowed = {EARLY_RECORDING_ID, LATE_RECORDING_ID}
    if (
        not recording_ids
        or len(recording_ids) != len(set(recording_ids))
        or not set(recording_ids).issubset(allowed)
    ):
        raise ValueError("microphone recording route is outside the frozen protocol")

    manifest = load_public_replay_dataset(pack_dir)
    if (
        manifest.dataset_id != DATASET_ID
        or manifest.sensor != "microphone"
        or manifest.analyzer_id != "pocketlab.microphone.derived.v1"
        or manifest.analyzer_version != "1.0.0"
        or manifest.source.source_id != "noisecapture-andorra-relative-level-20231008"
        or manifest.source.license_spdx != "ODbL-1.0"
        or manifest.public_replay_status != "source_validated"
        or manifest.agent_ready
        or manifest.privacy_review.deployment_scope != "local_only"
        or not manifest.privacy_review.requires_user_acknowledgement
    ):
        raise ValueError("microphone public replay manifest contract changed")
    if manifest.source.doi is None:
        raise ValueError("microphone source associated-work DOI is required")

    by_id = {item.recording_id: item for item in manifest.recordings}
    snapshots: list[PublicSensorEvidenceSnapshot] = []
    tool_trace: list[PublicSensorToolExecution] = []
    for sequence, recording_id in enumerate(recording_ids, start=1):
        recording = by_id.get(recording_id)
        if recording is None:
            raise ValueError("microphone recording is missing from the reviewed manifest")
        if (
            recording.independent_measurement
            or recording.gate_c_eligible
            or recording.analysis_confidence_ceiling != "medium"
            or recording.invalidated_analysis_fields
            or recording.invalidated_metric_keys
        ):
            raise ValueError("microphone evidence boundary changed")
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        if not upload.provenance.privacy_acknowledged:
            raise ValueError("microphone public replay privacy gate was not applied")
        analysis = analyze_sensor_recording(upload)
        if analysis != recording.reference_analysis:
            raise ValueError("microphone analyzer drifted from the frozen reference")
        evidence_id = f"microphone-{recording_id}"
        snapshot = PublicSensorEvidenceSnapshot(
            evidence_id=evidence_id,
            dataset_id=manifest.dataset_id,
            recording_id=recording_id,
            sensor="microphone",
            data_class=manifest.data_class,
            condition_label=recording.label,
            device_scope=(
                f"{recording.device_alias}; public NoiseCapture-derived level only, "
                "not raw audio and not phyphox"
            ),
            source_title=manifest.source.title,
            source_url=manifest.source.record_url,
            doi=manifest.source.doi,
            license_spdx=manifest.source.license_spdx,
            analysis=analysis,
            processing_disclosures=tuple(recording.processing_disclosures),
            claim_boundary=tuple(manifest.claim_boundary),
        )
        snapshots.append(snapshot)
        gate_passed = _recording_gate(snapshot)
        tool_trace.append(
            PublicSensorToolExecution(
                sequence=sequence,
                tool_id=ANALYZE_TOOL_ID,
                status="completed" if gate_passed else "rejected",
                evidence_ids=(evidence_id,),
                result_codes=(
                    "relative_level_reference_matched" if gate_passed else "relative_level_reference_failed",
                    "one_second_cadence_verified" if gate_passed else "cadence_gate_failed",
                    "raw_audio_absent",
                    "calibrated_spl_not_supported",
                ),
            )
        )

    comparison = compare_public_microphone_windows(tuple(snapshots))
    tool_trace.append(
        PublicSensorToolExecution(
            sequence=len(tool_trace) + 1,
            tool_id=COMPARE_TOOL_ID,
            status="completed" if comparison.quality_passed else "rejected",
            evidence_ids=comparison.evidence_ids,
            result_codes=comparison.result_codes,
        )
    )
    return tuple(snapshots), comparison, tuple(tool_trace)


def _snapshot_metrics(evidence: PublicSensorEvidenceSnapshot) -> list[PublicSensorComparisonMetric]:
    prefix = "early" if evidence.recording_id == EARLY_RECORDING_ID else "late"
    analysis = evidence.analysis
    return [
        PublicSensorComparisonMetric(
            key=f"{prefix}_mean_relative_level_db",
            label=f"{prefix} 平均相对级别",
            value=analysis.metric_value("mean_relative_level_db"),
            unit="dB_relative",
        ),
        PublicSensorComparisonMetric(
            key=f"{prefix}_peak_relative_level_db",
            label=f"{prefix} 峰值相对级别",
            value=analysis.metric_value("peak_relative_level_db"),
            unit="dB_relative",
        ),
        PublicSensorComparisonMetric(
            key=f"{prefix}_relative_level_span_db",
            label=f"{prefix} 相对级别范围",
            value=analysis.metric_value("relative_level_span_db"),
            unit="dB_relative",
        ),
    ]


def compare_public_microphone_windows(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
) -> PublicSensorComparison:
    """Apply frozen descriptive gates without inventing a source or room-position cause."""

    if not evidence or len(evidence) != len({item.evidence_id for item in evidence}):
        raise ValueError("microphone comparison requires unique evidence")
    if any(item.sensor != "microphone" or item.gate_c_eligible for item in evidence):
        raise ValueError("microphone comparison accepts only public non-Gate-C evidence")
    by_recording = {item.recording_id: item for item in evidence}
    allowed_sets = (
        {EARLY_RECORDING_ID},
        {LATE_RECORDING_ID},
        {EARLY_RECORDING_ID, LATE_RECORDING_ID},
    )
    if set(by_recording) not in allowed_sets:
        raise ValueError("microphone comparison route is outside the frozen windows")

    metrics: list[PublicSensorComparisonMetric] = []
    single_gates: dict[str, bool] = {}
    result_codes: list[str] = []
    for recording_id in (EARLY_RECORDING_ID, LATE_RECORDING_ID):
        item = by_recording.get(recording_id)
        if item is None:
            continue
        passed = _recording_gate(item)
        single_gates[recording_id] = passed
        metrics.extend(_snapshot_metrics(item))
        result_codes.append(
            f"{'early' if recording_id == EARLY_RECORDING_ID else 'late'}_relative_level_gate_"
            f"{'passed' if passed else 'failed'}"
        )

    if len(by_recording) == 2:
        early = by_recording[EARLY_RECORDING_ID].analysis
        late = by_recording[LATE_RECORDING_ID].analysis
        mean_delta = late.metric_value("mean_relative_level_db") - early.metric_value(
            "mean_relative_level_db"
        )
        peak_delta = late.metric_value("peak_relative_level_db") - early.metric_value(
            "peak_relative_level_db"
        )
        span_delta = late.metric_value("relative_level_span_db") - early.metric_value(
            "relative_level_span_db"
        )
        contrast_passed = (
            mean_delta >= MINIMUM_MEAN_CONTRAST_DB
            and peak_delta >= MINIMUM_PEAK_CONTRAST_DB
            and span_delta >= MINIMUM_SPAN_CONTRAST_DB
        )
        quality_passed = all(single_gates.values()) and contrast_passed
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="late_minus_early_mean_db",
                    label="后段减前段平均相对级别",
                    value=mean_delta,
                    unit="dB_relative",
                ),
                PublicSensorComparisonMetric(
                    key="late_minus_early_peak_db",
                    label="后段减前段峰值相对级别",
                    value=peak_delta,
                    unit="dB_relative",
                ),
                PublicSensorComparisonMetric(
                    key="late_minus_early_span_db",
                    label="后段减前段相对范围",
                    value=span_delta,
                    unit="dB_relative",
                ),
            )
        )
        result_codes.append(
            "chronological_contrast_gate_passed"
            if contrast_passed
            else "chronological_contrast_gate_failed"
        )
        interpretation = (
            "同一公开 NoiseCapture 轨迹的后 20 秒具有更高的派生相对级别和更宽的变化范围；"
            "这只证明冻结序列的时间对比，不说明声源、房间位置或绝对声压原因。"
            if quality_passed
            else "来源、采样、派生级别或前后对比门未同时通过。"
        )
        comparison_id = "microphone-chronological-relative-level-contrast"
    else:
        quality_passed = next(iter(single_gates.values()))
        side = "early" if EARLY_RECORDING_ID in by_recording else "late"
        interpretation = (
            f"公开 NoiseCapture 的{('前' if side == 'early' else '后')} 20 秒窗口通过"
            "相对级别、固定节奏和来源回归门；结论仅为该窗口的描述统计。"
            if quality_passed
            else "该公开窗口未通过预注册来源或派生级别门。"
        )
        comparison_id = f"microphone-{side}-relative-level-check"

    return PublicSensorComparison(
        comparison_id=comparison_id,
        sensor="microphone",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=tuple(result_codes),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=tuple(metrics),
        interpretation=interpretation,
    )
