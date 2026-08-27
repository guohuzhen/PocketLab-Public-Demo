from __future__ import annotations

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

DATASET_ID: Final = "proximity-nist-perfloc-as7-pixel-20180516-v1"
EARLY_RECORDING_ID: Final = "as7-proximity-events-early"
LATE_RECORDING_ID: Final = "as7-proximity-events-late"

ANALYZE_TOOL_ID: Final = "analyze_proximity_event_slice"
COMPARE_TOOL_ID: Final = "compare_proximity_state_codes"

EXPECTED_EVENT_COUNT: Final = 4
EXPECTED_TRANSITION_COUNT: Final = 3
EXPECTED_NEAR_STATE_CM: Final = 0.0
EXPECTED_FAR_STATE_CM: Final = 5.0
EXPECTED_ACCURACY_STATE: Final = 3.0
REQUIRED_INVALIDATED_FIELDS: Final = {
    "sampling_rate_hz",
    "sampling_jitter_ratio",
    "max_sampling_gap_ratio",
}


def load_public_proximity_evidence(
    pack_dir: Path,
    recording_ids: tuple[str, ...],
) -> tuple[
    tuple[PublicSensorEvidenceSnapshot, ...],
    PublicSensorComparison,
    tuple[PublicSensorToolExecution, ...],
]:
    """Load exact reviewed sparse events and apply the frozen binary-state gates."""

    allowed = {EARLY_RECORDING_ID, LATE_RECORDING_ID}
    if (
        not recording_ids
        or len(recording_ids) != len(set(recording_ids))
        or not set(recording_ids).issubset(allowed)
    ):
        raise ValueError("proximity recording route is outside the frozen protocol")

    manifest = load_public_replay_dataset(pack_dir)
    if (
        manifest.dataset_id != DATASET_ID
        or manifest.sensor != "proximity"
        or manifest.analyzer_id != "pocketlab.proximity.v2"
        or manifest.public_replay_status != "source_validated"
        or manifest.agent_ready
    ):
        raise ValueError("proximity public replay manifest contract changed")
    if manifest.source.doi is None:
        raise ValueError("proximity source DOI is required")

    by_id = {item.recording_id: item for item in manifest.recordings}
    snapshots: list[PublicSensorEvidenceSnapshot] = []
    tool_trace: list[PublicSensorToolExecution] = []
    for sequence, recording_id in enumerate(recording_ids, start=1):
        recording = by_id.get(recording_id)
        if recording is None:
            raise ValueError("proximity recording is missing from the reviewed manifest")
        if (
            set(recording.invalidated_analysis_fields) != REQUIRED_INVALIDATED_FIELDS
            or recording.invalidated_metric_keys != ["near_state_fraction"]
            or recording.analysis_confidence_ceiling != "low"
        ):
            raise ValueError("proximity sparse-event invalidation contract changed")
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        if (
            len(upload.samples) != EXPECTED_EVENT_COUNT
            or any(
                sample.values.get("accuracy") != EXPECTED_ACCURACY_STATE
                for sample in upload.samples
            )
        ):
            raise ValueError("proximity source event or accuracy contract changed")
        analysis = analyze_sensor_recording(upload)
        if analysis != recording.reference_analysis:
            raise ValueError("proximity analyzer drifted from the frozen reference")
        evidence_id = f"proximity-{recording_id}"
        snapshots.append(
            PublicSensorEvidenceSnapshot(
                evidence_id=evidence_id,
                dataset_id=manifest.dataset_id,
                recording_id=recording_id,
                sensor="proximity",
                data_class=manifest.data_class,
                condition_label=recording.label,
                device_scope=(
                    f"{recording.device_alias}; sparse public Android SensorEvent capture, "
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
                sequence=sequence,
                tool_id=ANALYZE_TOOL_ID,
                evidence_ids=(evidence_id,),
                result_codes=(
                    "source_event_count_4",
                    "accuracy_state_3",
                    "cadence_fields_invalidated",
                    "analysis_reference_matched",
                ),
            )
        )

    comparison = compare_public_proximity_states(tuple(snapshots))
    tool_trace.append(
        PublicSensorToolExecution(
            sequence=len(tool_trace) + 1,
            tool_id=COMPARE_TOOL_ID,
            evidence_ids=comparison.evidence_ids,
            result_codes=comparison.result_codes,
        )
    )
    return tuple(snapshots), comparison, tuple(tool_trace)


def _slice_gate(evidence: PublicSensorEvidenceSnapshot) -> tuple[bool, list[PublicSensorComparisonMetric]]:
    analysis = evidence.analysis
    event_count = float(analysis.sample_count)
    level_count = analysis.metric_value("observed_level_count")
    mode = analysis.metric_value("signal_mode_code")
    transitions = analysis.metric_value("transition_count")
    near = analysis.metric_value("near_state_value_cm")
    far = analysis.metric_value("far_state_value_cm")
    passed = (
        event_count == EXPECTED_EVENT_COUNT
        and level_count == 2.0
        and mode == 1.0
        and transitions == EXPECTED_TRANSITION_COUNT
        and near == EXPECTED_NEAR_STATE_CM
        and far == EXPECTED_FAR_STATE_CM
        and any("event-driven" in item for item in analysis.warnings)
    )
    prefix = "early" if evidence.recording_id == EARLY_RECORDING_ID else "late"
    return passed, [
        PublicSensorComparisonMetric(
            key=f"{prefix}_event_count",
            label=f"{prefix} 稀疏事件数",
            value=event_count,
            unit="count",
        ),
        PublicSensorComparisonMetric(
            key=f"{prefix}_transition_count",
            label=f"{prefix} 状态切换数",
            value=transitions,
            unit="count",
        ),
        PublicSensorComparisonMetric(
            key=f"{prefix}_near_state_cm",
            label=f"{prefix} near 状态编码",
            value=near,
            unit="cm",
        ),
        PublicSensorComparisonMetric(
            key=f"{prefix}_far_state_cm",
            label=f"{prefix} far 状态编码",
            value=far,
            unit="cm",
        ),
    ]


def compare_public_proximity_states(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
) -> PublicSensorComparison:
    """Check binary state coding without interpreting cadence as dwell or response time."""

    if not evidence or len(evidence) != len({item.evidence_id for item in evidence}):
        raise ValueError("proximity comparison requires unique evidence")
    if any(item.sensor != "proximity" or item.gate_c_eligible for item in evidence):
        raise ValueError("proximity comparison accepts only public non-Gate-C evidence")
    by_recording = {item.recording_id: item for item in evidence}
    if set(by_recording) not in (
        {EARLY_RECORDING_ID},
        {LATE_RECORDING_ID},
        {EARLY_RECORDING_ID, LATE_RECORDING_ID},
    ):
        raise ValueError("proximity comparison route is outside the frozen pair")

    metrics: list[PublicSensorComparisonMetric] = []
    result_codes: list[str] = []
    slice_results: dict[str, bool] = {}
    for recording_id in (EARLY_RECORDING_ID, LATE_RECORDING_ID):
        item = by_recording.get(recording_id)
        if item is None:
            continue
        passed, slice_metrics = _slice_gate(item)
        slice_results[recording_id] = passed
        metrics.extend(slice_metrics)
        result_codes.append(
            f"{'early' if recording_id == EARLY_RECORDING_ID else 'late'}_binary_state_gate_"
            f"{'passed' if passed else 'failed'}"
        )

    if len(by_recording) == 2:
        early = by_recording[EARLY_RECORDING_ID].analysis
        late = by_recording[LATE_RECORDING_ID].analysis
        near_delta = abs(
            early.metric_value("near_state_value_cm")
            - late.metric_value("near_state_value_cm")
        )
        far_delta = abs(
            early.metric_value("far_state_value_cm")
            - late.metric_value("far_state_value_cm")
        )
        consistent = near_delta == 0.0 and far_delta == 0.0
        quality_passed = all(slice_results.values()) and consistent
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="near_state_code_delta_cm",
                    label="两切片 near 编码差",
                    value=near_delta,
                    unit="cm",
                ),
                PublicSensorComparisonMetric(
                    key="far_state_code_delta_cm",
                    label="两切片 far 编码差",
                    value=far_delta,
                    unit="cm",
                ),
                PublicSensorComparisonMetric(
                    key="total_transition_count",
                    label="两切片总状态切换数",
                    value=sum(
                        item.analysis.metric_value("transition_count")
                        for item in evidence
                    ),
                    unit="count",
                ),
            )
        )
        result_codes.append(
            "binary_state_code_consistency_passed"
            if consistent
            else "binary_state_code_consistency_failed"
        )
        interpretation = (
            "同一公开 Pixel XL acquisition 的前后稀疏切片都呈 0/5 cm 二态编码且切换重复，"
            "只支持该来源的二态状态机响应；不支持占空比、响应时间或真实距离结论。"
            if quality_passed
            else "稀疏事件、二态编码或前后状态一致性门未同时通过。"
        )
        comparison_id = "proximity-binary-state-consistency"
    else:
        quality_passed = next(iter(slice_results.values()))
        side = "early" if EARLY_RECORDING_ID in by_recording else "late"
        interpretation = (
            "该公开稀疏切片通过 0/5 cm 二态编码与切换门，只能描述来源中的状态事件。"
            if quality_passed
            else "该公开稀疏切片未通过预注册二态事件门。"
        )
        comparison_id = f"proximity-{side}-binary-state-check"

    return PublicSensorComparison(
        comparison_id=comparison_id,
        sensor="proximity",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=tuple(result_codes),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=tuple(metrics),
        interpretation=interpretation,
    )
