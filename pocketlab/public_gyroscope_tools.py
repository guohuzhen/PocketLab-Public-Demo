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

DATASET_ID: Final = "gyroscope-nist-perfloc-as7-pixel-20180516-v1"
ANCHOR_RECORDING_ID: Final = "as7-dot55-floor-attitude-anchor"
TRANSITION_RECORDING_ID: Final = "as7-dot55-to56-handheld-transition"

ANALYZE_TOOL_ID: Final = "analyze_gyroscope_recording"
COMPARE_TOOL_ID: Final = "compare_gyroscope_motion_states"

MAX_STATIONARY_MEAN_RAD_S: Final = 0.01
MAX_STATIONARY_PEAK_RAD_S: Final = 0.02
MIN_ACTIVE_MEAN_RAD_S: Final = 0.20
MIN_ACTIVE_PEAK_RAD_S: Final = 1.0
MIN_STATE_SEPARATION_RATIO: Final = 20.0


def _confidence_passed(value: str) -> bool:
    return value in {"medium", "high"}


def load_public_gyroscope_evidence(
    pack_dir: Path,
    recording_ids: tuple[str, ...],
) -> tuple[
    tuple[PublicSensorEvidenceSnapshot, ...],
    PublicSensorComparison,
    tuple[PublicSensorToolExecution, ...],
]:
    """Load exact registered recordings, run v1 analysis, and apply frozen gates."""

    allowed = {ANCHOR_RECORDING_ID, TRANSITION_RECORDING_ID}
    if (
        not recording_ids
        or len(recording_ids) != len(set(recording_ids))
        or not set(recording_ids).issubset(allowed)
    ):
        raise ValueError("gyroscope recording route is outside the frozen protocol")

    manifest = load_public_replay_dataset(pack_dir)
    if (
        manifest.dataset_id != DATASET_ID
        or manifest.sensor != "gyroscope"
        or manifest.analyzer_id != "pocketlab.gyroscope.v1"
        or manifest.public_replay_status != "source_validated"
        or manifest.agent_ready
    ):
        raise ValueError("gyroscope public replay manifest contract changed")
    if manifest.source.doi is None:
        raise ValueError("gyroscope source DOI is required")

    by_id = {item.recording_id: item for item in manifest.recordings}
    snapshots: list[PublicSensorEvidenceSnapshot] = []
    tool_trace: list[PublicSensorToolExecution] = []
    for sequence, recording_id in enumerate(recording_ids, start=1):
        recording = by_id.get(recording_id)
        if recording is None:
            raise ValueError("gyroscope recording is missing from the reviewed manifest")
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        analysis = analyze_sensor_recording(upload)
        if analysis != recording.reference_analysis:
            raise ValueError("gyroscope analyzer drifted from the frozen reference")
        evidence_id = f"gyroscope-{recording_id}"
        snapshots.append(
            PublicSensorEvidenceSnapshot(
                evidence_id=evidence_id,
                dataset_id=manifest.dataset_id,
                recording_id=recording_id,
                sensor="gyroscope",
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
                sequence=sequence,
                tool_id=ANALYZE_TOOL_ID,
                evidence_ids=(evidence_id,),
                result_codes=(
                    f"confidence_{analysis.confidence}",
                    "analysis_reference_matched",
                ),
            )
        )

    comparison = compare_public_gyroscope_states(tuple(snapshots))
    tool_trace.append(
        PublicSensorToolExecution(
            sequence=len(tool_trace) + 1,
            tool_id=COMPARE_TOOL_ID,
            evidence_ids=comparison.evidence_ids,
            result_codes=comparison.result_codes,
        )
    )
    return tuple(snapshots), comparison, tuple(tool_trace)


def compare_public_gyroscope_states(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
) -> PublicSensorComparison:
    """Evaluate only stationary bias, handheld response, or their paired contrast."""

    if not evidence or len(evidence) != len({item.evidence_id for item in evidence}):
        raise ValueError("gyroscope comparison requires unique evidence")
    if any(item.sensor != "gyroscope" or item.gate_c_eligible for item in evidence):
        raise ValueError("gyroscope comparison accepts only public non-Gate-C evidence")

    by_recording = {item.recording_id: item for item in evidence}
    if set(by_recording) not in (
        {ANCHOR_RECORDING_ID},
        {TRANSITION_RECORDING_ID},
        {ANCHOR_RECORDING_ID, TRANSITION_RECORDING_ID},
    ):
        raise ValueError("gyroscope comparison route is outside the frozen pair")

    metrics: list[PublicSensorComparisonMetric] = []
    result_codes: list[str] = []
    anchor_passed: bool | None = None
    transition_passed: bool | None = None

    anchor = by_recording.get(ANCHOR_RECORDING_ID)
    if anchor is not None:
        anchor_mean = anchor.analysis.metric_value("mean_angular_speed_rad_s")
        anchor_peak = anchor.analysis.metric_value("peak_angular_speed_rad_s")
        anchor_bias = anchor.analysis.metric_value("stationary_bias_candidate_rad_s")
        anchor_passed = (
            _confidence_passed(anchor.analysis.confidence)
            and anchor_mean <= MAX_STATIONARY_MEAN_RAD_S
            and anchor_peak <= MAX_STATIONARY_PEAK_RAD_S
        )
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="stationary_mean_rad_s",
                    label="静止窗口平均角速度",
                    value=anchor_mean,
                    unit="rad/s",
                ),
                PublicSensorComparisonMetric(
                    key="stationary_peak_rad_s",
                    label="静止窗口峰值角速度",
                    value=anchor_peak,
                    unit="rad/s",
                ),
                PublicSensorComparisonMetric(
                    key="stationary_bias_candidate_rad_s",
                    label="静止零偏候选",
                    value=anchor_bias,
                    unit="rad/s",
                ),
            )
        )
        result_codes.append(
            "stationary_gate_passed" if anchor_passed else "stationary_gate_failed"
        )

    transition = by_recording.get(TRANSITION_RECORDING_ID)
    if transition is not None:
        transition_mean = transition.analysis.metric_value("mean_angular_speed_rad_s")
        transition_peak = transition.analysis.metric_value("peak_angular_speed_rad_s")
        transition_passed = (
            _confidence_passed(transition.analysis.confidence)
            and transition_mean >= MIN_ACTIVE_MEAN_RAD_S
            and transition_peak >= MIN_ACTIVE_PEAK_RAD_S
        )
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="handheld_mean_rad_s",
                    label="手持窗口平均角速度",
                    value=transition_mean,
                    unit="rad/s",
                ),
                PublicSensorComparisonMetric(
                    key="handheld_peak_rad_s",
                    label="手持窗口峰值角速度",
                    value=transition_peak,
                    unit="rad/s",
                ),
            )
        )
        result_codes.append(
            "handheld_gate_passed" if transition_passed else "handheld_gate_failed"
        )

    if anchor is not None and transition is not None:
        anchor_mean = anchor.analysis.metric_value("mean_angular_speed_rad_s")
        transition_mean = transition.analysis.metric_value("mean_angular_speed_rad_s")
        ratio = transition_mean / max(anchor_mean, 1e-12)
        ratio_passed = ratio >= MIN_STATE_SEPARATION_RATIO
        quality_passed = bool(anchor_passed and transition_passed and ratio_passed)
        metrics.append(
            PublicSensorComparisonMetric(
                key="motion_to_stationary_ratio",
                label="手持与静止平均角速度比",
                value=ratio,
                unit="ratio",
            )
        )
        result_codes.append(
            "state_separation_passed" if ratio_passed else "state_separation_failed"
        )
        interpretation = (
            "同一公开 Pixel XL 采集中的静止锚点与手持转动窗口通过预注册质量门，"
            "可支持陀螺仪对角运动有明显响应的有边界结论。"
            if quality_passed
            else "静止、手持响应或状态分离门未同时通过，需要真机重采后再判断。"
        )
        comparison_id = "gyroscope-stationary-handheld-contrast"
    elif anchor is not None:
        quality_passed = bool(anchor_passed)
        interpretation = (
            "公开静止锚点满足低角速度质量门，只能作为零偏候选参考。"
            if quality_passed
            else "公开静止锚点未通过预注册低角速度质量门。"
        )
        comparison_id = "gyroscope-stationary-bias-check"
    else:
        quality_passed = bool(transition_passed)
        interpretation = (
            "公开手持窗口满足角运动响应门，只能说明该记录中存在明显转动。"
            if quality_passed
            else "公开手持窗口未通过预注册角运动响应门。"
        )
        comparison_id = "gyroscope-handheld-response-check"

    return PublicSensorComparison(
        comparison_id=comparison_id,
        sensor="gyroscope",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=tuple(result_codes),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=tuple(metrics),
        interpretation=interpretation,
    )
