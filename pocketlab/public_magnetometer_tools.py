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

DATASET_ID: Final = "magnetometer-nist-perfloc-as7-pixel-20180516-v1"
ANCHOR_RECORDING_ID: Final = "as7-dot15-stable-magnetic-anchor"
CHANGE_RECORDING_ID: Final = "as7-dot15-to16-magnetic-field-change"

ANALYZE_TOOL_ID: Final = "analyze_magnetometer_recording"
COMPARE_TOOL_ID: Final = "compare_magnetic_field_states"

MAX_STABLE_STD_UT: Final = 1.0
MAX_STABLE_PEAK_TO_PEAK_UT: Final = 2.0
MAX_STABLE_DEVIATION_UT: Final = 2.0
MIN_CHANGE_STD_UT: Final = 5.0
MIN_CHANGE_PEAK_TO_PEAK_UT: Final = 20.0
MIN_CHANGE_DEVIATION_UT: Final = 10.0
MIN_VARIABILITY_RATIO: Final = 15.0


def _confidence_passed(value: str) -> bool:
    return value in {"medium", "high"}


def load_public_magnetometer_evidence(
    pack_dir: Path,
    recording_ids: tuple[str, ...],
) -> tuple[
    tuple[PublicSensorEvidenceSnapshot, ...],
    PublicSensorComparison,
    tuple[PublicSensorToolExecution, ...],
]:
    """Load exact reviewed field windows and apply the frozen response gates."""

    allowed = {ANCHOR_RECORDING_ID, CHANGE_RECORDING_ID}
    if (
        not recording_ids
        or len(recording_ids) != len(set(recording_ids))
        or not set(recording_ids).issubset(allowed)
    ):
        raise ValueError("magnetometer recording route is outside the frozen protocol")

    manifest = load_public_replay_dataset(pack_dir)
    if (
        manifest.dataset_id != DATASET_ID
        or manifest.sensor != "magnetometer"
        or manifest.analyzer_id != "pocketlab.magnetometer.v1"
        or manifest.public_replay_status != "source_validated"
        or manifest.agent_ready
    ):
        raise ValueError("magnetometer public replay manifest contract changed")
    if manifest.source.doi is None:
        raise ValueError("magnetometer source DOI is required")

    by_id = {item.recording_id: item for item in manifest.recordings}
    snapshots: list[PublicSensorEvidenceSnapshot] = []
    tool_trace: list[PublicSensorToolExecution] = []
    for sequence, recording_id in enumerate(recording_ids, start=1):
        recording = by_id.get(recording_id)
        if recording is None:
            raise ValueError("magnetometer recording is missing from the reviewed manifest")
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        analysis = analyze_sensor_recording(upload)
        if analysis != recording.reference_analysis:
            raise ValueError("magnetometer analyzer drifted from the frozen reference")
        evidence_id = f"magnetometer-{recording_id}"
        snapshots.append(
            PublicSensorEvidenceSnapshot(
                evidence_id=evidence_id,
                dataset_id=manifest.dataset_id,
                recording_id=recording_id,
                sensor="magnetometer",
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

    comparison = compare_public_magnetometer_states(tuple(snapshots))
    tool_trace.append(
        PublicSensorToolExecution(
            sequence=len(tool_trace) + 1,
            tool_id=COMPARE_TOOL_ID,
            evidence_ids=comparison.evidence_ids,
            result_codes=comparison.result_codes,
        )
    )
    return tuple(snapshots), comparison, tuple(tool_trace)


def compare_public_magnetometer_states(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
) -> PublicSensorComparison:
    """Evaluate stable, changing, or paired field windows without causal claims."""

    if not evidence or len(evidence) != len({item.evidence_id for item in evidence}):
        raise ValueError("magnetometer comparison requires unique evidence")
    if any(item.sensor != "magnetometer" or item.gate_c_eligible for item in evidence):
        raise ValueError("magnetometer comparison accepts only public non-Gate-C evidence")

    by_recording = {item.recording_id: item for item in evidence}
    if set(by_recording) not in (
        {ANCHOR_RECORDING_ID},
        {CHANGE_RECORDING_ID},
        {ANCHOR_RECORDING_ID, CHANGE_RECORDING_ID},
    ):
        raise ValueError("magnetometer comparison route is outside the frozen pair")

    metrics: list[PublicSensorComparisonMetric] = []
    result_codes: list[str] = []
    anchor_passed: bool | None = None
    change_passed: bool | None = None

    anchor = by_recording.get(ANCHOR_RECORDING_ID)
    if anchor is not None:
        std = anchor.analysis.metric_value("field_magnitude_std_ut")
        peak_to_peak = anchor.analysis.metric_value("field_peak_to_peak_ut")
        deviation = anchor.analysis.metric_value("max_field_deviation_ut")
        mean = anchor.analysis.metric_value("mean_field_magnitude_ut")
        anchor_passed = (
            _confidence_passed(anchor.analysis.confidence)
            and std <= MAX_STABLE_STD_UT
            and peak_to_peak <= MAX_STABLE_PEAK_TO_PEAK_UT
            and deviation <= MAX_STABLE_DEVIATION_UT
        )
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="stable_mean_field_ut",
                    label="稳定窗口平均场强模长",
                    value=mean,
                    unit="uT",
                ),
                PublicSensorComparisonMetric(
                    key="stable_field_std_ut",
                    label="稳定窗口场强标准差",
                    value=std,
                    unit="uT",
                ),
                PublicSensorComparisonMetric(
                    key="stable_field_peak_to_peak_ut",
                    label="稳定窗口场强峰峰值",
                    value=peak_to_peak,
                    unit="uT",
                ),
            )
        )
        result_codes.append(
            "stable_field_gate_passed" if anchor_passed else "stable_field_gate_failed"
        )

    change = by_recording.get(CHANGE_RECORDING_ID)
    if change is not None:
        std = change.analysis.metric_value("field_magnitude_std_ut")
        peak_to_peak = change.analysis.metric_value("field_peak_to_peak_ut")
        deviation = change.analysis.metric_value("max_field_deviation_ut")
        mean = change.analysis.metric_value("mean_field_magnitude_ut")
        change_passed = (
            _confidence_passed(change.analysis.confidence)
            and std >= MIN_CHANGE_STD_UT
            and peak_to_peak >= MIN_CHANGE_PEAK_TO_PEAK_UT
            and deviation >= MIN_CHANGE_DEVIATION_UT
        )
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="changing_mean_field_ut",
                    label="变化窗口平均场强模长",
                    value=mean,
                    unit="uT",
                ),
                PublicSensorComparisonMetric(
                    key="changing_field_std_ut",
                    label="变化窗口场强标准差",
                    value=std,
                    unit="uT",
                ),
                PublicSensorComparisonMetric(
                    key="changing_field_peak_to_peak_ut",
                    label="变化窗口场强峰峰值",
                    value=peak_to_peak,
                    unit="uT",
                ),
                PublicSensorComparisonMetric(
                    key="changing_max_deviation_ut",
                    label="变化窗口相对中位数最大偏差",
                    value=deviation,
                    unit="uT",
                ),
            )
        )
        result_codes.append(
            "field_change_gate_passed" if change_passed else "field_change_gate_failed"
        )

    if anchor is not None and change is not None:
        anchor_range = anchor.analysis.metric_value("field_peak_to_peak_ut")
        change_range = change.analysis.metric_value("field_peak_to_peak_ut")
        ratio = change_range / max(anchor_range, 1e-12)
        ratio_passed = ratio >= MIN_VARIABILITY_RATIO
        quality_passed = bool(anchor_passed and change_passed and ratio_passed)
        metrics.append(
            PublicSensorComparisonMetric(
                key="field_variability_ratio",
                label="变化与稳定窗口峰峰值比",
                value=ratio,
                unit="ratio",
            )
        )
        result_codes.append(
            "field_variability_separation_passed"
            if ratio_passed
            else "field_variability_separation_failed"
        )
        interpretation = (
            "同一公开 Pixel XL acquisition 的稳定窗口与场变化窗口通过预注册质量门，"
            "只支持磁力计对局部场变化有明显响应的描述性结论。"
            if quality_passed
            else "稳定性、场变化或分离度门未同时通过，需要真机受控扫描后再判断。"
        )
        comparison_id = "magnetometer-stable-changing-contrast"
    elif anchor is not None:
        quality_passed = bool(anchor_passed)
        interpretation = (
            "公开稳定窗口通过低波动质量门，只能作为同 acquisition 的背景参考。"
            if quality_passed
            else "公开稳定窗口未通过预注册低波动质量门。"
        )
        comparison_id = "magnetometer-stable-field-check"
    else:
        quality_passed = bool(change_passed)
        interpretation = (
            "公开变化窗口通过场变化质量门，只说明该记录中存在明显变化候选。"
            if quality_passed
            else "公开变化窗口未通过预注册场变化质量门。"
        )
        comparison_id = "magnetometer-field-change-check"

    return PublicSensorComparison(
        comparison_id=comparison_id,
        sensor="magnetometer",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=tuple(result_codes),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=tuple(metrics),
        interpretation=interpretation,
    )
