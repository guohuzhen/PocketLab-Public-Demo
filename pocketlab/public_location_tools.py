from __future__ import annotations

import math
import statistics
from itertools import pairwise
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
from pocketlab.sensor_models import SensorRecordingUpload

DATASET_ID: Final = "location-uci-gps-trajectories-20160228-v1"
ROUTE_A_RECORDING_ID: Final = "go-track-repeated-route-a"
ROUTE_B_RECORDING_ID: Final = "go-track-repeated-route-b"

ANALYZE_TOOL_ID: Final = "analyze_location_relative_route"
COMPARE_TOOL_ID: Final = "compare_location_repeated_routes"

EARTH_MEAN_RADIUS_M: Final = 6_371_008.8
MINIMUM_SAMPLE_COUNT: Final = 50
MINIMUM_DURATION_S: Final = 300.0
MINIMUM_ROUTE_DISTANCE_M: Final = 3_000.0
MAXIMUM_ROUTE_DISTANCE_M: Final = 3_400.0
MINIMUM_PATH_EFFICIENCY: Final = 0.60
MAXIMUM_PATH_EFFICIENCY: Final = 0.80
MAXIMUM_GAP_RATIO: Final = 2.25
MAXIMUM_LENGTH_DIFFERENCE_PERCENT: Final = 2.0
MAXIMUM_MEDIAN_NEAREST_DISTANCE_M: Final = 40.0
MAXIMUM_P95_NEAREST_DISTANCE_M: Final = 75.0
MAXIMUM_RELATIVE_ENDPOINT_SEPARATION_M: Final = 50.0

REQUIRED_INVALIDATED_METRICS: Final = {
    "median_horizontal_accuracy_m",
    "max_horizontal_accuracy_m",
    "mean_reported_speed_m_s",
    "altitude_change_m",
    "altitude_span_m",
}


def _recording_gate(evidence: PublicSensorEvidenceSnapshot) -> bool:
    analysis = evidence.analysis
    return (
        analysis.sample_count >= MINIMUM_SAMPLE_COUNT
        and analysis.duration_s >= MINIMUM_DURATION_S
        and analysis.max_sampling_gap_ratio <= MAXIMUM_GAP_RATIO
        and analysis.confidence == "medium"
        and MINIMUM_ROUTE_DISTANCE_M
        <= analysis.metric_value("trajectory_distance_m")
        <= MAXIMUM_ROUTE_DISTANCE_M
        and MINIMUM_PATH_EFFICIENCY
        <= analysis.metric_value("path_efficiency_ratio")
        <= MAXIMUM_PATH_EFFICIENCY
        and any("status" in warning for warning in analysis.warnings)
        and any("accuracy" in warning for warning in analysis.warnings)
        and any("synthetic zero-origin" in warning for warning in analysis.warnings)
    )


def _route_points_m(upload: SensorRecordingUpload) -> tuple[tuple[float, float], ...]:
    points = tuple(
        (
            EARTH_MEAN_RADIUS_M * math.radians(sample.values["lon"]),
            EARTH_MEAN_RADIUS_M * math.radians(sample.values["lat"]),
        )
        for sample in upload.samples
    )
    if not points or math.hypot(*points[0]) > 1e-6:
        raise ValueError("location replay does not start at its synthetic origin")
    if any(
        math.hypot(right[0] - left[0], right[1] - left[1]) > 500.0
        for left, right in pairwise(points)
    ):
        raise ValueError("location replay contains an unsupported local-grid jump")
    return points


def _nearest_distance(
    point: tuple[float, float],
    candidates: tuple[tuple[float, float], ...],
) -> float:
    return min(
        math.hypot(point[0] - other[0], point[1] - other[1])
        for other in candidates
    )


def _percentile_nearest(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def load_public_location_evidence(
    pack_dir: Path,
    recording_ids: tuple[str, ...],
) -> tuple[
    tuple[PublicSensorEvidenceSnapshot, ...],
    PublicSensorComparison,
    tuple[PublicSensorToolExecution, ...],
]:
    """Load only registered synthetic-origin route replays and apply frozen gates."""

    allowed = {ROUTE_A_RECORDING_ID, ROUTE_B_RECORDING_ID}
    if (
        not recording_ids
        or len(recording_ids) != len(set(recording_ids))
        or not set(recording_ids).issubset(allowed)
    ):
        raise ValueError("location recording route is outside the frozen protocol")

    manifest = load_public_replay_dataset(pack_dir)
    if (
        manifest.dataset_id != DATASET_ID
        or manifest.sensor != "location"
        or manifest.analyzer_id != "pocketlab.location.haversine.v1"
        or manifest.analyzer_version != "1.0.0"
        or manifest.source.source_id != "uci-gps-trajectories-c54s5z"
        or manifest.source.license_spdx != "CC-BY-4.0"
        or manifest.public_replay_status != "source_validated"
        or manifest.agent_ready
        or manifest.privacy_review.deployment_scope != "local_only"
        or not manifest.privacy_review.requires_user_acknowledgement
    ):
        raise ValueError("location public replay manifest contract changed")
    if manifest.source.doi != "10.24432/C54S5Z":
        raise ValueError("location source DOI changed")

    by_id = {item.recording_id: item for item in manifest.recordings}
    snapshots: list[PublicSensorEvidenceSnapshot] = []
    routes: dict[str, tuple[tuple[float, float], ...]] = {}
    tool_trace: list[PublicSensorToolExecution] = []
    for sequence, recording_id in enumerate(recording_ids, start=1):
        recording = by_id.get(recording_id)
        if recording is None:
            raise ValueError("location recording is missing from the reviewed manifest")
        if (
            not recording.independent_measurement
            or recording.gate_c_eligible
            or recording.analysis_confidence_ceiling != "medium"
            or recording.invalidated_analysis_fields
            or set(recording.invalidated_metric_keys) != REQUIRED_INVALIDATED_METRICS
        ):
            raise ValueError("location evidence boundary changed")
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        if not upload.provenance.privacy_acknowledged:
            raise ValueError("location public replay privacy gate was not applied")
        if set(upload.channels) != {"lat", "lon"}:
            raise ValueError("location replay exposed a non-frozen channel")
        routes[recording_id] = _route_points_m(upload)
        analysis = analyze_sensor_recording(upload)
        if analysis != recording.reference_analysis:
            raise ValueError("location analyzer drifted from the frozen reference")
        evidence_id = f"location-{recording_id}"
        snapshot = PublicSensorEvidenceSnapshot(
            evidence_id=evidence_id,
            dataset_id=manifest.dataset_id,
            recording_id=recording_id,
            sensor="location",
            data_class=manifest.data_class,
            condition_label=recording.label,
            device_scope=(
                f"{recording.device_alias}; synthetic zero-origin route only, "
                "without absolute coordinates, accuracy, or phyphox provenance"
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
        passed = _recording_gate(snapshot)
        tool_trace.append(
            PublicSensorToolExecution(
                sequence=sequence,
                tool_id=ANALYZE_TOOL_ID,
                status="completed" if passed else "rejected",
                evidence_ids=(evidence_id,),
                result_codes=(
                    "relative_route_quality_passed"
                    if passed
                    else "relative_route_quality_failed",
                    "synthetic_origin_verified",
                    "absolute_location_absent",
                    "accuracy_status_unavailable",
                ),
            )
        )

    comparison = compare_public_location_routes(tuple(snapshots), routes)
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


def _route_metrics(
    evidence: PublicSensorEvidenceSnapshot,
) -> list[PublicSensorComparisonMetric]:
    prefix = "route_a" if evidence.recording_id == ROUTE_A_RECORDING_ID else "route_b"
    analysis = evidence.analysis
    return [
        PublicSensorComparisonMetric(
            key=f"{prefix}_distance_m",
            label=f"{prefix} 轨迹长度",
            value=analysis.metric_value("trajectory_distance_m"),
            unit="m",
        ),
        PublicSensorComparisonMetric(
            key=f"{prefix}_average_speed_m_s",
            label=f"{prefix} 平均路径速率",
            value=analysis.metric_value("average_path_speed_m_s"),
            unit="m/s",
        ),
        PublicSensorComparisonMetric(
            key=f"{prefix}_efficiency_ratio",
            label=f"{prefix} 路径效率",
            value=analysis.metric_value("path_efficiency_ratio"),
            unit="1",
        ),
    ]


def compare_public_location_routes(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
    routes: dict[str, tuple[tuple[float, float], ...]],
) -> PublicSensorComparison:
    """Compare relative route geometry without returning or inferring absolute location."""

    if not evidence or len(evidence) != len({item.evidence_id for item in evidence}):
        raise ValueError("location comparison requires unique evidence")
    if any(item.sensor != "location" or item.gate_c_eligible for item in evidence):
        raise ValueError("location comparison accepts only public non-Gate-C evidence")
    by_recording = {item.recording_id: item for item in evidence}
    if set(by_recording) not in (
        {ROUTE_A_RECORDING_ID},
        {ROUTE_B_RECORDING_ID},
        {ROUTE_A_RECORDING_ID, ROUTE_B_RECORDING_ID},
    ):
        raise ValueError("location comparison route is outside the frozen pair")
    if set(routes) != set(by_recording):
        raise ValueError("location route geometry must match the exact evidence set")

    metrics: list[PublicSensorComparisonMetric] = []
    result_codes: list[str] = []
    route_gates: dict[str, bool] = {}
    for recording_id in (ROUTE_A_RECORDING_ID, ROUTE_B_RECORDING_ID):
        item = by_recording.get(recording_id)
        if item is None:
            continue
        passed = _recording_gate(item)
        route_gates[recording_id] = passed
        metrics.extend(_route_metrics(item))
        result_codes.append(
            f"{'route_a' if recording_id == ROUTE_A_RECORDING_ID else 'route_b'}_quality_"
            f"{'passed' if passed else 'failed'}"
        )

    if len(by_recording) == 2:
        analysis_a = by_recording[ROUTE_A_RECORDING_ID].analysis
        analysis_b = by_recording[ROUTE_B_RECORDING_ID].analysis
        distance_a = analysis_a.metric_value("trajectory_distance_m")
        distance_b = analysis_b.metric_value("trajectory_distance_m")
        length_difference_percent = (
            abs(distance_a - distance_b) / ((distance_a + distance_b) / 2.0) * 100.0
        )
        speed_ratio = analysis_b.metric_value(
            "average_path_speed_m_s"
        ) / analysis_a.metric_value("average_path_speed_m_s")
        efficiency_delta = abs(
            analysis_a.metric_value("path_efficiency_ratio")
            - analysis_b.metric_value("path_efficiency_ratio")
        )
        route_a = routes[ROUTE_A_RECORDING_ID]
        route_b = routes[ROUTE_B_RECORDING_ID]
        nearest = [_nearest_distance(point, route_b) for point in route_a]
        nearest.extend(_nearest_distance(point, route_a) for point in route_b)
        median_nearest = statistics.median(nearest)
        p95_nearest = _percentile_nearest(nearest, 0.95)
        endpoint_separation = math.hypot(
            route_a[-1][0] - route_b[-1][0], route_a[-1][1] - route_b[-1][1]
        )
        geometry_passed = (
            length_difference_percent <= MAXIMUM_LENGTH_DIFFERENCE_PERCENT
            and median_nearest <= MAXIMUM_MEDIAN_NEAREST_DISTANCE_M
            and p95_nearest <= MAXIMUM_P95_NEAREST_DISTANCE_M
            and endpoint_separation <= MAXIMUM_RELATIVE_ENDPOINT_SEPARATION_M
        )
        quality_passed = all(route_gates.values()) and geometry_passed
        metrics.extend(
            (
                PublicSensorComparisonMetric(
                    key="route_length_difference_percent",
                    label="重复路线长度差",
                    value=length_difference_percent,
                    unit="%",
                ),
                PublicSensorComparisonMetric(
                    key="route_b_over_a_speed_ratio",
                    label="B/A 平均路径速率比",
                    value=speed_ratio,
                    unit="1",
                ),
                PublicSensorComparisonMetric(
                    key="path_efficiency_delta",
                    label="路径效率绝对差",
                    value=efficiency_delta,
                    unit="1",
                ),
                PublicSensorComparisonMetric(
                    key="symmetric_median_nearest_distance_m",
                    label="相对轨迹对称最近点中位距离",
                    value=median_nearest,
                    unit="m",
                ),
                PublicSensorComparisonMetric(
                    key="symmetric_p95_nearest_distance_m",
                    label="相对轨迹对称最近点 P95 距离",
                    value=p95_nearest,
                    unit="m",
                ),
                PublicSensorComparisonMetric(
                    key="relative_endpoint_separation_m",
                    label="相对终点分离",
                    value=endpoint_separation,
                    unit="m",
                ),
            )
        )
        result_codes.append(
            "relative_route_geometry_gate_passed"
            if geometry_passed
            else "relative_route_geometry_gate_failed"
        )
        interpretation = (
            "两次独立 Go!Track acquisition 在隐私变换后的相对坐标中具有接近的路径长度、"
            "轨迹形状与终点，但平均路径速率不同；缺少 accuracy/status，因此只能描述重复路线"
            "的一致性，不能量化 GPS 绝对误差或归因于开阔/遮挡环境。"
            if quality_passed
            else "来源、单轨迹质量或相对路线几何门未同时通过。"
        )
        comparison_id = "location-relative-repeated-route-comparison"
    else:
        quality_passed = next(iter(route_gates.values()))
        side = "a" if ROUTE_A_RECORDING_ID in by_recording else "b"
        interpretation = (
            f"公开 Go!Track acquisition {side.upper()} 通过相对轨迹、时间轴与来源门；"
            "结果只描述隐私变换后的单次路径。"
            if quality_passed
            else "该公开相对轨迹未通过预注册来源或质量门。"
        )
        comparison_id = f"location-relative-route-{side}-check"

    return PublicSensorComparison(
        comparison_id=comparison_id,
        sensor="location",
        status="passed" if quality_passed else "failed",
        quality_passed=quality_passed,
        result_codes=tuple(result_codes),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        metrics=tuple(metrics),
        interpretation=interpretation,
    )
