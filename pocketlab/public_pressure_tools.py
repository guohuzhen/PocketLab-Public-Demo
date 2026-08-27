from __future__ import annotations

from typing import cast

import numpy as np

from pocketlab.analyzers.scalar import PressureAnalyzer
from pocketlab.public_pressure_models import (
    PressureClaimKind,
    PressureDirection,
    PublicPressureClaimAuditResult,
    PublicPressureGroundTruth,
    PublicPressureHeightComparison,
    PublicPressureLoopClosureAudit,
    PublicPressureMetric,
    PublicPressurePlatformSummary,
    PublicPressureTrace,
    PublicPressureTraceResult,
)
from pocketlab.sensor_models import SensorRecordingUpload

PLATFORM_TARGET_DURATION_S = 3.00
PLATFORM_MINIMUM_DURATION_S = 2.00
PLATFORM_MINIMUM_SAMPLES = 5
PLATFORM_PRESSURE_MAD_MAX_HPA = 0.05
PLATFORM_PRESSURE_RANGE_MAX_HPA = 0.15
PLATFORM_PRESSURE_SLOPE_MAX_HPA_PER_MIN = 0.10
GROUND_TRUTH_ENDPOINT_MAX_LAG_S = 10.0
ANCHOR_PRESSURE_WINDOW_S = 1.0
ANCHOR_MINIMUM_PRESSURE_SAMPLES = 3
DIRECTION_DEADBAND_M = 0.50
HEIGHT_MINIMUM_GROUND_TRUTH_DISPLACEMENT_M = 3.00
HEIGHT_ABSOLUTE_TOLERANCE_M = 1.50
HEIGHT_RELATIVE_TOLERANCE = 0.20
LOOP_GROUND_TRUTH_CLOSURE_MAX_M = 0.75
LOOP_CLOSURE_TOLERANCE_M = 1.50
LOOP_MINIMUM_EXCURSION_M = 3.00
LOOP_MINIMUM_PRESSURE_EXCURSION_M = 2.00
LOOP_MINIMUM_TRANSITIONS = 2
LOOP_DIRECTION_AGREEMENT_MIN = 0.80
LOOP_EXCURSION_RATIO_MIN = 0.50
LOOP_EXCURSION_RATIO_MAX = 1.50

_HEIGHT_SCALE_M = PressureAnalyzer._HEIGHT_SCALE_M
_PRESSURE_EXPONENT = PressureAnalyzer._PRESSURE_EXPONENT

_COMMON_LIMITATIONS = (
    "The standard-atmosphere conversion is a near-surface approximation, not absolute altitude.",
    "Weather, HVAC, temperature, enclosure effects, and sensor drift can change pressure.",
    "Public Android replay is not the user's phyphox run and cannot establish Gate C.",
    "This offline contract does not establish market validation or agent readiness.",
)


def _require_trace(trace: PublicPressureTrace) -> PublicPressureTrace:
    if not isinstance(trace, PublicPressureTrace):
        raise TypeError("trace must be a validated PublicPressureTrace")
    # frozen models can still be bypassed with model_copy(update=...), so every
    # public tool revalidates the complete payload at its trust boundary.
    return PublicPressureTrace.model_validate(trace.model_dump(mode="python"))


def _require_ground_truth(
    ground_truth: PublicPressureGroundTruth,
) -> PublicPressureGroundTruth:
    if not isinstance(ground_truth, PublicPressureGroundTruth):
        raise TypeError("ground_truth must be a validated PublicPressureGroundTruth")
    return PublicPressureGroundTruth.model_validate(
        ground_truth.model_dump(mode="python")
    )


def _require_matching_evidence(
    trace: PublicPressureTrace,
    ground_truth: PublicPressureGroundTruth,
) -> tuple[PublicPressureTrace, PublicPressureGroundTruth]:
    trace = _require_trace(trace)
    ground_truth = _require_ground_truth(ground_truth)
    if trace.lineage != ground_truth.lineage:
        raise ValueError("pressure replay and ground-truth lineage must match exactly")
    if ground_truth.anchors[-1].relative_time_s > trace.samples[-1].relative_time_s:
        raise ValueError("ground-truth anchors must fall within the pressure replay")
    return trace, ground_truth


def _arrays(trace: PublicPressureTrace) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(
        [sample.relative_time_s for sample in trace.samples],
        dtype=np.float64,
    )
    pressures = np.asarray(
        [sample.pressure_hpa for sample in trace.samples],
        dtype=np.float64,
    )
    return times, pressures


def _build_pressure_upload(trace: PublicPressureTrace) -> SensorRecordingUpload:
    return SensorRecordingUpload.model_validate(
        {
            "label": trace.lineage.candidate_id,
            "device": "deidentified-public-phone",
            "sensor": "pressure",
            "channels": {"pressure": {"unit": "hPa"}},
            "samples": [
                {
                    "timestamp_ms": sample.relative_time_s * 1_000.0,
                    "values": {"pressure": sample.pressure_hpa},
                }
                for sample in trace.samples
            ],
            # This upload is an internal adapter into the scalar analyzer, not the
            # public-source trust boundary. The independently revalidated lineage
            # on PublicPressureTrace carries the real replay identity.
            "provenance": {"source": "test_fixture"},
        }
    )


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _direction(height_change_m: float) -> PressureDirection:
    if height_change_m > DIRECTION_DEADBAND_M:
        return "ascending"
    if height_change_m < -DIRECTION_DEADBAND_M:
        return "descending"
    return "level"


def _height_from_reference_pressure(
    pressure_hpa: np.ndarray | float,
    reference_pressure_hpa: float,
) -> np.ndarray | float:
    return _HEIGHT_SCALE_M * (
        1.0
        - (np.asarray(pressure_hpa) / reference_pressure_hpa) ** _PRESSURE_EXPONENT
    )


def _endpoint_indices(times: np.ndarray, *, start: bool) -> np.ndarray:
    if start:
        return np.flatnonzero(times <= times[0] + PLATFORM_TARGET_DURATION_S)
    return np.flatnonzero(times >= times[-1] - PLATFORM_TARGET_DURATION_S)


def _platform_summary(
    times: np.ndarray,
    pressures: np.ndarray,
) -> PublicPressurePlatformSummary:
    elapsed = times - times[0]
    duration = float(times[-1] - times[0])
    slope_hpa_per_min = float(np.polyfit(elapsed, pressures, deg=1)[0] * 60.0)
    pressure_mad = _mad(pressures)
    pressure_range = float(np.ptp(pressures))
    duration_passed = (
        len(pressures) >= PLATFORM_MINIMUM_SAMPLES
        and duration >= PLATFORM_MINIMUM_DURATION_S
    )
    pressure_stable = (
        pressure_mad <= PLATFORM_PRESSURE_MAD_MAX_HPA
        and abs(slope_hpa_per_min) <= PLATFORM_PRESSURE_SLOPE_MAX_HPA_PER_MIN
    )
    contamination_passed = pressure_range <= PLATFORM_PRESSURE_RANGE_MAX_HPA + 1e-12
    quality = (
        "good"
        if duration_passed and pressure_stable and contamination_passed
        else "poor"
    )
    return PublicPressurePlatformSummary(
        sample_count=len(pressures),
        start_time_s=float(times[0]),
        end_time_s=float(times[-1]),
        duration_s=duration,
        median_pressure_hpa=float(np.median(pressures)),
        pressure_mad_hpa=pressure_mad,
        pressure_range_hpa=pressure_range,
        pressure_slope_hpa_per_min=slope_hpa_per_min,
        duration_passed=duration_passed,
        pressure_stable=pressure_stable,
        contamination_passed=contamination_passed,
        quality=quality,
    )


def _endpoint_metrics(
    analysis_metrics: list[PublicPressureMetric],
    start: PublicPressurePlatformSummary,
    end: PublicPressurePlatformSummary,
) -> tuple[PublicPressureMetric, ...]:
    pressure_change = end.median_pressure_hpa - start.median_pressure_hpa
    height_change = float(
        _height_from_reference_pressure(
            end.median_pressure_hpa,
            start.median_pressure_hpa,
        )
    )
    replacements = {
        "reference_pressure_hpa": (start.median_pressure_hpa, "hPa"),
        "end_pressure_hpa": (end.median_pressure_hpa, "hPa"),
        "pressure_change_hpa": (pressure_change, "hPa"),
        "relative_height_change_m": (height_change, "m"),
    }
    return tuple(
        PublicPressureMetric(
            key=metric.key,
            value=replacements.get(metric.key, (metric.value, metric.unit))[0],
            unit=replacements.get(metric.key, (metric.value, metric.unit))[1],
        )
        for metric in analysis_metrics
    )


def inspect_pressure_trace(trace: PublicPressureTrace) -> PublicPressureTraceResult:
    """Inspect pressure only; sparse evaluation labels never enter this result."""

    trace = _require_trace(trace)
    times, pressures = _arrays(trace)
    analysis = PressureAnalyzer().analyze(_build_pressure_upload(trace))
    start_indices = _endpoint_indices(times, start=True)
    end_indices = _endpoint_indices(times, start=False)
    start = _platform_summary(times[start_indices], pressures[start_indices])
    end = _platform_summary(times[end_indices], pressures[end_indices])
    platforms_passed = start.quality == "good" and end.quality == "good"
    warnings = list(analysis.warnings)
    if start.quality != "good":
        warnings.append("The start endpoint is not a stable pressure platform.")
    if end.quality != "good":
        warnings.append("The end endpoint is not a stable pressure platform.")

    metrics = _endpoint_metrics(
        [
            PublicPressureMetric(
                key=metric.key,
                value=float(metric.value),
                unit=metric.unit,
            )
            for metric in analysis.metrics
        ],
        start,
        end,
    )
    metric_values = {metric.key: metric.value for metric in metrics}
    estimated_height = metric_values["relative_height_change_m"]
    return PublicPressureTraceResult(
        source_id=trace.lineage.source_id,
        candidate_id=trace.lineage.candidate_id,
        sample_count=len(trace.samples),
        duration_s=float(analysis.duration_s),
        confidence=analysis.confidence if platforms_passed else "low",
        warnings=tuple(warnings),
        metrics=metrics,
        start_platform=start,
        end_platform=end,
        pressure_change_hpa=metric_values["pressure_change_hpa"],
        standard_atmosphere_height_change_m=estimated_height,
        pressure_direction=_direction(estimated_height),
        platforms_passed=platforms_passed,
        evaluation_ready=platforms_passed,
        claim_boundary=_COMMON_LIMITATIONS,
    )


def _ground_truth_endpoint_requirements(
    trace: PublicPressureTrace,
    ground_truth: PublicPressureGroundTruth,
) -> list[str]:
    missing: list[str] = []
    duration = trace.samples[-1].relative_time_s
    first = ground_truth.anchors[0].relative_time_s
    last = ground_truth.anchors[-1].relative_time_s
    if first > GROUND_TRUTH_ENDPOINT_MAX_LAG_S:
        missing.append("A ground-truth anchor is required near the start endpoint.")
    if duration - last > GROUND_TRUTH_ENDPOINT_MAX_LAG_S:
        missing.append("A ground-truth anchor is required near the end endpoint.")
    return missing


def compare_pressure_height_to_ground_truth(
    trace: PublicPressureTrace,
    ground_truth: PublicPressureGroundTruth,
) -> PublicPressureHeightComparison:
    """Compare stable pressure endpoints against separate sparse source labels."""

    trace, ground_truth = _require_matching_evidence(trace, ground_truth)
    inspection = inspect_pressure_trace(trace)
    missing: list[str] = []
    if not inspection.start_platform.pressure_stable:
        missing.append("A stable start pressure platform is required.")
    if not inspection.end_platform.pressure_stable:
        missing.append("A stable end pressure platform is required.")
    if not inspection.start_platform.duration_passed:
        missing.append("The start pressure platform is too short.")
    if not inspection.end_platform.duration_passed:
        missing.append("The end pressure platform is too short.")
    if not inspection.start_platform.contamination_passed:
        missing.append("The start pressure platform contains a transition or outlier.")
    if not inspection.end_platform.contamination_passed:
        missing.append("The end pressure platform contains a transition or outlier.")
    missing.extend(_ground_truth_endpoint_requirements(trace, ground_truth))

    estimated = inspection.standard_atmosphere_height_change_m
    ground_truth_height = (
        ground_truth.anchors[-1].relative_elevation_m
        - ground_truth.anchors[0].relative_elevation_m
    )
    if abs(ground_truth_height) < HEIGHT_MINIMUM_GROUND_TRUTH_DISPLACEMENT_M:
        missing.append(
            "Ground-truth endpoint displacement is below the preregistered minimum."
        )
    if missing:
        return PublicPressureHeightComparison(
            status="not_evaluable",
            evaluable=False,
            passed=False,
            platforms_passed=inspection.platforms_passed,
            ground_truth_available=True,
            estimated_height_change_m=estimated,
            minimum_displacement_m=HEIGHT_MINIMUM_GROUND_TRUTH_DISPLACEMENT_M,
            estimated_direction=inspection.pressure_direction,
            missing_requirements=tuple(missing),
            limitations=_COMMON_LIMITATIONS,
        )

    ground_truth_direction = _direction(ground_truth_height)
    signed_error = estimated - ground_truth_height
    absolute_error = abs(signed_error)
    tolerance = max(
        HEIGHT_ABSOLUTE_TOLERANCE_M,
        HEIGHT_RELATIVE_TOLERANCE * abs(ground_truth_height),
    )
    direction_agreement = inspection.pressure_direction == ground_truth_direction
    passed = absolute_error <= tolerance and direction_agreement
    return PublicPressureHeightComparison(
        status="within_tolerance" if passed else "outside_tolerance",
        evaluable=True,
        passed=passed,
        platforms_passed=True,
        ground_truth_available=True,
        estimated_height_change_m=estimated,
        ground_truth_height_change_m=ground_truth_height,
        signed_error_m=signed_error,
        absolute_error_m=absolute_error,
        tolerance_m=tolerance,
        minimum_displacement_m=HEIGHT_MINIMUM_GROUND_TRUTH_DISPLACEMENT_M,
        estimated_direction=inspection.pressure_direction,
        ground_truth_direction=ground_truth_direction,
        direction_agreement=direction_agreement,
        missing_requirements=(),
        limitations=_COMMON_LIMITATIONS,
    )


def _pressure_at_anchors(
    times: np.ndarray,
    pressures: np.ndarray,
    ground_truth: PublicPressureGroundTruth,
) -> tuple[np.ndarray | None, list[str]]:
    anchor_pressures: list[float] = []
    missing: list[str] = []
    for anchor in ground_truth.anchors:
        indices = np.flatnonzero(
            np.abs(times - anchor.relative_time_s) <= ANCHOR_PRESSURE_WINDOW_S
        )
        if len(indices) < ANCHOR_MINIMUM_PRESSURE_SAMPLES:
            missing.append(
                f"Ground-truth dot {anchor.dot_index} lacks a robust pressure neighborhood."
            )
            continue
        anchor_pressures.append(float(np.median(pressures[indices])))
    if missing:
        return None, missing
    return np.asarray(anchor_pressures, dtype=np.float64), []


def audit_pressure_loop_closure(
    trace: PublicPressureTrace,
    ground_truth: PublicPressureGroundTruth,
) -> PublicPressureLoopClosureAudit:
    """Audit endpoint closure plus robust direction and excursion agreement."""

    trace, ground_truth = _require_matching_evidence(trace, ground_truth)
    inspection = inspect_pressure_trace(trace)
    times, pressures = _arrays(trace)
    anchor_pressures, anchor_missing = _pressure_at_anchors(
        times,
        pressures,
        ground_truth,
    )
    truth_heights = np.asarray(
        [anchor.relative_elevation_m for anchor in ground_truth.anchors],
        dtype=np.float64,
    )
    ground_truth_excursion = float(np.ptp(truth_heights))
    ground_truth_closure = float(truth_heights[-1] - truth_heights[0])
    loop_confirmed = bool(
        ground_truth_excursion >= LOOP_MINIMUM_EXCURSION_M
        and abs(ground_truth_closure) <= LOOP_GROUND_TRUTH_CLOSURE_MAX_M
    )
    pressure_closure = inspection.standard_atmosphere_height_change_m

    pressure_excursion = 0.0
    transition_count = 0
    direction_agreement_rate: float | None = None
    excursion_ratio: float | None = None
    trajectory_missing = list(anchor_missing)
    if anchor_pressures is not None:
        pressure_heights = np.asarray(
            _height_from_reference_pressure(anchor_pressures, anchor_pressures[0]),
            dtype=np.float64,
        )
        pressure_excursion = float(np.ptp(pressure_heights))
        truth_deltas = np.diff(truth_heights)
        pressure_deltas = np.diff(pressure_heights)
        meaningful = np.abs(truth_deltas) > DIRECTION_DEADBAND_M
        transition_count = int(np.count_nonzero(meaningful))
        if transition_count:
            agreement = np.sign(pressure_deltas[meaningful]) == np.sign(
                truth_deltas[meaningful]
            )
            direction_agreement_rate = float(np.mean(agreement))
        if ground_truth_excursion > 0.0:
            excursion_ratio = pressure_excursion / ground_truth_excursion

    missing: list[str] = []
    if not inspection.platforms_passed:
        missing.append("Stable start and end pressure platforms are required.")
    missing.extend(_ground_truth_endpoint_requirements(trace, ground_truth))
    if ground_truth_excursion < LOOP_MINIMUM_EXCURSION_M:
        missing.append("Ground truth does not contain the minimum vertical excursion.")
    if abs(ground_truth_closure) > LOOP_GROUND_TRUTH_CLOSURE_MAX_M:
        missing.append("Ground truth does not return to the starting elevation.")
    if pressure_excursion < LOOP_MINIMUM_PRESSURE_EXCURSION_M:
        missing.append("Pressure does not contain a resolved robust vertical excursion.")
    missing.extend(trajectory_missing)
    if transition_count < LOOP_MINIMUM_TRANSITIONS:
        missing.append("The loop does not contain enough evaluable vertical transitions.")
    if (
        direction_agreement_rate is not None
        and direction_agreement_rate < LOOP_DIRECTION_AGREEMENT_MIN
    ):
        missing.append("Pressure and ground-truth transition directions do not agree.")
    if excursion_ratio is not None and not (
        LOOP_EXCURSION_RATIO_MIN <= excursion_ratio <= LOOP_EXCURSION_RATIO_MAX
    ):
        missing.append("Pressure and ground-truth excursion magnitudes do not agree.")

    if missing:
        return PublicPressureLoopClosureAudit(
            status="not_evaluable",
            evaluable=False,
            passed=False,
            platforms_passed=inspection.platforms_passed,
            ground_truth_available=True,
            ground_truth_loop_confirmed=loop_confirmed,
            pressure_excursion_m=pressure_excursion,
            ground_truth_excursion_m=ground_truth_excursion,
            pressure_closure_height_change_m=pressure_closure,
            ground_truth_closure_height_change_m=ground_truth_closure,
            transition_count=transition_count,
            closure_tolerance_m=LOOP_CLOSURE_TOLERANCE_M,
            minimum_excursion_m=LOOP_MINIMUM_EXCURSION_M,
            missing_requirements=tuple(dict.fromkeys(missing)),
            limitations=_COMMON_LIMITATIONS,
        )

    signed_error = pressure_closure - ground_truth_closure
    absolute_error = abs(signed_error)
    passed = absolute_error <= LOOP_CLOSURE_TOLERANCE_M
    return PublicPressureLoopClosureAudit(
        status="closed_within_tolerance" if passed else "drift_detected",
        evaluable=True,
        passed=passed,
        platforms_passed=True,
        ground_truth_available=True,
        ground_truth_loop_confirmed=True,
        pressure_excursion_m=pressure_excursion,
        ground_truth_excursion_m=ground_truth_excursion,
        pressure_closure_height_change_m=pressure_closure,
        ground_truth_closure_height_change_m=ground_truth_closure,
        signed_closure_error_m=signed_error,
        absolute_closure_error_m=absolute_error,
        transition_count=transition_count,
        direction_agreement_rate=cast(float, direction_agreement_rate),
        excursion_ratio=cast(float, excursion_ratio),
        closure_tolerance_m=LOOP_CLOSURE_TOLERANCE_M,
        minimum_excursion_m=LOOP_MINIMUM_EXCURSION_M,
        missing_requirements=(),
        limitations=_COMMON_LIMITATIONS,
    )


def audit_pressure_claim_support(
    claim_kind: PressureClaimKind,
    trace: PublicPressureTrace,
    ground_truth: PublicPressureGroundTruth | None = None,
) -> PublicPressureClaimAuditResult:
    """Allow only bounded claims and keep reportability separate from success."""

    known_claims = {
        "descriptive_pressure_change",
        "height_change_against_ground_truth",
        "loop_closure",
        "absolute_altitude",
        "causal_vertical_motion",
        "device_calibration",
        "market_validation",
    }
    if claim_kind not in known_claims:
        raise ValueError(f"unknown public Pressure claim kind: {claim_kind}")
    trace = _require_trace(trace)
    forbidden_common = (
        "Do not claim absolute altitude from this relative pressure contract.",
        "Do not attribute pressure change solely to vertical motion.",
        "Do not claim Gate C, market validation, or agent readiness.",
    )
    if claim_kind in {
        "absolute_altitude",
        "causal_vertical_motion",
        "device_calibration",
        "market_validation",
    }:
        missing_by_claim = {
            "absolute_altitude": (
                "Traceable absolute altitude and atmospheric calibration references are required.",
            ),
            "causal_vertical_motion": (
                "Controlled weather/HVAC/drift checks and independent motion evidence are required.",
            ),
            "device_calibration": (
                "A traceable reference barometer and preregistered calibration procedure are required.",
            ),
            "market_validation": (
                "Representative real-device, user, failure, and end-to-end Agent evaluations are required.",
            ),
        }
        return PublicPressureClaimAuditResult(
            claim_kind=claim_kind,
            status="unsupported",
            evaluation_outcome="not_applicable",
            allowed_phrasing=(),
            forbidden_phrasing=forbidden_common,
            required_missing_evidence=missing_by_claim[claim_kind],
        )

    if claim_kind == "descriptive_pressure_change":
        inspection = inspect_pressure_trace(trace)
        supported = inspection.evaluation_ready
        return PublicPressureClaimAuditResult(
            claim_kind=claim_kind,
            status="supported_with_limitations" if supported else "unsupported",
            evaluation_outcome="passed" if supported else "not_evaluable",
            allowed_phrasing=(
                "Stable endpoint pressures differ by the reported amount, with a standard-atmosphere approximate relative-height equivalent.",
            )
            if supported
            else (),
            forbidden_phrasing=forbidden_common,
            required_missing_evidence=()
            if supported
            else ("Stable start and end pressure platforms are required.",),
        )

    if ground_truth is None:
        return PublicPressureClaimAuditResult(
            claim_kind=claim_kind,
            status="unsupported",
            evaluation_outcome="not_evaluable",
            allowed_phrasing=(),
            forbidden_phrasing=forbidden_common,
            required_missing_evidence=(
                "A separate server-side ground-truth anchor artifact is required.",
            ),
        )

    if claim_kind == "height_change_against_ground_truth":
        comparison = compare_pressure_height_to_ground_truth(trace, ground_truth)
        supported = comparison.evaluable
        return PublicPressureClaimAuditResult(
            claim_kind=claim_kind,
            status="supported_with_limitations" if supported else "unsupported",
            evaluation_outcome=(
                "passed"
                if comparison.passed
                else "failed"
                if supported
                else "not_evaluable"
            ),
            allowed_phrasing=(
                "The standard-atmosphere estimate may be reported as within or outside the preregistered error tolerance against sparse relative ground-truth anchors.",
            )
            if supported
            else (),
            forbidden_phrasing=forbidden_common,
            required_missing_evidence=()
            if supported
            else comparison.missing_requirements,
        )

    closure = audit_pressure_loop_closure(trace, ground_truth)
    supported = closure.evaluable
    return PublicPressureClaimAuditResult(
        claim_kind=claim_kind,
        status="supported_with_limitations" if supported else "unsupported",
        evaluation_outcome=(
            "passed"
            if closure.passed
            else "failed"
            if supported
            else "not_evaluable"
        ),
        allowed_phrasing=(
            "The registered relative-elevation loop may be reported as closing within tolerance or showing endpoint drift after trajectory checks.",
        )
        if supported
        else (),
        forbidden_phrasing=forbidden_common,
        required_missing_evidence=()
        if supported
        else closure.missing_requirements,
    )
