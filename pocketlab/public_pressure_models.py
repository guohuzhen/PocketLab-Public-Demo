from __future__ import annotations

from itertools import pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PressureDirection = Literal["ascending", "descending", "level", "indeterminate"]
PressureComparisonStatus = Literal[
    "not_evaluable",
    "within_tolerance",
    "outside_tolerance",
]
PressureClosureStatus = Literal[
    "not_evaluable",
    "closed_within_tolerance",
    "drift_detected",
]
PressureClaimKind = Literal[
    "descriptive_pressure_change",
    "height_change_against_ground_truth",
    "loop_closure",
    "absolute_altitude",
    "causal_vertical_motion",
    "device_calibration",
    "market_validation",
]
PressureClaimSupportStatus = Literal["supported_with_limitations", "unsupported"]
PressureEvaluationOutcome = Literal[
    "passed",
    "failed",
    "not_evaluable",
    "not_applicable",
]

_OPAQUE_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,119}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class PublicPressureLineage(_FrozenStrictModel):
    """Frozen identity shared by the pressure replay and hidden eval labels."""

    source_id: str = Field(pattern=_OPAQUE_ID_PATTERN)
    candidate_id: str = Field(pattern=_OPAQUE_ID_PATTERN)
    source_recording_sha256: str = Field(pattern=_SHA256_PATTERN)
    transform_id: str = Field(pattern=_OPAQUE_ID_PATTERN)
    transform_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")


class PublicPressureSample(_FrozenStrictModel):
    """One de-identified pressure observation on relative axes only."""

    relative_time_s: float = Field(ge=0.0, le=86_400.0)
    pressure_hpa: float = Field(ge=100.0, le=1_200.0)


class PublicPressureTrace(_FrozenStrictModel):
    """Model-visible, pressure-only public replay boundary.

    Absolute timestamps, coordinates, device identifiers, free text, and ground-truth
    elevation are intentionally excluded. Ground truth lives in a separate
    server/eval-only artifact so it cannot leak into a Planner observation.
    """

    schema_version: Literal["1.0"] = "1.0"
    lineage: PublicPressureLineage
    replay_sha256: str = Field(pattern=_SHA256_PATTERN)
    samples: tuple[PublicPressureSample, ...] = Field(
        min_length=20,
        max_length=120_000,
    )

    @model_validator(mode="after")
    def relative_series_is_complete_and_ordered(self) -> Self:
        times = [sample.relative_time_s for sample in self.samples]
        if times[0] != 0.0:
            raise ValueError("relative_time_s must start at exactly 0.0")
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("relative_time_s must be strictly increasing")
        if times[-1] < 4.0:
            raise ValueError("public pressure trace must span at least 4.0 seconds")
        return self


class PublicPressureGroundTruthAnchor(_FrozenStrictModel):
    """One sparse, server-side evaluation label published by the source dataset."""

    dot_index: int = Field(ge=0, le=1_000_000)
    relative_time_s: float = Field(ge=0.0, le=86_400.0)
    relative_elevation_m: float = Field(ge=-1_000.0, le=1_000.0)


class PublicPressureGroundTruth(_FrozenStrictModel):
    """Sparse labels that must never be serialized into a Planner request."""

    schema_version: Literal["1.0"] = "1.0"
    lineage: PublicPressureLineage
    anchor_sha256: str = Field(pattern=_SHA256_PATTERN)
    anchors: tuple[PublicPressureGroundTruthAnchor, ...] = Field(
        min_length=2,
        max_length=10_000,
    )

    @model_validator(mode="after")
    def anchors_are_ordered(self) -> Self:
        dot_indices = [anchor.dot_index for anchor in self.anchors]
        times = [anchor.relative_time_s for anchor in self.anchors]
        if any(right <= left for left, right in pairwise(dot_indices)):
            raise ValueError("ground-truth dot_index values must be strictly increasing")
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("ground-truth relative_time_s values must be strictly increasing")
        return self


class PublicPressureMetric(_FrozenStrictModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    value: float
    unit: str = Field(min_length=1, max_length=24)


class PublicPressurePlatformSummary(_FrozenStrictModel):
    sample_count: int = Field(ge=5)
    start_time_s: float = Field(ge=0.0)
    end_time_s: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)
    median_pressure_hpa: float = Field(ge=100.0, le=1_200.0)
    pressure_mad_hpa: float = Field(ge=0.0)
    pressure_range_hpa: float = Field(ge=0.0)
    pressure_slope_hpa_per_min: float
    duration_passed: bool
    pressure_stable: bool
    contamination_passed: bool
    quality: Literal["good", "poor"]

    @model_validator(mode="after")
    def quality_matches_component_gates(self) -> Self:
        if self.duration_s != self.end_time_s - self.start_time_s:
            raise ValueError("platform duration does not match its endpoint times")
        expected = (
            self.duration_passed
            and self.pressure_stable
            and self.contamination_passed
        )
        if (self.quality == "good") != expected:
            raise ValueError("platform quality does not match its stability gates")
        return self


class PublicPressureTraceResult(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["public_pressure_trace"] = "public_pressure_trace"
    source_id: str = Field(pattern=_OPAQUE_ID_PATTERN)
    candidate_id: str = Field(pattern=_OPAQUE_ID_PATTERN)
    sample_count: int = Field(ge=20)
    duration_s: float = Field(ge=4.0)
    confidence: Literal["low", "medium", "high"]
    warnings: tuple[str, ...]
    metrics: tuple[PublicPressureMetric, ...] = Field(min_length=1)
    start_platform: PublicPressurePlatformSummary
    end_platform: PublicPressurePlatformSummary
    pressure_change_hpa: float
    standard_atmosphere_height_change_m: float
    pressure_direction: PressureDirection
    platforms_passed: bool
    evaluation_ready: bool
    claim_boundary: tuple[str, ...] = Field(min_length=1)
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def readiness_is_consistent(self) -> Self:
        expected = (
            self.start_platform.quality == "good"
            and self.end_platform.quality == "good"
        )
        if self.platforms_passed != expected:
            raise ValueError("platforms_passed does not match platform summaries")
        if self.evaluation_ready != self.platforms_passed:
            raise ValueError("trace evaluation requires two stable endpoint platforms")
        return self


class PublicPressureHeightComparison(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["pressure_height_ground_truth_comparison"] = (
        "pressure_height_ground_truth_comparison"
    )
    status: PressureComparisonStatus
    evaluable: bool
    passed: bool
    platforms_passed: bool
    ground_truth_available: bool
    estimated_height_change_m: float
    ground_truth_height_change_m: float | None = None
    signed_error_m: float | None = None
    absolute_error_m: float | None = Field(default=None, ge=0.0)
    tolerance_m: float | None = Field(default=None, gt=0.0)
    minimum_displacement_m: float = Field(gt=0.0)
    estimated_direction: PressureDirection
    ground_truth_direction: PressureDirection | None = None
    direction_agreement: bool | None = None
    missing_requirements: tuple[str, ...]
    limitations: tuple[str, ...] = Field(min_length=1)
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def comparison_state_is_consistent(self) -> Self:
        if self.evaluable != (self.status != "not_evaluable"):
            raise ValueError("comparison evaluable flag does not match status")
        if self.passed != (self.status == "within_tolerance"):
            raise ValueError("comparison passed flag does not match status")
        detailed = (
            self.ground_truth_height_change_m,
            self.signed_error_m,
            self.absolute_error_m,
            self.tolerance_m,
            self.ground_truth_direction,
            self.direction_agreement,
        )
        if self.evaluable:
            if not self.platforms_passed or not self.ground_truth_available:
                raise ValueError("evaluable comparison requires platforms and ground truth")
            if any(value is None for value in detailed):
                raise ValueError("evaluable comparison requires complete error fields")
            if self.missing_requirements:
                raise ValueError("evaluable comparison cannot report missing requirements")
        else:
            if not self.missing_requirements:
                raise ValueError("non-evaluable comparison requires an explicit reason")
            if any(value is not None for value in detailed):
                raise ValueError("non-evaluable comparison cannot expose partial eval labels")
        return self


class PublicPressureLoopClosureAudit(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["pressure_loop_closure_audit"] = "pressure_loop_closure_audit"
    status: PressureClosureStatus
    evaluable: bool
    passed: bool
    platforms_passed: bool
    ground_truth_available: bool
    ground_truth_loop_confirmed: bool
    pressure_excursion_m: float = Field(ge=0.0)
    ground_truth_excursion_m: float | None = Field(default=None, ge=0.0)
    pressure_closure_height_change_m: float
    ground_truth_closure_height_change_m: float | None = None
    signed_closure_error_m: float | None = None
    absolute_closure_error_m: float | None = Field(default=None, ge=0.0)
    transition_count: int = Field(ge=0)
    direction_agreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    excursion_ratio: float | None = Field(default=None, ge=0.0)
    closure_tolerance_m: float = Field(gt=0.0)
    minimum_excursion_m: float = Field(gt=0.0)
    missing_requirements: tuple[str, ...]
    limitations: tuple[str, ...] = Field(min_length=1)
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def closure_state_is_consistent(self) -> Self:
        if self.evaluable != (self.status != "not_evaluable"):
            raise ValueError("closure evaluable flag does not match status")
        if self.passed != (self.status == "closed_within_tolerance"):
            raise ValueError("closure passed flag does not match status")
        details = (
            self.ground_truth_excursion_m,
            self.ground_truth_closure_height_change_m,
            self.signed_closure_error_m,
            self.absolute_closure_error_m,
            self.direction_agreement_rate,
            self.excursion_ratio,
        )
        if self.evaluable:
            if not (
                self.platforms_passed
                and self.ground_truth_available
                and self.ground_truth_loop_confirmed
            ):
                raise ValueError("evaluable closure requires platforms and a ground-truth loop")
            if any(value is None for value in details):
                raise ValueError("evaluable closure requires complete trajectory fields")
            if self.missing_requirements:
                raise ValueError("evaluable closure cannot report missing requirements")
        else:
            if not self.missing_requirements:
                raise ValueError("non-evaluable closure requires an explicit reason")
            if any(
                value is not None
                for value in (
                    self.signed_closure_error_m,
                    self.absolute_closure_error_m,
                    self.direction_agreement_rate,
                    self.excursion_ratio,
                )
            ):
                raise ValueError("non-evaluable closure cannot expose partial verdict fields")
        return self


class PublicPressureClaimAuditResult(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["pressure_claim_audit"] = "pressure_claim_audit"
    claim_kind: PressureClaimKind
    status: PressureClaimSupportStatus
    evaluation_outcome: PressureEvaluationOutcome
    allowed_phrasing: tuple[str, ...]
    forbidden_phrasing: tuple[str, ...] = Field(min_length=1)
    required_missing_evidence: tuple[str, ...]
    gate_c_eligible: Literal[False] = False
    gate_e_status: Literal["not_evaluated"] = "not_evaluated"
    gate_h_status: Literal["not_evaluated"] = "not_evaluated"
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def support_state_is_consistent(self) -> Self:
        if self.status == "unsupported":
            if self.allowed_phrasing:
                raise ValueError("unsupported claims cannot have allowed phrasing")
            if not self.required_missing_evidence:
                raise ValueError("unsupported claims require missing evidence")
            if self.evaluation_outcome in {"passed", "failed"}:
                raise ValueError("unsupported claims cannot have a benchmark verdict")
        else:
            if not self.allowed_phrasing:
                raise ValueError("supported claims require bounded allowed phrasing")
            if self.required_missing_evidence:
                raise ValueError("supported claims cannot report missing evidence")
            if self.evaluation_outcome in {"not_evaluable", "not_applicable"}:
                raise ValueError("supported evaluable claims require pass/fail outcome")
        return self
