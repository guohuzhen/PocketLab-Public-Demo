from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from pocketlab.general_acquisition import GeneralEvidenceEnvelope
from pocketlab.general_exploration_models import (
    GeneralExperimentProtocol,
    StrictFrozenModel,
)
from pocketlab.reality_feedback import RealityFeedbackRecord
from pocketlab.sensor_models import SensorKind

_IDENTIFIER = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
_SHA256 = r"^[0-9a-f]{64}$"

GeneralTaskAction = Literal[
    "collect_condition",
    "collect_supporting_sensor",
    "replicate_condition",
    "correct_condition",
    "probe_optional_sensor",
    "probe_optional_condition",
]
GeneralTaskReason = Literal[
    "initial_baseline",
    "missing_condition",
    "missing_supporting_sensor",
    "replication_required",
    "quality_correction",
    "optional_sensor_probe",
    "optional_condition_probe",
]
GeneralPlannerRationaleCode = Literal[
    "maximize_condition_coverage",
    "balance_sensor_coverage",
    "replicate_highest_uncertainty",
    "resolve_quality_failure",
    "select_relevant_optional_sensor",
    "select_relevant_control_condition",
    "prefer_protocol_default",
]
GeneralPlannerFallbackReason = Literal[
    "timeout",
    "rate-limit",
    "connection-error",
    "runtime-error",
    "max-turns-exceeded",
    "model-behavior-error",
    "tool-timeout",
    "token-budget-exceeded",
    "invalid-tool-count",
    "unexpected-tool-call",
    "malformed-output",
    "decision-conflict",
    "missing-runtime-trace",
    "user-requested-fallback",
]


class GeneralAdaptiveSufficiencyAssessment(StrictFrozenModel):
    policy_enabled: bool
    minimum_coverage_met: bool
    decision_window_open: bool
    all_evidence_high_quality: bool
    correction_free: bool
    observed_max_within_slot_relative_range: float | None = Field(default=None, ge=0)
    observed_min_relative_contrast: float | None = Field(default=None, ge=0)
    observed_min_contrast_to_uncertainty_ratio: float | None = Field(default=None, ge=0)
    eligible: bool
    blocker_codes: tuple[
        Literal[
            "policy-disabled",
            "minimum-repeat-coverage-missing",
            "adaptive-decision-window-closed",
            "required-evidence-not-high-quality",
            "correction-history-present",
            "within-slot-variation-too-high",
            "relative-contrast-too-small",
            "contrast-not-above-uncertainty",
        ],
        ...,
    ] = Field(default=(), max_length=7)

    @model_validator(mode="after")
    def eligibility_matches_audited_gates(self) -> Self:
        if self.decision_window_open and not self.minimum_coverage_met:
            raise ValueError("adaptive decision windows require minimum coverage")
        if self.eligible != (self.policy_enabled and not self.blocker_codes):
            raise ValueError("adaptive sufficiency eligibility must match all gates")
        if not self.minimum_coverage_met and any(
            value is not None
            for value in (
                self.observed_max_within_slot_relative_range,
                self.observed_min_relative_contrast,
                self.observed_min_contrast_to_uncertainty_ratio,
            )
        ):
            raise ValueError("insufficient coverage cannot expose derived sufficiency metrics")
        if self.minimum_coverage_met and any(
            value is None
            for value in (
                self.observed_max_within_slot_relative_range,
                self.observed_min_relative_contrast,
                self.observed_min_contrast_to_uncertainty_ratio,
            )
        ):
            raise ValueError("covered sufficiency assessments require all derived metrics")
        if len(self.blocker_codes) != len(set(self.blocker_codes)):
            raise ValueError("adaptive sufficiency blockers must be unique")
        return self


class GeneralHypothesisTerminationAudit(StrictFrozenModel):
    """Server-owned receipt for closing or continuing a competing-hypothesis graph.

    A descriptive result may not silently finish while a registered discriminator is
    still unobserved.  The sole evidence-based exemption is a shared observable whose
    valid condition contrast separates every registered hypothesis; the receipt then
    names both the omitted observations and the evidence that justified the omission.
    """

    registered_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    registered_discriminator_ids: tuple[str, ...] = Field(default=(), max_length=32)
    observed_discriminator_ids: tuple[str, ...] = Field(default=(), max_length=32)
    waived_discriminator_ids: tuple[str, ...] = Field(default=(), max_length=32)
    unresolved_discriminator_ids: tuple[str, ...] = Field(default=(), max_length=32)
    gate_satisfied: bool = True
    disposition: Literal[
        "not-applicable",
        "pending-discriminator-evidence",
        "all-discriminators-observed",
        "remaining-discriminators-exempted",
    ] = "not-applicable"
    exemption_reason_code: Literal[
        "observed-shared-discriminator-separates-all-hypotheses"
    ] | None = None
    exemption_basis_observation_ids: tuple[str, ...] = Field(default=(), max_length=8)
    source_evidence_ids: tuple[str, ...] = Field(default=(), max_length=16)
    causal: Literal[False] = False

    @model_validator(mode="after")
    def receipt_is_closed_and_auditable(self) -> Self:
        groups = (
            self.registered_hypothesis_ids,
            self.registered_discriminator_ids,
            self.observed_discriminator_ids,
            self.waived_discriminator_ids,
            self.unresolved_discriminator_ids,
            self.exemption_basis_observation_ids,
            self.source_evidence_ids,
        )
        if any(len(values) != len(set(values)) for values in groups):
            raise ValueError("hypothesis termination receipt IDs must be unique")
        registered = set(self.registered_discriminator_ids)
        observed = set(self.observed_discriminator_ids)
        waived = set(self.waived_discriminator_ids)
        unresolved = set(self.unresolved_discriminator_ids)
        if any(
            left.intersection(right)
            for left, right in (
                (observed, waived),
                (observed, unresolved),
                (waived, unresolved),
            )
        ):
            raise ValueError("hypothesis discriminator receipt groups must be disjoint")
        if observed | waived | unresolved != registered:
            raise ValueError("hypothesis discriminator receipt must partition the registry")
        if not self.registered_hypothesis_ids:
            if (
                registered
                or self.disposition != "not-applicable"
                or not self.gate_satisfied
                or self.exemption_reason_code is not None
                or self.exemption_basis_observation_ids
                or self.source_evidence_ids
            ):
                raise ValueError("empty hypothesis graphs require a not-applicable receipt")
            return self
        if not registered:
            raise ValueError("registered hypotheses require discriminator observations")
        if self.disposition == "pending-discriminator-evidence":
            if (
                self.gate_satisfied
                or not unresolved
                or waived
                or self.exemption_reason_code is not None
                or self.exemption_basis_observation_ids
                or self.source_evidence_ids
            ):
                raise ValueError("pending hypothesis receipts cannot claim an exemption")
        elif self.disposition == "all-discriminators-observed":
            if (
                not self.gate_satisfied
                or observed != registered
                or waived
                or unresolved
                or self.exemption_reason_code is not None
                or self.exemption_basis_observation_ids
                or self.source_evidence_ids
            ):
                raise ValueError("fully observed hypothesis receipts cannot claim an exemption")
        elif self.disposition == "remaining-discriminators-exempted":
            if (
                not self.gate_satisfied
                or not observed
                or not waived
                or unresolved
                or self.exemption_reason_code is None
                or not self.exemption_basis_observation_ids
                or not self.source_evidence_ids
            ):
                raise ValueError("hypothesis exemptions require observed basis and source evidence")
            if not set(self.exemption_basis_observation_ids) <= observed:
                raise ValueError("hypothesis exemption basis must already be observed")
        else:
            raise ValueError("registered hypotheses cannot use a not-applicable receipt")
        return self


class GeneralHypothesisConclusionAudit(StrictFrozenModel):
    """Server-owned, noncausal conclusion drawn from pre-registered predictions."""

    policy_id: Literal["server-hypothesis-conclusion-v1"] = (
        "server-hypothesis-conclusion-v1"
    )
    registered_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    compatible_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    weakened_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    mixed_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    untested_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    conclusion_available: bool = True
    conclusion_code: Literal[
        "not-applicable",
        "pending-discriminator-evidence",
        "one-hypothesis-favored",
        "no-unique-hypothesis-favored",
    ] = "not-applicable"
    favored_hypothesis_id: str | None = Field(default=None, max_length=64)
    source_evidence_ids: tuple[str, ...] = Field(default=(), max_length=128)
    causal: Literal[False] = False

    @model_validator(mode="after")
    def conclusion_is_partitioned_and_auditable(self) -> Self:
        groups = (
            self.registered_hypothesis_ids,
            self.compatible_hypothesis_ids,
            self.weakened_hypothesis_ids,
            self.mixed_hypothesis_ids,
            self.untested_hypothesis_ids,
            self.source_evidence_ids,
        )
        if any(len(values) != len(set(values)) for values in groups):
            raise ValueError("hypothesis conclusion IDs must be unique")
        registered = set(self.registered_hypothesis_ids)
        categories = tuple(
            set(values)
            for values in (
                self.compatible_hypothesis_ids,
                self.weakened_hypothesis_ids,
                self.mixed_hypothesis_ids,
                self.untested_hypothesis_ids,
            )
        )
        if any(
            left.intersection(right)
            for index, left in enumerate(categories)
            for right in categories[index + 1 :]
        ):
            raise ValueError("hypothesis conclusion categories must be disjoint")
        if set().union(*categories) != registered:
            raise ValueError("hypothesis conclusion categories must partition the registry")
        if not registered:
            if (
                self.conclusion_code != "not-applicable"
                or not self.conclusion_available
                or self.favored_hypothesis_id is not None
                or self.source_evidence_ids
            ):
                raise ValueError("empty hypothesis graphs require a not-applicable conclusion")
            return self
        if self.conclusion_code == "pending-discriminator-evidence":
            if self.conclusion_available or self.favored_hypothesis_id is not None:
                raise ValueError("pending hypothesis conclusions cannot claim a verdict")
            return self
        if not self.conclusion_available or not self.source_evidence_ids:
            raise ValueError("final hypothesis conclusions require observed source evidence")
        if self.conclusion_code == "one-hypothesis-favored":
            if (
                self.favored_hypothesis_id is None
                or set(self.compatible_hypothesis_ids) != {self.favored_hypothesis_id}
                or set(self.weakened_hypothesis_ids)
                != registered - {self.favored_hypothesis_id}
                or self.mixed_hypothesis_ids
                or self.untested_hypothesis_ids
            ):
                raise ValueError("a favored verdict requires one compatible and all others weakened")
        elif self.conclusion_code == "no-unique-hypothesis-favored":
            if self.favored_hypothesis_id is not None:
                raise ValueError("non-discriminating conclusions cannot favor a hypothesis")
            if len(self.compatible_hypothesis_ids) == 1 and (
                set(self.weakened_hypothesis_ids)
                == registered - set(self.compatible_hypothesis_ids)
                and not self.mixed_hypothesis_ids
                and not self.untested_hypothesis_ids
            ):
                raise ValueError("a uniquely compatible hypothesis requires a favored verdict")
        else:
            raise ValueError("registered hypotheses cannot use a not-applicable conclusion")
        return self


class GeneralDesignCandidate(StrictFrozenModel):
    candidate_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    action: GeneralTaskAction
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    sensors: tuple[SensorKind, ...] = Field(min_length=1, max_length=8)
    repeat_index: int = Field(ge=1, le=32)
    title: str = Field(min_length=1, max_length=180)
    instruction: str = Field(min_length=1, max_length=1000)
    reason_code: GeneralTaskReason
    input_evidence_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def candidate_sensor_set_is_unique(self) -> Self:
        if len(self.sensors) != len(set(self.sensors)):
            raise ValueError("candidate sensors must be unique")
        if len(self.input_evidence_ids) != len(set(self.input_evidence_ids)):
            raise ValueError("candidate input evidence IDs must be unique")
        return self


class GeneralExperimentTask(StrictFrozenModel):
    task_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    sequence: int = Field(ge=1, le=256)
    action: GeneralTaskAction
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    sensors: tuple[SensorKind, ...] = Field(min_length=1, max_length=8)
    repeat_index: int = Field(ge=1, le=32)
    title: str = Field(min_length=1, max_length=180)
    instruction: str = Field(min_length=1, max_length=1000)
    reason_code: GeneralTaskReason
    input_evidence_ids: tuple[str, ...] = Field(default=(), max_length=64)
    status: Literal["pending", "completed"] = "pending"
    output_evidence_ids: tuple[str, ...] = Field(default=(), max_length=8)
    measurement_valid: bool | None = None
    rejection_reasons: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def task_state_is_consistent(self) -> Self:
        if len(self.sensors) != len(set(self.sensors)):
            raise ValueError("task sensors must be unique")
        if self.status == "pending":
            if self.output_evidence_ids or self.measurement_valid is not None:
                raise ValueError("pending tasks cannot retain measurement outcomes")
            if self.rejection_reasons:
                raise ValueError("pending tasks cannot retain rejection reasons")
        if self.status == "completed":
            if len(self.output_evidence_ids) != len(self.sensors):
                raise ValueError("completed tasks require one output evidence per sensor")
            if self.measurement_valid is None:
                raise ValueError("completed tasks require a measurement validity decision")
            if self.measurement_valid and self.rejection_reasons:
                raise ValueError("valid measurements cannot retain rejection reasons")
            if not self.measurement_valid and not self.rejection_reasons:
                raise ValueError("invalid measurements require rejection reasons")
        if len(self.rejection_reasons) != len(set(self.rejection_reasons)):
            raise ValueError("task rejection reasons must be unique")
        return self


class GeneralMeasurementSubmission(StrictFrozenModel):
    case_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    task_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    expected_revision: int = Field(ge=1)
    evidence: tuple[GeneralEvidenceEnvelope, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def evidence_ids_and_sensors_are_unique(self) -> Self:
        evidence_ids = [item.evidence_id for item in self.evidence]
        sensors = [item.sensor for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("measurement submission evidence IDs must be unique")
        if len(sensors) != len(set(sensors)):
            raise ValueError("measurement submission sensors must be unique")
        return self


class GeneralTerminationVector(StrictFrozenModel):
    required_evidence_count: int = Field(ge=1, le=192)
    valid_evidence_count: int = Field(ge=0, le=256)
    invalid_evidence_count: int = Field(ge=0, le=256)
    condition_coverage_ratio: float = Field(ge=0, le=1)
    sensor_coverage_ratio: float = Field(ge=0, le=1)
    repeat_coverage_ratio: float = Field(ge=0, le=1)
    correction_count: int = Field(ge=0, le=8)
    completion_basis: Literal[
        "none",
        "registered-three-repeats",
        "adaptive-two-repeat-sufficiency",
    ] = "none"
    adaptive_sufficiency: GeneralAdaptiveSufficiencyAssessment
    hypothesis_termination: GeneralHypothesisTerminationAudit = Field(
        default_factory=GeneralHypothesisTerminationAudit
    )
    hypothesis_conclusion: GeneralHypothesisConclusionAudit | None = None
    evidence_complete: bool
    reasoning_required: bool = False
    guidance_ready: bool = False
    conclusion_ready: bool
    forced_stop: bool
    reason_code: Literal[
        "continue",
        "evidence-complete",
        "adaptive-evidence-sufficient",
        "correction-budget-exhausted",
        "measurement-budget-exhausted",
    ]
    blocker_codes: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def termination_state_is_consistent(self) -> Self:
        completion_reasons = {
            "evidence-complete": "registered-three-repeats",
            "adaptive-evidence-sufficient": "adaptive-two-repeat-sufficiency",
        }
        evidence_complete = self.reason_code in completion_reasons
        if self.evidence_complete != evidence_complete:
            raise ValueError("evidence completeness must match the registered evidence window")
        if self.reasoning_required != (self.evidence_complete and not self.guidance_ready):
            raise ValueError("reasoning is required between evidence completion and guidance")
        if self.conclusion_ready and (not self.evidence_complete or not self.guidance_ready):
            raise ValueError("conclusions require complete evidence and validated guidance")
        expected_basis = completion_reasons.get(self.reason_code, "none")
        if self.completion_basis != expected_basis:
            raise ValueError("termination completion basis does not match its reason")
        if (
            self.reason_code == "adaptive-evidence-sufficient"
            and not self.adaptive_sufficiency.eligible
        ):
            raise ValueError("adaptive completion requires a server-certified assessment")
        budget_stop = self.reason_code in {
            "correction-budget-exhausted",
            "measurement-budget-exhausted",
        }
        if self.forced_stop != budget_stop:
            raise ValueError("forced stop must match a budget reason")
        if self.reason_code == "continue" and not self.blocker_codes:
            raise ValueError("continuing experiments require visible blockers")
        if self.guidance_ready and self.blocker_codes:
            raise ValueError("conclusion-ready vectors cannot retain blockers")
        if self.conclusion_ready and not self.hypothesis_termination.gate_satisfied:
            raise ValueError("conclusion-ready vectors must close the hypothesis gate")
        if self.hypothesis_conclusion is not None:
            if set(self.hypothesis_conclusion.registered_hypothesis_ids) != set(
                self.hypothesis_termination.registered_hypothesis_ids
            ):
                raise ValueError("hypothesis conclusion and termination registry must match")
            if self.conclusion_ready and not self.hypothesis_conclusion.conclusion_available:
                raise ValueError("conclusion-ready vectors require a hypothesis conclusion")
        return self


class GeneralConditionMetricSummary(StrictFrozenModel):
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    sensor: SensorKind
    metric_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=80)
    unit: str = Field(min_length=1, max_length=24)
    values: tuple[float, ...] = Field(min_length=2, max_length=32)
    median: float
    median_absolute_deviation: float = Field(ge=0)


class GeneralAuxiliaryObservation(StrictFrozenModel):
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    sensor: SensorKind
    metric_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=80)
    value: float
    unit: str = Field(min_length=1, max_length=24)
    quality: Literal["medium", "high"]
    interpretation: Literal[
        "single_optional_probe_not_a_condition_comparison",
        "paired_optional_probe_descriptive_contrast",
        "single_optional_condition_probe_not_registered_comparison",
    ] = "single_optional_probe_not_a_condition_comparison"


class GeneralMetricContrast(StrictFrozenModel):
    sensor: SensorKind
    metric_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=80)
    unit: str = Field(min_length=1, max_length=24)
    reference_condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    comparison_condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    absolute_delta: float
    relative_delta_ratio: float | None = None
    descriptive_threshold: float = Field(ge=0)
    direction: Literal[
        "increase",
        "decrease",
        "within_observed_repeatability",
    ]
    causal: Literal[False] = False


class GeneralVisualizationPoint(StrictFrozenModel):
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    condition_label: str = Field(min_length=1, max_length=100)
    median: float
    median_absolute_deviation: float = Field(ge=0)
    repeat_count: int = Field(ge=2, le=32)


class GeneralComparisonSeries(StrictFrozenModel):
    sensor: SensorKind
    metric_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=80)
    metric_label: str = Field(min_length=1, max_length=120)
    unit: str = Field(min_length=1, max_length=24)
    points: tuple[GeneralVisualizationPoint, ...] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def series_is_privacy_safe_and_unique(self) -> Self:
        if self.sensor == "bluetooth":
            raise ValueError("Bluetooth cannot produce a numeric comparison series")
        condition_ids = [item.condition_id for item in self.points]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("visualization series conditions must be unique")
        normalized_key = self.metric_key.lower()
        if self.sensor == "location" and any(
            token in normalized_key for token in ("lat", "lon", "coordinate")
        ):
            raise ValueError("absolute coordinates cannot enter general visualizations")
        if self.sensor == "microphone" and any(
            token in normalized_key for token in ("raw_audio", "waveform", "transcript")
        ):
            raise ValueError("raw audio cannot enter general visualizations")
        return self


class GeneralVisualizationArtifact(StrictFrozenModel):
    artifact_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    kind: Literal["condition_comparison_small_multiples"] = "condition_comparison_small_multiples"
    title: str = Field(min_length=1, max_length=180)
    independent_variable: str = Field(min_length=1, max_length=120)
    series: tuple[GeneralComparisonSeries, ...] = Field(min_length=1, max_length=8)
    source_evidence_ids: tuple[str, ...] = Field(min_length=4, max_length=128)
    warnings: tuple[str, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def artifact_graph_is_unique(self) -> Self:
        sensors = [item.sensor for item in self.series]
        if len(sensors) != len(set(sensors)):
            raise ValueError("one visualization artifact may contain one series per sensor")
        if len(self.source_evidence_ids) != len(set(self.source_evidence_ids)):
            raise ValueError("visualization evidence references must be unique")
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("visualization warnings must be unique")
        return self


class GeneralHypothesisObservationAssessment(StrictFrozenModel):
    observation_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    sensor: SensorKind
    metric_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=80)
    metric_unit: str = Field(min_length=1, max_length=24)
    expected_relation: Literal[
        "comparison_higher",
        "comparison_lower",
        "within_relative_deadband",
        "different_unspecified",
    ]
    observed_relation: Literal[
        "comparison-higher",
        "comparison-lower",
        "within-relative-deadband",
    ] | None = None
    match_code: Literal["not_observed", "matches_expected", "conflicts_expected"]
    source_evidence_ids: tuple[str, ...] = Field(default=(), max_length=6)
    causal: Literal[False] = False

    @model_validator(mode="after")
    def observation_state_is_consistent(self) -> Self:
        if (self.observed_relation is None) != (self.match_code == "not_observed"):
            raise ValueError("hypothesis assessment relation and match code are inconsistent")
        if (not self.source_evidence_ids) != (self.match_code == "not_observed"):
            raise ValueError("hypothesis assessment evidence and match code are inconsistent")
        if len(self.source_evidence_ids) != len(set(self.source_evidence_ids)):
            raise ValueError("hypothesis assessment evidence IDs must be unique")
        return self


class GeneralHypothesisAssessment(StrictFrozenModel):
    hypothesis_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    statement_untrusted: str = Field(min_length=8, max_length=500)
    epistemic_status: Literal["untested_hypothesis"] = "untested_hypothesis"
    assessment_code: Literal[
        "untested",
        "observed_prediction_matched",
        "observed_prediction_conflicted",
        "mixed_observations",
    ]
    observations: tuple[GeneralHypothesisObservationAssessment, ...] = Field(
        min_length=1,
        max_length=8,
    )
    causal: Literal[False] = False

    @model_validator(mode="after")
    def assessment_matches_observation_states(self) -> Self:
        states = {item.match_code for item in self.observations}
        expected = (
            "mixed_observations"
            if {"matches_expected", "conflicts_expected"} <= states
            else "observed_prediction_matched"
            if "matches_expected" in states
            else "observed_prediction_conflicted"
            if "conflicts_expected" in states
            else "untested"
        )
        if self.assessment_code != expected:
            raise ValueError("hypothesis assessment code does not match observations")
        return self


class GeneralReasoningExplanationAssessment(StrictFrozenModel):
    explanation_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    label: str = Field(min_length=3, max_length=240)
    role: Literal[
        "target_mechanism",
        "alternative_mechanism",
        "confound",
        "measurement_artifact",
    ]
    verdict: Literal["favored", "plausible", "weakened", "unsupported", "untested"]
    can_explain_primary_effect: bool
    supporting_fact_ids: tuple[str, ...] = Field(default=(), max_length=16)
    conflicting_fact_ids: tuple[str, ...] = Field(default=(), max_length=16)
    reasoning: str = Field(min_length=3, max_length=700)
    missing_test: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def assessment_references_are_unique(self) -> Self:
        if len(self.supporting_fact_ids) != len(set(self.supporting_fact_ids)):
            raise ValueError("reasoning support fact IDs must be unique")
        if len(self.conflicting_fact_ids) != len(set(self.conflicting_fact_ids)):
            raise ValueError("reasoning conflict fact IDs must be unique")
        if set(self.supporting_fact_ids).intersection(self.conflicting_fact_ids):
            raise ValueError("one fact cannot both support and conflict with an explanation")
        if self.verdict == "favored" and self.role != "target_mechanism":
            raise ValueError("only a target mechanism can be the favored answer")
        return self


class GeneralReasoningRuntimeSnapshot(StrictFrozenModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9-]+$", max_length=80)
    status: Literal["completed", "failed", "cancelled"]
    transport: Literal["agent_tool", "validated_json_chat", "deterministic_fallback"] = (
        "agent_tool"
    )
    transport_fallback_reason: str | None = Field(default=None, max_length=80)
    model: str = Field(min_length=1, max_length=120)
    reasoning_mode: Literal["fast", "deep", "provider_default"] | None = None
    reasoning_effort: str | None = Field(default=None, max_length=20)
    model_requests: int = Field(ge=0, le=8)
    tool_calls: int = Field(ge=0, le=4)
    elapsed_ms: int = Field(ge=0, le=86_400_000)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    token_budget: int = Field(ge=1, le=32_000)
    token_budget_exceeded: bool
    error_kind: str | None = Field(default=None, max_length=80)
    error_type: str | None = Field(default=None, max_length=120)


class GeneralReasoningReceipt(StrictFrozenModel):
    policy_id: Literal["general-evidence-reasoning-v1"] = "general-evidence-reasoning-v1"
    case_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    expected_revision: int = Field(ge=1)
    request_sha256: str = Field(pattern=_SHA256)
    decision: Literal["finalize", "continue", "offer_user_choice", "user_stop"]
    answer_headline: str = Field(min_length=8, max_length=300)
    mechanism_explanation: str = Field(min_length=12, max_length=1200)
    claim_scope: Literal[
        "local_intervention_supported",
        "ranked_explanation",
        "descriptive_only",
    ]
    confidence: Literal["low", "medium", "high"]
    confidence_score: float = Field(ge=0, le=1)
    evidence_strength_score: float = Field(ge=0, le=1)
    direct_answer_first: Literal[True] = True
    explanations: tuple[GeneralReasoningExplanationAssessment, ...] = Field(
        min_length=1,
        max_length=8,
    )
    source_fact_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    remaining_uncertainties: tuple[str, ...] = Field(default=(), max_length=8)
    falsification_conditions: tuple[str, ...] = Field(default=(), max_length=8)
    selected_candidate_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    next_measurement_reason: str | None = Field(default=None, max_length=600)
    runtime: GeneralReasoningRuntimeSnapshot | None = None

    @model_validator(mode="after")
    def receipt_is_actionable_and_closed(self) -> Self:
        explanation_ids = [item.explanation_id for item in self.explanations]
        if len(explanation_ids) != len(set(explanation_ids)):
            raise ValueError("reasoning explanations must be unique")
        if len(self.source_fact_ids) != len(set(self.source_fact_ids)):
            raise ValueError("reasoning source facts must be unique")
        referenced = {
            fact_id
            for item in self.explanations
            for fact_id in (*item.supporting_fact_ids, *item.conflicting_fact_ids)
        }
        if not referenced <= set(self.source_fact_ids):
            raise ValueError("explanation facts must remain inside the reasoning receipt")
        favored = [item for item in self.explanations if item.verdict == "favored"]
        if self.decision == "finalize" and len(favored) != 1:
            raise ValueError("final reasoning requires exactly one favored target mechanism")
        if self.decision == "continue":
            if self.selected_candidate_id is None or self.next_measurement_reason is None:
                raise ValueError("continuing reasoning requires one selected measurement")
        elif self.selected_candidate_id is not None:
            raise ValueError("non-continuing reasoning cannot select a measurement candidate")
        if self.confidence == "high" and self.confidence_score < 0.82:
            raise ValueError("high confidence requires a calibrated score of at least 0.82")
        if self.confidence == "medium" and not 0.60 <= self.confidence_score < 0.82:
            raise ValueError("medium confidence must remain inside its calibrated interval")
        if self.confidence == "low" and self.confidence_score >= 0.60:
            raise ValueError("low confidence must remain below the medium interval")
        return self


class GeneralExperimentReport(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    outcome: Literal["completed_descriptive", "completed_inconclusive"]
    answer: str = Field(min_length=1, max_length=1200)
    confidence: Literal["low", "medium", "high"]
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    answer_headline: str | None = Field(default=None, max_length=300)
    mechanism_explanation: str | None = Field(default=None, max_length=1200)
    reasoning: GeneralReasoningReceipt | None = None
    evidence_scope: Literal["physical_recordings", "simulated_rehearsal"] = (
        "physical_recordings"
    )
    completion_basis: Literal[
        "none",
        "registered-three-repeats",
        "adaptive-two-repeat-sufficiency",
    ] = "none"
    termination_reason: str = Field(min_length=1, max_length=300)
    summaries: tuple[GeneralConditionMetricSummary, ...] = Field(default=(), max_length=32)
    contrasts: tuple[GeneralMetricContrast, ...] = Field(default=(), max_length=32)
    auxiliary_observations: tuple[GeneralAuxiliaryObservation, ...] = Field(
        default=(), max_length=4
    )
    hypothesis_assessments: tuple[GeneralHypothesisAssessment, ...] = Field(
        default=(),
        max_length=4,
    )
    hypothesis_termination: GeneralHypothesisTerminationAudit = Field(
        default_factory=GeneralHypothesisTerminationAudit
    )
    # None preserves records created before the v1 structured conclusion policy.
    hypothesis_conclusion: GeneralHypothesisConclusionAudit | None = None
    visualizations: tuple[GeneralVisualizationArtifact, ...] = Field(default=(), max_length=4)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=128)
    excluded_evidence_ids: tuple[str, ...] = Field(default=(), max_length=128)
    claim_boundaries: tuple[str, ...] = Field(min_length=1, max_length=16)
    descriptive_only: bool = True
    causal: Literal[False] = False
    general_exploration_beta: Literal[False] = False
    agent_ready: Literal[False] = False
    market_validated: Literal[False] = False

    @model_validator(mode="after")
    def report_state_is_consistent(self) -> Self:
        if set(self.evidence_ids).intersection(self.excluded_evidence_ids):
            raise ValueError("report evidence cannot be both included and excluded")
        if self.outcome == "completed_descriptive":
            if not self.summaries or not self.contrasts or not self.evidence_ids:
                raise ValueError("descriptive reports require summaries, contrasts and evidence")
            if self.completion_basis == "none":
                raise ValueError("descriptive reports require an audited completion basis")
            if not self.hypothesis_termination.gate_satisfied:
                raise ValueError("descriptive reports must close the hypothesis termination gate")
        elif self.contrasts:
            raise ValueError("inconclusive reports cannot retain descriptive contrasts")
        elif self.completion_basis != "none":
            raise ValueError("inconclusive reports cannot claim a completion basis")
        summary_pairs = {(item.sensor, item.metric_key) for item in self.summaries}
        assessment_ids = [item.hypothesis_id for item in self.hypothesis_assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("report hypothesis assessments must be unique")
        if self.hypothesis_conclusion is not None:
            if set(self.hypothesis_conclusion.registered_hypothesis_ids) != set(
                assessment_ids
            ):
                raise ValueError("report hypothesis conclusion registry must match assessments")
            if self.outcome == "completed_descriptive" and not (
                self.hypothesis_conclusion.conclusion_available
            ):
                raise ValueError("descriptive reports require an available hypothesis conclusion")
        if any(
            not set(observation.source_evidence_ids) <= set(self.evidence_ids)
            for assessment in self.hypothesis_assessments
            for observation in assessment.observations
        ):
            raise ValueError("hypothesis assessment evidence must remain inside report evidence")
        for artifact in self.visualizations:
            if not set(artifact.source_evidence_ids) <= set(self.evidence_ids):
                raise ValueError("visualization evidence must remain inside report evidence")
            if (
                not {(series.sensor, series.metric_key) for series in artifact.series}
                <= summary_pairs
            ):
                raise ValueError("visualization series must derive from report summaries")
        if self.reasoning is not None:
            if self.answer_headline != self.reasoning.answer_headline:
                raise ValueError("report headline must match the reasoning receipt")
            if self.mechanism_explanation != self.reasoning.mechanism_explanation:
                raise ValueError("report explanation must match the reasoning receipt")
            if self.confidence != self.reasoning.confidence:
                raise ValueError("report confidence must match the reasoning receipt")
            if self.confidence_score != self.reasoning.confidence_score:
                raise ValueError("report confidence score must match the reasoning receipt")
        elif any(
            value is not None
            for value in (self.confidence_score, self.answer_headline, self.mechanism_explanation)
        ):
            raise ValueError("structured answer fields require a reasoning receipt")
        return self


class GeneralReasoningCheckpoint(StrictFrozenModel):
    checkpoint_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    triggered_at_task_count: int = Field(ge=8, le=96)
    continue_allowed: bool
    recommended_candidate_id: str | None = Field(
        default=None,
        pattern=_IDENTIFIER,
        max_length=80,
    )
    continuation_candidates: tuple[GeneralDesignCandidate, ...] = Field(
        default=(),
        max_length=8,
    )
    reasoning: GeneralReasoningReceipt
    provisional_report: GeneralExperimentReport
    prompt: str = Field(min_length=8, max_length=500)

    @model_validator(mode="after")
    def checkpoint_is_actionable(self) -> Self:
        candidate_ids = {item.candidate_id for item in self.continuation_candidates}
        if self.reasoning.decision != "offer_user_choice":
            raise ValueError("checkpoint reasoning must offer a user choice")
        if self.continue_allowed:
            if not candidate_ids or self.recommended_candidate_id not in candidate_ids:
                raise ValueError("continuable checkpoints require a recommended frozen candidate")
        elif self.recommended_candidate_id is not None or candidate_ids:
            raise ValueError("hard-stop checkpoints cannot retain continuation candidates")
        return self


class GeneralReasoningCheckpointDecision(StrictFrozenModel):
    expected_revision: int = Field(ge=1)
    action: Literal["continue", "stop"]


class GeneralDesignDecisionTrace(StrictFrozenModel):
    revision: int = Field(ge=1)
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    selected_candidate_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    source: Literal[
        "server_initial",
        "deterministic_fallback",
        "deterministic_policy",
        "bounded_agent",
        "reasoning_agent",
        "user_checkpoint",
    ]
    reason_code: GeneralTaskReason
    input_evidence_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def selected_candidate_is_allowlisted(self) -> Self:
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected candidate must be in the frozen candidate set")
        return self


class GeneralPlannerRuntimeSnapshot(StrictFrozenModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9-]+$", max_length=80)
    status: Literal["completed", "failed", "cancelled"]
    transport: Literal["function_tool", "validated_json_text"] = "function_tool"
    transport_fallback_reason: str | None = Field(default=None, max_length=80)
    model: str = Field(min_length=1, max_length=120)
    attempts: int = Field(ge=1, le=4)
    model_requests: int = Field(ge=0, le=8)
    tool_calls: int = Field(ge=0, le=4)
    elapsed_ms: int = Field(ge=0, le=86_400_000)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    token_budget: int = Field(ge=1, le=32_000)
    token_budget_exceeded: bool
    error_kind: str | None = Field(default=None, max_length=80)
    error_type: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def token_counts_are_consistent(self) -> Self:
        if self.total_tokens is not None and (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("planner token totals must match input and output usage")
        return self


class GeneralHypothesisObservationAudit(StrictFrozenModel):
    hypothesis_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    observation_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    sensor: SensorKind
    expected_relation: Literal[
        "comparison_higher",
        "comparison_lower",
        "within_relative_deadband",
        "different_unspecified",
    ]
    observed_relation: (
        Literal[
            "comparison-higher",
            "comparison-lower",
            "within-relative-deadband",
        ]
        | None
    ) = None
    match_code: Literal[
        "not_observed",
        "matches_expected",
        "conflicts_expected",
    ]

    @model_validator(mode="after")
    def observation_state_is_consistent(self) -> Self:
        if (self.observed_relation is None) != (self.match_code == "not_observed"):
            raise ValueError("hypothesis audit relation and match code are inconsistent")
        return self


class GeneralPlannerDecisionAudit(StrictFrozenModel):
    expected_revision: int = Field(ge=1)
    commit_revision: int = Field(ge=2)
    completed_task_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    prepared_sha256: str = Field(pattern=_SHA256)
    request_sha256: str = Field(pattern=_SHA256)
    candidate_ids: tuple[str, ...] = Field(min_length=2, max_length=8)
    selected_candidate_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    fallback_candidate_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    source: Literal["agent", "deterministic_policy", "deterministic_fallback"]
    outcome: Literal["accepted", "deterministic", "fallback"]
    rationale_code: GeneralPlannerRationaleCode
    fallback_reason: GeneralPlannerFallbackReason | None = None
    runtime: GeneralPlannerRuntimeSnapshot | None = None
    selected_information_goal: (
        Literal[
            "condition_coverage",
            "sensor_coverage",
            "uncertainty_reduction",
            "quality_recovery",
            "hypothesis_discrimination",
            "control_challenge",
        ]
        | None
    ) = None
    selected_effort_points: int | None = Field(default=None, ge=1, le=12)
    selected_candidate_fact_codes: tuple[
        Literal[
            "lowest-effort",
            "highest-observed-uncertainty",
            "no-observed-uncertainty",
            "privacy-sensitive-derived-metric",
            "optional-observation",
            "registered-repetition",
        ],
        ...,
    ] = Field(default=(), max_length=6)
    evidence_fact_codes: tuple[
        Literal[
            "valid-high-quality-evidence",
            "valid-medium-quality-evidence",
            "valid-low-quality-evidence",
            "invalid-evidence",
        ],
        ...,
    ] = Field(default=(), max_length=4)
    contrast_fact_codes: tuple[
        Literal[
            "comparison-higher",
            "comparison-lower",
            "within-relative-deadband",
        ],
        ...,
    ] = Field(default=(), max_length=3)
    selected_discriminates_hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    hypothesis_match_codes: tuple[
        Literal[
            "not_observed",
            "matches_expected",
            "conflicts_expected",
        ],
        ...,
    ] = Field(default=(), max_length=3)
    hypothesis_observation_states: tuple[GeneralHypothesisObservationAudit, ...] = Field(
        default=(),
        max_length=32,
    )

    @model_validator(mode="after")
    def audit_graph_is_closed(self) -> Self:
        if self.commit_revision != self.expected_revision + 1:
            raise ValueError("planner audit must describe exactly one revision commit")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("planner audit candidate IDs must be unique")
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("planner selection must remain inside the frozen candidates")
        if self.fallback_candidate_id not in self.candidate_ids:
            raise ValueError("planner fallback must remain inside the frozen candidates")
        if (self.selected_information_goal is None) != (self.selected_effort_points is None):
            raise ValueError("planner audit information goal and effort must be paired")
        for values, label in (
            (self.selected_candidate_fact_codes, "candidate facts"),
            (self.evidence_fact_codes, "evidence facts"),
            (self.contrast_fact_codes, "contrast facts"),
            (self.selected_discriminates_hypothesis_ids, "selected hypotheses"),
            (self.hypothesis_match_codes, "hypothesis matches"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"planner audit {label} must be unique")
        observation_keys = [
            (item.hypothesis_id, item.observation_id) for item in self.hypothesis_observation_states
        ]
        if len(observation_keys) != len(set(observation_keys)):
            raise ValueError("planner audit hypothesis observations must be unique")
        if self.outcome == "accepted":
            if (
                self.source != "agent"
                or self.fallback_reason is not None
                or self.runtime is None
                or self.runtime.status != "completed"
            ):
                raise ValueError("accepted planner decisions require a completed Agent run")
        elif self.outcome == "deterministic":
            if (
                self.source != "deterministic_policy"
                or self.fallback_reason is not None
                or self.runtime is not None
            ):
                raise ValueError("deterministic planner audits cannot claim an Agent run")
        elif (
            self.source != "deterministic_fallback"
            or self.selected_candidate_id != self.fallback_candidate_id
            or self.fallback_reason is None
        ):
            raise ValueError("planner fallbacks must select the frozen fallback with a reason")
        return self


class GeneralCompilerProvenance(StrictFrozenModel):
    source: Literal["manual", "bounded_agent_compiler"] = "manual"
    receipt_id: str | None = Field(
        default=None,
        pattern=r"^general-compile-[0-9a-f]{20}$",
    )
    draft_sha256: str | None = Field(default=None, pattern=_SHA256)
    compiler_model: str | None = Field(default=None, min_length=1, max_length=120)
    transport: Literal["function_tool", "validated_json_text"] | None = None
    tool_event_names: tuple[str, ...] = Field(default=(), max_length=2)
    created_at: str | None = Field(default=None, min_length=10, max_length=64)

    @model_validator(mode="after")
    def provenance_is_complete_or_explicitly_manual(self) -> Self:
        agent_fields = (
            self.receipt_id,
            self.draft_sha256,
            self.compiler_model,
            self.transport,
            self.created_at,
        )
        if self.source == "bounded_agent_compiler":
            if any(item is None for item in agent_fields):
                raise ValueError("bounded compiler provenance requires a complete receipt")
            if self.transport == "function_tool" and not self.tool_event_names:
                raise ValueError("function-tool provenance requires an accepted tool event")
            if self.transport == "validated_json_text" and self.tool_event_names:
                raise ValueError("validated JSON provenance cannot claim tool events")
        elif any(item is not None for item in agent_fields) or self.tool_event_names:
            raise ValueError("manual compiler provenance cannot claim Agent receipt fields")
        return self


class GeneralExperimentCase(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    revision: int = Field(ge=1)
    status: Literal[
        "collecting",
        "awaiting_user_decision",
        "completed_descriptive",
        "completed_inconclusive",
    ]
    compiler_provenance: GeneralCompilerProvenance = Field(
        default_factory=GeneralCompilerProvenance
    )
    protocol: GeneralExperimentProtocol
    current_task: GeneralExperimentTask | None
    completed_tasks: tuple[GeneralExperimentTask, ...] = Field(default=(), max_length=256)
    evidence: tuple[GeneralEvidenceEnvelope, ...] = Field(default=(), max_length=256)
    decision_trace: tuple[GeneralDesignDecisionTrace, ...] = Field(default=(), max_length=256)
    planner_trace: tuple[GeneralPlannerDecisionAudit, ...] = Field(default=(), max_length=256)
    reasoning_trace: tuple[GeneralReasoningReceipt, ...] = Field(default=(), max_length=64)
    reasoning_checkpoint_count: int = Field(default=0, ge=0, le=16)
    reasoning_checkpoint: GeneralReasoningCheckpoint | None = None
    correction_count: int = Field(ge=0, le=8)
    termination: GeneralTerminationVector
    report: GeneralExperimentReport | None = None
    revision_parent_case_id: str | None = Field(
        default=None,
        pattern=_IDENTIFIER,
        max_length=80,
    )
    revision_feedback: RealityFeedbackRecord | None = None
    superseded_by_case_id: str | None = Field(
        default=None,
        pattern=_IDENTIFIER,
        max_length=80,
    )

    @model_validator(mode="after")
    def case_graph_is_closed(self) -> Self:
        task_ids = [item.task_id for item in self.completed_tasks]
        evidence_ids = [item.evidence_id for item in self.evidence]
        output_evidence_ids = [
            evidence_id for task in self.completed_tasks for evidence_id in task.output_evidence_ids
        ]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("completed task IDs must be unique")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("case evidence IDs must be unique")
        if (
            self.compiler_provenance.draft_sha256 is not None
            and self.compiler_provenance.draft_sha256 != self.protocol.draft_sha256
        ):
            raise ValueError("compiler provenance must attest the frozen protocol draft")
        known_evidence = set(evidence_ids)
        rehearsal = set(self.protocol.selected_sources) == {"protocol_emulator"}
        if any(item.lineage.simulated != rehearsal for item in self.evidence):
            raise ValueError("case evidence scope must match the frozen protocol source")
        for task in self.completed_tasks:
            if task.status != "completed" or not set(task.output_evidence_ids) <= known_evidence:
                raise ValueError("completed task evidence references must be closed")
        if len(output_evidence_ids) != len(set(output_evidence_ids)):
            raise ValueError("one evidence item cannot complete multiple tasks")
        if set(output_evidence_ids) != known_evidence:
            raise ValueError("case evidence must be bound to exactly one completed task")
        valid_count = sum(
            len(task.output_evidence_ids) for task in self.completed_tasks if task.measurement_valid
        )
        invalid_count = len(output_evidence_ids) - valid_count
        if (
            self.termination.required_evidence_count
            != self.protocol.evidence_policy.required_recording_count
            or self.termination.valid_evidence_count != valid_count
            or self.termination.invalid_evidence_count != invalid_count
            or self.termination.correction_count != self.correction_count
        ):
            raise ValueError("termination counts must match the closed case graph")
        if self.correction_count != sum(
            task.measurement_valid is False for task in self.completed_tasks
        ):
            raise ValueError("correction count must match invalid measurement tasks")
        planner_revisions = [item.commit_revision for item in self.planner_trace]
        if len(planner_revisions) != len(set(planner_revisions)):
            raise ValueError("planner audits must be unique per commit revision")
        for audit in self.planner_trace:
            decision = next(
                (item for item in self.decision_trace if item.revision == audit.commit_revision),
                None,
            )
            expected_source = {
                "accepted": "bounded_agent",
                "deterministic": "deterministic_policy",
                "fallback": "deterministic_fallback",
            }[audit.outcome]
            if (
                decision is None
                or decision.source != expected_source
                or decision.selected_candidate_id != audit.selected_candidate_id
            ):
                raise ValueError("planner audits must match the committed design decision")
        for decision in self.decision_trace:
            if decision.source == "bounded_agent" and not any(
                audit.commit_revision == decision.revision and audit.outcome == "accepted"
                for audit in self.planner_trace
            ):
                raise ValueError("bounded Agent decisions require a matching planner audit")
            if decision.source == "reasoning_agent" and not any(
                receipt.expected_revision + 1 == decision.revision
                and receipt.decision == "continue"
                and receipt.selected_candidate_id == decision.selected_candidate_id
                for receipt in self.reasoning_trace
            ):
                raise ValueError("reasoning Agent decisions require a matching reasoning receipt")
        if any(receipt.case_id != self.case_id for receipt in self.reasoning_trace):
            raise ValueError("reasoning receipts cannot cross case boundaries")
        if self.current_task is not None and self.current_task.status != "pending":
            raise ValueError("current task must remain pending")
        if self.status == "collecting":
            if (
                self.current_task is None
                or self.report is not None
                or self.reasoning_checkpoint is not None
            ):
                raise ValueError("collecting cases require a current task and no report")
        elif self.status == "awaiting_user_decision":
            if (
                self.current_task is not None
                or self.report is not None
                or self.reasoning_checkpoint is None
            ):
                raise ValueError("checkpoint cases require one pending user decision")
            if self.reasoning_checkpoint.reasoning not in self.reasoning_trace:
                raise ValueError("checkpoint reasoning must remain in the case trace")
            if self.reasoning_checkpoint_count < 1:
                raise ValueError("checkpoint cases require a positive checkpoint count")
        else:
            if (
                self.current_task is not None
                or self.report is None
                or self.reasoning_checkpoint is not None
            ):
                raise ValueError("terminal cases require a report and no current task")
            if self.status != self.report.outcome:
                raise ValueError("case status must match report outcome")
            if self.report.completion_basis != self.termination.completion_basis:
                raise ValueError("report and termination completion bases must agree")
            expected_scope = "simulated_rehearsal" if rehearsal else "physical_recordings"
            if self.report.evidence_scope != expected_scope:
                raise ValueError("report evidence scope must match the frozen protocol source")
            if (
                not set(self.report.evidence_ids) <= known_evidence
                or not set(self.report.excluded_evidence_ids) <= known_evidence
            ):
                raise ValueError("report evidence references must remain inside the case")
            if any(
                not set(artifact.source_evidence_ids) <= known_evidence
                for artifact in self.report.visualizations
            ):
                raise ValueError("report visualization references must remain inside the case")
        return self


class PreparedGeneralTransition(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    base_case: GeneralExperimentCase
    completed_task: GeneralExperimentTask
    submitted_evidence: tuple[GeneralEvidenceEnvelope, ...] = Field(min_length=1, max_length=8)
    next_candidates: tuple[GeneralDesignCandidate, ...] = Field(default=(), max_length=8)
    fallback_candidate_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    termination: GeneralTerminationVector
    report: GeneralExperimentReport | None = None
    correction_count: int = Field(ge=0, le=8)
    prepared_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def transition_graph_is_closed(self) -> Self:
        if self.base_case.current_task is None:
            raise ValueError("prepared transition requires a current task")
        if self.completed_task.task_id != self.base_case.current_task.task_id:
            raise ValueError("prepared transition completed the wrong task")
        if tuple(item.evidence_id for item in self.submitted_evidence) != (
            self.completed_task.output_evidence_ids
        ):
            raise ValueError("prepared evidence must match completed task outputs")
        candidate_ids = [item.candidate_id for item in self.next_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("prepared candidate IDs must be unique")
        if self.next_candidates:
            if self.fallback_candidate_id not in candidate_ids:
                raise ValueError("fallback must reference a prepared candidate")
            if (
                self.termination.reasoning_required
                or self.termination.guidance_ready
                or self.termination.forced_stop
            ):
                raise ValueError("terminal transitions cannot retain candidates")
        elif self.fallback_candidate_id is not None:
            raise ValueError("empty candidate sets cannot define fallback")
        has_report_state = (
            self.termination.reasoning_required
            or self.termination.guidance_ready
            or self.termination.forced_stop
        )
        if has_report_state != (self.report is not None):
            raise ValueError("terminal transitions and reports must agree")
        if (
            self.report is not None
            and self.report.completion_basis != self.termination.completion_basis
        ):
            raise ValueError("prepared report must match the termination completion basis")
        return self
