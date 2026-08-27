from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pocketlab.sensor_models import SensorKind

_IDENTIFIER = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
_METRIC_KEY = r"^[A-Za-z][A-Za-z0-9_]*$"
_SHA256 = r"^[0-9a-f]{64}$"

GeneralObjective = Literal[
    "compare_conditions",
    "characterize_trend",
    "detect_event",
    "estimate_relationship",
    "check_repeatability",
    "map_relative_pattern",
    "combine_signals",
]
GeneralClaimKind = Literal[
    "descriptive",
    "relative_comparison",
    "association",
    "causal",
    "absolute_calibration",
    "medical_diagnosis",
    "person_identification",
    "surveillance",
    "dangerous_operation",
]
GeneralAcquisitionSource = Literal[
    "phyphox_live",
    "phone_upload",
    "public_replay",
    "protocol_emulator",
]
GeneralSensorRole = Literal["primary", "supporting", "control"]
GeneralAlignment = Literal["sequential", "simultaneous", "either"]
GeneralCompilationStatus = Literal["executable", "plan_only", "rejected"]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class GeneralMetricCapability(StrictFrozenModel):
    metric_key: str = Field(pattern=_METRIC_KEY, max_length=80)
    unit: str = Field(min_length=1, max_length=24)
    label: str = Field(min_length=1, max_length=100)


class GeneralSensorCapabilityContract(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    sensor: SensorKind
    analyzer_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    metrics: tuple[GeneralMetricCapability, ...] = Field(default=(), max_length=16)
    privacy_ack_required: bool = False
    supports_live_capture: bool
    supports_file_upload: bool
    supports_public_replay: bool
    supports_bounded_agent: bool
    limitations: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def capability_is_internally_consistent(self) -> Self:
        metric_keys = [item.metric_key for item in self.metrics]
        if len(metric_keys) != len(set(metric_keys)):
            raise ValueError("sensor capability metric keys must be unique")
        if self.supports_bounded_agent and (self.analyzer_id is None or not self.metrics):
            raise ValueError("bounded Agent capability requires an analyzer and metrics")
        if self.sensor == "bluetooth":
            if self.analyzer_id is not None or self.metrics or self.supports_bounded_agent:
                raise ValueError("Bluetooth remains capability-check only")
        elif self.analyzer_id is None:
            raise ValueError("numeric sensor capabilities require an analyzer")
        return self


class GeneralConditionDraft(StrictFrozenModel):
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    label: str = Field(min_length=1, max_length=100)
    factor_level: str = Field(min_length=1, max_length=120)
    instruction: str = Field(min_length=1, max_length=800)
    activation: Literal["required", "optional_control"] = "required"


class GeneralSensorIntentDraft(StrictFrozenModel):
    sensor: SensorKind
    role: GeneralSensorRole
    activation: Literal["required", "optional_probe"] = "required"
    metric_key: str | None = Field(default=None, pattern=_METRIC_KEY, max_length=80)
    metric_unit: str | None = Field(default=None, min_length=1, max_length=24)
    measurement_purpose: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def metric_fields_are_paired(self) -> Self:
        if (self.metric_key is None) != (self.metric_unit is None):
            raise ValueError("metric_key and metric_unit must be provided together")
        if self.sensor == "bluetooth" and self.metric_key is not None:
            raise ValueError("Bluetooth capability checks cannot declare a numeric metric")
        if self.sensor != "bluetooth" and self.metric_key is None:
            raise ValueError("numeric sensors require a metric")
        if self.role == "primary" and self.activation != "required":
            raise ValueError("the primary sensor must remain required")
        if self.sensor == "bluetooth" and self.activation != "required":
            raise ValueError("Bluetooth cannot be an optional numeric probe")
        return self


class GeneralHypothesisPredictionDraft(StrictFrozenModel):
    prediction_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    sensor: SensorKind
    metric_key: str = Field(pattern=_METRIC_KEY, max_length=80)
    metric_unit: str = Field(min_length=1, max_length=24)
    reference_condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    comparison_condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    expected_relation: Literal[
        "comparison_higher",
        "comparison_lower",
        "within_relative_deadband",
        "different_unspecified",
    ]
    measurement_role: Literal["primary_observation", "discriminator"]

    @model_validator(mode="after")
    def conditions_are_distinct(self) -> Self:
        if self.reference_condition_id == self.comparison_condition_id:
            raise ValueError("hypothesis predictions require two distinct conditions")
        if self.sensor == "bluetooth":
            raise ValueError("Bluetooth cannot appear in a numeric hypothesis prediction")
        return self


class GeneralHypothesisDraft(StrictFrozenModel):
    hypothesis_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    statement_untrusted: str = Field(min_length=8, max_length=500)
    predictions: tuple[GeneralHypothesisPredictionDraft, ...] = Field(
        min_length=1,
        max_length=8,
    )

    @field_validator("predictions", mode="before")
    @classmethod
    def normalize_predictions(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def prediction_ids_are_unique(self) -> Self:
        ids = [item.prediction_id for item in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis prediction IDs must be unique")
        if not any(item.measurement_role == "discriminator" for item in self.predictions):
            raise ValueError("each hypothesis requires a discriminating observation")
        return self


class GeneralExplorationDraft(StrictFrozenModel):
    """Strict output contract for a future bounded question compiler.

    User text remains untrusted data. The model may populate this draft, but the
    server compiler still owns analyzer IDs, action allowlists, evidence policy,
    readiness and termination.
    """

    schema_version: Literal["1.0"] = "1.0"
    title: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=5, max_length=1200)
    objective: GeneralObjective
    requested_claim: GeneralClaimKind
    independent_variable: str = Field(min_length=1, max_length=120)
    conditions: tuple[GeneralConditionDraft, ...] = Field(min_length=2, max_length=4)
    sensor_intents: tuple[GeneralSensorIntentDraft, ...] = Field(min_length=1, max_length=9)
    alignment: GeneralAlignment = "sequential"
    controls: tuple[str, ...] = Field(min_length=2, max_length=16)
    expected_pattern: str = Field(min_length=1, max_length=500)
    hypotheses: tuple[GeneralHypothesisDraft, ...] = Field(default=(), max_length=4)
    safety_notes: tuple[str, ...] = Field(min_length=1, max_length=12)
    privacy_notes: tuple[str, ...] = Field(default=(), max_length=12)
    claim_boundaries: tuple[str, ...] = Field(min_length=2, max_length=12)

    @field_validator("hypotheses", mode="before")
    @classmethod
    def normalize_hypotheses(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def draft_graph_is_closed(self) -> Self:
        condition_ids = [item.condition_id for item in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition IDs must be unique")
        required_conditions = [item for item in self.conditions if item.activation == "required"]
        if len(required_conditions) < 2 or self.conditions[0].activation != "required":
            raise ValueError("at least two required conditions must lead the draft")
        sensors = [item.sensor for item in self.sensor_intents]
        if len(sensors) != len(set(sensors)):
            raise ValueError("each sensor may appear only once")
        if sum(item.role == "primary" for item in self.sensor_intents) != 1:
            raise ValueError("exactly one primary sensor is required")
        if self.hypotheses:
            if self.objective != "compare_conditions":
                raise ValueError("hypothesis graph v1 is limited to condition comparisons")
            if len(self.hypotheses) < 2:
                raise ValueError("hypothesis graph requires at least two competing hypotheses")
            hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
            if len(hypothesis_ids) != len(set(hypothesis_ids)):
                raise ValueError("hypothesis IDs must be unique")
            condition_set = set(condition_ids)
            intent_by_sensor = {item.sensor: item for item in self.sensor_intents}
            prediction_ids: list[str] = []
            signatures: list[tuple[tuple[str, ...], ...]] = []
            predicted_optional_sensors: set[SensorKind] = set()
            for hypothesis in self.hypotheses:
                signature: list[tuple[str, ...]] = []
                for prediction in hypothesis.predictions:
                    prediction_ids.append(prediction.prediction_id)
                    intent = intent_by_sensor.get(prediction.sensor)
                    if (
                        intent is None
                        or intent.metric_key != prediction.metric_key
                        or intent.metric_unit != prediction.metric_unit
                    ):
                        raise ValueError(
                            "hypothesis predictions must bind an exact declared sensor metric"
                        )
                    if {
                        prediction.reference_condition_id,
                        prediction.comparison_condition_id,
                    } - condition_set:
                        raise ValueError("hypothesis predictions reference an unknown condition")
                    if prediction.measurement_role == "discriminator":
                        if intent.activation != "optional_probe":
                            raise ValueError("hypothesis discriminators must use optional probes")
                        predicted_optional_sensors.add(prediction.sensor)
                    signature.append(
                        (
                            prediction.sensor,
                            prediction.metric_key,
                            prediction.metric_unit,
                            prediction.reference_condition_id,
                            prediction.comparison_condition_id,
                            prediction.expected_relation,
                            prediction.measurement_role,
                        )
                    )
                signatures.append(tuple(sorted(signature)))
            if len(prediction_ids) != len(set(prediction_ids)):
                raise ValueError("prediction IDs must be globally unique")
            if len(signatures) != len(set(signatures)):
                raise ValueError("competing hypotheses require distinct observable signatures")
            optional_sensors = {
                item.sensor for item in self.sensor_intents if item.activation == "optional_probe"
            }
            if optional_sensors - predicted_optional_sensors:
                raise ValueError(
                    "every optional probe must discriminate at least one registered hypothesis"
                )
        for values, label in (
            (self.controls, "controls"),
            (self.safety_notes, "safety_notes"),
            (self.privacy_notes, "privacy_notes"),
            (self.claim_boundaries, "claim_boundaries"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class VerifiedPublicReplayMatch(StrictFrozenModel):
    """Server-owned attestation that a public pack matches this exact draft."""

    match_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    draft_sha256: str = Field(pattern=_SHA256)
    objective: GeneralObjective
    sensors: tuple[SensorKind, ...] = Field(min_length=1, max_length=8)
    dataset_ids: tuple[str, ...] = Field(min_length=1, max_length=12)
    claim_scope: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def match_references_are_unique(self) -> Self:
        if len(self.sensors) != len(set(self.sensors)):
            raise ValueError("public replay match sensors must be unique")
        if len(self.dataset_ids) != len(set(self.dataset_ids)):
            raise ValueError("public replay dataset IDs must be unique")
        return self


class GeneralCompileContext(StrictFrozenModel):
    """Server-resolved execution context; it is not accepted directly from a model."""

    selected_sources: tuple[GeneralAcquisitionSource, ...] = Field(default=(), max_length=4)
    detected_sensors: tuple[SensorKind, ...] = Field(default=(), max_length=9)
    privacy_acknowledged_sensors: tuple[SensorKind, ...] = Field(default=(), max_length=2)
    public_replay_match: VerifiedPublicReplayMatch | None = None
    supports_simultaneous_capture: bool = False
    external_reference_available: bool = False
    allow_deferred_live_detection: bool = False
    enable_adaptive_sufficiency: bool = False
    enable_server_owned_optional_activation: bool = False

    @model_validator(mode="after")
    def context_sets_are_unique(self) -> Self:
        for values, label in (
            (self.selected_sources, "selected_sources"),
            (self.detected_sensors, "detected_sensors"),
            (self.privacy_acknowledged_sensors, "privacy_acknowledged_sensors"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class GeneralSensorRequirement(StrictFrozenModel):
    sensor: SensorKind
    role: GeneralSensorRole
    activation: Literal["required", "optional_probe"] = "required"
    analyzer_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    metric_key: str | None = Field(default=None, pattern=_METRIC_KEY, max_length=80)
    metric_unit: str | None = Field(default=None, min_length=1, max_length=24)
    measurement_purpose: str = Field(min_length=1, max_length=400)
    privacy_ack_required: bool
    bounded_agent_supported: bool


class GeneralExpectedObservation(StrictFrozenModel):
    observation_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    sensor: SensorKind
    metric_key: str = Field(pattern=_METRIC_KEY, max_length=80)
    metric_unit: str = Field(min_length=1, max_length=24)
    reference_condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    comparison_condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    expected_relation: Literal[
        "comparison_higher",
        "comparison_lower",
        "within_relative_deadband",
        "different_unspecified",
    ]
    measurement_role: Literal["primary_observation", "discriminator"]
    validation_source: Literal["server_graph_validation_v1"] = "server_graph_validation_v1"

    @model_validator(mode="after")
    def observation_conditions_are_distinct(self) -> Self:
        if self.reference_condition_id == self.comparison_condition_id:
            raise ValueError("expected observations require two distinct conditions")
        if self.sensor == "bluetooth":
            raise ValueError("Bluetooth cannot appear in an expected observation")
        return self


class GeneralHypothesisSpec(StrictFrozenModel):
    hypothesis_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    statement_untrusted: str = Field(min_length=8, max_length=500)
    observations: tuple[GeneralExpectedObservation, ...] = Field(min_length=1, max_length=8)
    epistemic_status: Literal["untested_hypothesis"] = "untested_hypothesis"

    @field_validator("observations", mode="before")
    @classmethod
    def normalize_observations(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def observation_ids_are_unique(self) -> Self:
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("expected observation IDs must be unique")
        if not any(item.measurement_role == "discriminator" for item in self.observations):
            raise ValueError("each hypothesis requires a discriminating observation")
        return self


class GeneralOptionalActivationRule(StrictFrozenModel):
    """Server-owned numeric rule linking one optional probe to one control.

    The question compiler and Planner cannot author or change these values.  A rule is
    evaluated only against valid evidence whose sensor, metric key, and unit all match.
    """

    rule_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    probe_sensor: SensorKind
    metric_key: str = Field(pattern=_METRIC_KEY, max_length=80)
    metric_unit: str = Field(min_length=1, max_length=24)
    comparator: Literal["gt"] = "gt"
    threshold: float = Field(allow_inf_nan=False)
    target_condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    policy_source: Literal["server_registry_v1"] = "server_registry_v1"


class GeneralAdaptiveSufficiencyPolicy(StrictFrozenModel):
    """Server-owned thresholds for omitting the registered third repeat.

    This policy is intentionally conservative and is never authored by the
    question compiler or Planner.  It only certifies a clear descriptive
    contrast; it does not establish causality or absolute calibration.
    """

    enabled: bool = False
    minimum_repeats_per_slot: Literal[2] = 2
    require_all_high_quality: Literal[True] = True
    require_no_corrections: Literal[True] = True
    maximum_within_slot_relative_range: float = Field(default=0.05, gt=0, le=0.25)
    minimum_relative_contrast: float = Field(default=0.10, gt=0, le=1)
    minimum_contrast_to_uncertainty_ratio: float = Field(default=4.0, ge=2, le=20)
    uncertainty_floor_relative: float = Field(default=0.01, gt=0, le=0.10)


class GeneralEvidencePolicy(StrictFrozenModel):
    minimum_repeats_per_condition: int = Field(default=2, ge=2, le=4)
    # Keep the original field name because it is persisted in protocol/history
    # records.  Its meaning is now the preregistered target, not a rigid stop.
    required_repeats_per_condition: int = Field(default=3, ge=2, le=6)
    minimum_confidence: Literal["medium"] = "medium"
    required_recording_count: int = Field(ge=6, le=128)
    max_corrections: int = Field(ge=2, le=8)
    max_optional_probe_count: int = Field(default=0, ge=0, le=2)
    optional_probe_evidence_mode: Literal[
        "single_observation",
        "paired_condition_contrast",
    ] = "single_observation"
    max_optional_condition_count: int = Field(default=0, ge=0, le=2)
    adaptive_sufficiency: GeneralAdaptiveSufficiencyPolicy = Field(
        default_factory=GeneralAdaptiveSufficiencyPolicy
    )
    max_measurements: int = Field(ge=8, le=192)
    user_checkpoint_task_count: int = Field(default=20, ge=8, le=64)
    hard_task_count: int = Field(default=32, ge=12, le=96)
    physical_sources: tuple[Literal["phyphox_live", "phone_upload", "public_replay"], ...] = (
        "phyphox_live",
        "phone_upload",
        "public_replay",
    )
    emulator_is_physical_evidence: Literal[False] = False
    public_replay_counts_as_user_phone_gate_c: Literal[False] = False

    @model_validator(mode="after")
    def budget_contains_required_evidence(self) -> Self:
        if self.max_measurements < self.required_recording_count + self.max_corrections:
            raise ValueError("measurement budget must contain evidence and corrections")
        if self.required_repeats_per_condition < self.minimum_repeats_per_condition:
            raise ValueError("registered repeats cannot be below the minimum evidence window")
        if self.hard_task_count <= self.user_checkpoint_task_count:
            raise ValueError("hard task limit must remain above the user checkpoint")
        return self


class GeneralExperimentProtocol(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    protocol_version: Literal["1.0.0"] = "1.0.0"
    draft_sha256: str = Field(pattern=_SHA256)
    title: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=5, max_length=1200)
    objective: GeneralObjective
    requested_claim: GeneralClaimKind
    independent_variable: str = Field(min_length=1, max_length=120)
    conditions: tuple[GeneralConditionDraft, ...] = Field(min_length=2, max_length=4)
    sensors: tuple[GeneralSensorRequirement, ...] = Field(min_length=1, max_length=9)
    alignment: GeneralAlignment
    controls: tuple[str, ...] = Field(min_length=2, max_length=16)
    expected_pattern: str = Field(min_length=1, max_length=500)
    hypotheses: tuple[GeneralHypothesisSpec, ...] = Field(default=(), max_length=4)
    safety_notes: tuple[str, ...] = Field(min_length=1, max_length=12)
    privacy_notes: tuple[str, ...] = Field(default=(), max_length=12)
    claim_boundaries: tuple[str, ...] = Field(min_length=2, max_length=12)
    selected_sources: tuple[GeneralAcquisitionSource, ...] = Field(default=(), max_length=4)
    public_replay_match_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    optional_activation_rules: tuple[GeneralOptionalActivationRule, ...] = Field(
        default=(),
        max_length=2,
    )
    evidence_policy: GeneralEvidencePolicy
    visualization_kinds: tuple[
        Literal["time_series", "comparison", "scatter", "timeline", "relative_map"],
        ...,
    ] = Field(min_length=1, max_length=4)
    candidate_actions: tuple[
        Literal[
            "collect_condition",
            "collect_supporting_sensor",
            "replicate_condition",
            "correct_condition",
            "probe_optional_sensor",
            "probe_optional_condition",
            "stop_inconclusive",
            "build_report",
        ],
        ...,
    ] = (
        "collect_condition",
        "collect_supporting_sensor",
        "replicate_condition",
        "correct_condition",
        "probe_optional_sensor",
        "probe_optional_condition",
        "stop_inconclusive",
        "build_report",
    )
    planner_permissions: Literal["select_candidate_id_only"] = "select_candidate_id_only"
    server_owned_decisions: tuple[
        Literal[
            "evidence_validity",
            "tool_parameters",
            "quality_gate",
            "optional_activation",
            "termination",
            "report_facts",
        ],
        ...,
    ] = (
        "evidence_validity",
        "tool_parameters",
        "quality_gate",
        "optional_activation",
        "termination",
        "report_facts",
    )
    agent_ready: Literal[False] = False
    market_validated: Literal[False] = False

    @field_validator("hypotheses", mode="before")
    @classmethod
    def normalize_hypotheses(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def protocol_graph_is_closed(self) -> Self:
        required_conditions = [item for item in self.conditions if item.activation == "required"]
        if len(required_conditions) < 2 or self.conditions[0].activation != "required":
            raise ValueError("protocol requires at least two leading required conditions")
        sensors = [item.sensor for item in self.sensors]
        if len(sensors) != len(set(sensors)):
            raise ValueError("protocol sensors must be unique")
        if sum(item.role == "primary" for item in self.sensors) != 1:
            raise ValueError("protocol requires exactly one primary sensor")
        if self.hypotheses:
            if self.objective != "compare_conditions" or len(self.hypotheses) < 2:
                raise ValueError(
                    "protocol hypothesis graph v1 requires a condition comparison and two hypotheses"
                )
            hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
            if len(hypothesis_ids) != len(set(hypothesis_ids)):
                raise ValueError("protocol hypothesis IDs must be unique")
            requirements = {item.sensor: item for item in self.sensors}
            condition_ids = {item.condition_id for item in self.conditions}
            observation_ids: list[str] = []
            signatures: list[tuple[tuple[str, ...], ...]] = []
            predicted_optional_sensors: set[SensorKind] = set()
            for hypothesis in self.hypotheses:
                signature: list[tuple[str, ...]] = []
                for observation in hypothesis.observations:
                    observation_ids.append(observation.observation_id)
                    requirement = requirements.get(observation.sensor)
                    if (
                        requirement is None
                        or requirement.metric_key != observation.metric_key
                        or requirement.metric_unit != observation.metric_unit
                    ):
                        raise ValueError(
                            "expected observations must bind an exact protocol sensor metric"
                        )
                    if {
                        observation.reference_condition_id,
                        observation.comparison_condition_id,
                    } - condition_ids:
                        raise ValueError("expected observations reference an unknown condition")
                    if observation.measurement_role == "discriminator":
                        if requirement.activation != "optional_probe":
                            raise ValueError(
                                "protocol hypothesis discriminators must use optional probes"
                            )
                        predicted_optional_sensors.add(observation.sensor)
                    signature.append(
                        (
                            observation.sensor,
                            observation.metric_key,
                            observation.metric_unit,
                            observation.reference_condition_id,
                            observation.comparison_condition_id,
                            observation.expected_relation,
                            observation.measurement_role,
                        )
                    )
                signatures.append(tuple(sorted(signature)))
            if len(observation_ids) != len(set(observation_ids)):
                raise ValueError("expected observation IDs must be globally unique")
            if len(signatures) != len(set(signatures)):
                raise ValueError("protocol hypotheses require distinct observable signatures")
            optional_sensors = {
                item.sensor for item in self.sensors if item.activation == "optional_probe"
            }
            if optional_sensors - predicted_optional_sensors:
                raise ValueError(
                    "every optional protocol probe must discriminate a registered hypothesis"
                )
        elif self.evidence_policy.optional_probe_evidence_mode != "single_observation":
            raise ValueError("paired optional contrasts require a hypothesis graph")
        expected_actions = {
            "collect_condition",
            "collect_supporting_sensor",
            "replicate_condition",
            "correct_condition",
            "probe_optional_sensor",
            "probe_optional_condition",
            "stop_inconclusive",
            "build_report",
        }
        if set(self.candidate_actions) != expected_actions:
            raise ValueError("general protocol action allowlist changed")
        if any(
            item.activation == "optional_probe" and item.role == "primary" for item in self.sensors
        ):
            raise ValueError("primary sensors cannot be optional probes")
        optional_count = sum(item.activation == "optional_probe" for item in self.sensors)
        if self.evidence_policy.max_optional_probe_count > optional_count:
            raise ValueError("optional probe budget exceeds available optional sensors")
        optional_condition_count = sum(
            item.activation == "optional_control" for item in self.conditions
        )
        if self.evidence_policy.max_optional_condition_count > optional_condition_count:
            raise ValueError("optional condition budget exceeds available optional controls")
        optional_sensors = {
            item.sensor: item for item in self.sensors if item.activation == "optional_probe"
        }
        optional_conditions = {
            item.condition_id for item in self.conditions if item.activation == "optional_control"
        }
        rule_ids = [item.rule_id for item in self.optional_activation_rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("optional activation rule IDs must be unique")
        targets = [item.target_condition_id for item in self.optional_activation_rules]
        if len(targets) != len(set(targets)):
            raise ValueError("optional activation rules must target unique conditions")
        for rule in self.optional_activation_rules:
            requirement = optional_sensors.get(rule.probe_sensor)
            if (
                requirement is None
                or requirement.metric_key != rule.metric_key
                or requirement.metric_unit != rule.metric_unit
                or rule.target_condition_id not in optional_conditions
            ):
                raise ValueError("optional activation rule must bind an exact probe and control")
        expected_server_decisions = {
            "evidence_validity",
            "tool_parameters",
            "quality_gate",
            "optional_activation",
            "termination",
            "report_facts",
        }
        if set(self.server_owned_decisions) != expected_server_decisions:
            raise ValueError("server-owned decision boundary changed")
        if "public_replay" in self.selected_sources and self.public_replay_match_id is None:
            raise ValueError("public replay protocols require a verified semantic match")
        if "public_replay" not in self.selected_sources and self.public_replay_match_id is not None:
            raise ValueError("public replay match is only valid for public replay protocols")
        return self


class GeneralExplorationCompilation(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    status: GeneralCompilationStatus
    protocol: GeneralExperimentProtocol | None
    blocker_codes: tuple[str, ...] = Field(default=(), max_length=16)
    user_messages: tuple[str, ...] = Field(default=(), max_length=16)
    can_run_with_current_context: bool
    requires_real_evidence: Literal[True] = True
    general_exploration_beta: Literal[False] = False
    agent_ready: Literal[False] = False
    market_validated: Literal[False] = False

    @model_validator(mode="after")
    def compilation_state_is_consistent(self) -> Self:
        if len(self.blocker_codes) != len(set(self.blocker_codes)):
            raise ValueError("blocker codes must be unique")
        if self.status == "executable":
            if self.protocol is None or self.blocker_codes or not self.can_run_with_current_context:
                raise ValueError("executable compilation cannot retain blockers")
        elif self.status == "plan_only":
            if self.protocol is None or not self.blocker_codes or self.can_run_with_current_context:
                raise ValueError("plan-only compilation requires a protocol and blockers")
        elif (
            self.protocol is not None or not self.blocker_codes or self.can_run_with_current_context
        ):
            raise ValueError("rejected compilation cannot retain an executable protocol")
        return self
