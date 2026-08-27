from __future__ import annotations

import math
import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

from pocketlab.sensor_models import CapabilityMaturity, SensorAnalysis, SensorKind

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
    ),
]
SemanticVersion = Annotated[
    str,
    StringConstraints(
        min_length=5,
        max_length=32,
        pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?$",
    ),
]
OpaqueRecordId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=240)]
LongText = Annotated[str, StringConstraints(min_length=1, max_length=1200)]
Unit = Annotated[str, StringConstraints(max_length=24)]
ParameterScalar = StrictFloat | StrictInt | StrictBool | StrictStr

InvestigationMode = Literal["diagnose", "explore"]
InvestigationPlanningPolicy = Literal["deterministic", "bounded_agent"]
PlannerTransport = Literal["not_attempted", "function_tool", "validated_json_text"]
ExperimentParameterType = Literal["number", "bool", "text"]
ExperimentTaskRole = Literal[
    "background",
    "condition",
    "replication",
    "correction",
    "exploration",
]
ExperimentTaskStatus = Literal["pending", "in_progress", "completed", "rejected"]
InvestigationStatus = Literal[
    "planning",
    "collecting",
    "completed_with_conclusion",
    "completed_inconclusive",
    "cancelled",
]
PlannerRationaleCode = Literal[
    "maximize_log_span",
    "preserve_signal_to_background",
    "reduce_saturation_risk",
    "respect_user_constraint",
    "prefer_protocol_default",
]

_ACTIVE_CONTENT = re.compile(r"(?:<|>|javascript:|https?://)", re.IGNORECASE)
_ABSOLUTE_LOCATION_FIELDS = {"lat", "lon", "latitude", "longitude"}


class StrictBase(BaseModel):
    """Shared boundary for persisted and API-facing Investigation v2 models."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_finite_number(value: object, *, field_name: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _ensure_unique(values: list[str], *, field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _ensure_safe_display_text(value: str, *, field_name: str) -> None:
    if _ACTIVE_CONTENT.search(value):
        raise ValueError(f"{field_name} cannot contain HTML, JavaScript or URLs")


class ExperimentParameterDefinition(StrictBase):
    key: Identifier
    value_type: ExperimentParameterType
    unit: Unit = ""
    minimum: float | None = None
    maximum: float | None = None
    recommended: ParameterScalar | None = None
    description: ShortText

    @model_validator(mode="after")
    def validate_parameter_contract(self) -> Self:
        if self.value_type == "number":
            if not self.unit:
                raise ValueError("number parameters require an explicit unit; use '1' if dimensionless")
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            ):
                raise ValueError("parameter minimum cannot exceed maximum")
            if self.recommended is not None:
                recommended = _require_finite_number(
                    self.recommended,
                    field_name=f"{self.key}.recommended",
                )
                if self.minimum is not None and recommended < self.minimum:
                    raise ValueError("recommended value is below minimum")
                if self.maximum is not None and recommended > self.maximum:
                    raise ValueError("recommended value is above maximum")
        else:
            if self.unit:
                raise ValueError("bool and text parameters must use an empty unit")
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("bool and text parameters cannot define numeric ranges")
            if self.recommended is not None:
                if self.value_type == "bool" and not isinstance(self.recommended, bool):
                    raise ValueError("bool parameter recommended value must be bool")
                if self.value_type == "text":
                    if not isinstance(self.recommended, str):
                        raise ValueError("text parameter recommended value must be text")
                    if not self.recommended or len(self.recommended) > 240:
                        raise ValueError("text parameter recommended value must contain 1-240 characters")
        return self

    def validate_target(self, target: ExperimentParameterValue) -> None:
        if target.key != self.key:
            raise ValueError(f"parameter target {target.key!r} does not match {self.key!r}")
        if target.unit != self.unit:
            raise ValueError(
                f"parameter {self.key!r} must use unit {self.unit!r}; received {target.unit!r}"
            )
        value = target.value
        if self.value_type == "number":
            numeric = _require_finite_number(value, field_name=self.key)
            if self.minimum is not None and numeric < self.minimum:
                raise ValueError(f"parameter {self.key!r} is below minimum")
            if self.maximum is not None and numeric > self.maximum:
                raise ValueError(f"parameter {self.key!r} is above maximum")
        elif self.value_type == "bool" and not isinstance(value, bool):
            raise ValueError(f"parameter {self.key!r} must be bool")
        elif self.value_type == "text":
            if not isinstance(value, str) or not value or len(value) > 240:
                raise ValueError(f"parameter {self.key!r} must be non-empty text")


class ExperimentParameterValue(StrictBase):
    key: Identifier
    value: ParameterScalar
    unit: Unit = ""

    @model_validator(mode="after")
    def finite_number(self) -> Self:
        if _is_number(self.value):
            _require_finite_number(self.value, field_name=self.key)
        elif isinstance(self.value, str) and (not self.value or len(self.value) > 240):
            raise ValueError("text parameter values must contain 1-240 characters")
        return self


class ExperimentParameterConstraint(StrictBase):
    """A user-confirmed execution bound enforced by the server, never parsed from notes."""

    key: Identifier
    unit: Unit
    minimum: float | None = None
    maximum: float | None = None
    source: Literal["user_confirmed"] = "user_confirmed"

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("an execution constraint requires a minimum or maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("execution constraint minimum cannot exceed maximum")
        return self

    def validate_definition(self, definition: ExperimentParameterDefinition) -> None:
        if self.key != definition.key or self.unit != definition.unit:
            raise ValueError("execution constraint must match its protocol parameter and unit")
        if definition.value_type != "number":
            raise ValueError("execution constraints currently support numeric parameters only")
        if (
            self.minimum is not None
            and definition.minimum is not None
            and self.minimum < definition.minimum
        ):
            raise ValueError("execution constraint minimum is below the protocol minimum")
        if (
            self.maximum is not None
            and definition.maximum is not None
            and self.maximum > definition.maximum
        ):
            raise ValueError("execution constraint maximum is above the protocol maximum")

    def allows(self, value: float) -> bool:
        return not (
            (self.minimum is not None and value < self.minimum)
            or (self.maximum is not None and value > self.maximum)
        )


class ExperimentToolDefinition(StrictBase):
    tool_id: Identifier
    version: SemanticVersion
    allowed_sensors: list[SensorKind] = Field(min_length=1, max_length=9)
    input_metric_keys: list[Identifier] = Field(min_length=1, max_length=24)
    output_metric_keys: list[Identifier] = Field(default_factory=list, max_length=24)
    deterministic: StrictBool = True
    read_only: StrictBool = True
    maturity: CapabilityMaturity

    @model_validator(mode="after")
    def validate_tool_definition(self) -> Self:
        _ensure_unique(self.allowed_sensors, field_name="allowed_sensors")
        _ensure_unique(self.input_metric_keys, field_name="input_metric_keys")
        _ensure_unique(self.output_metric_keys, field_name="output_metric_keys")
        if not self.deterministic or not self.read_only:
            raise ValueError("Investigation analysis tools must be deterministic and read-only")
        if self.maturity in {"detectable", "capture_ready"}:
            raise ValueError("executable analysis tools must be at least analysis_ready")
        return self


class ExperimentProtocol(StrictBase):
    protocol_id: Identifier
    protocol_version: SemanticVersion
    mode: InvestigationMode
    title: ShortText
    primary_sensor: SensorKind
    required_analyzer_id: Identifier
    measurement_metric_key: Identifier
    measurement_metric_unit: Unit
    allowed_tools: list[ExperimentToolDefinition] = Field(min_length=1, max_length=12)
    max_measurements: int = Field(ge=1, le=16)
    max_corrections: int = Field(ge=0, le=6)
    parameters: list[ExperimentParameterDefinition] = Field(default_factory=list, max_length=16)
    controls: list[ShortText] = Field(default_factory=list, max_length=16)
    safety_notes: list[ShortText] = Field(default_factory=list, max_length=12)
    claim_boundaries: list[ShortText] = Field(min_length=1, max_length=12)
    market_validated: Literal[False] = False

    @model_validator(mode="after")
    def validate_protocol_references(self) -> Self:
        parameter_keys = [item.key for item in self.parameters]
        tool_ids = [item.tool_id for item in self.allowed_tools]
        _ensure_unique(parameter_keys, field_name="parameters")
        _ensure_unique(tool_ids, field_name="allowed_tools")
        _ensure_unique(self.controls, field_name="controls")
        if self.max_corrections >= self.max_measurements:
            raise ValueError("max_corrections must be lower than max_measurements")
        if any(self.primary_sensor not in tool.allowed_sensors for tool in self.allowed_tools):
            raise ValueError("every allowed tool must explicitly allow the primary sensor")
        if not any(
            self.measurement_metric_key in tool.input_metric_keys
            for tool in self.allowed_tools
        ):
            raise ValueError("measurement_metric_key must be consumed by an allowed tool")
        return self


class ExperimentTask(StrictBase):
    task_id: Identifier
    sequence: int = Field(ge=1, le=32)
    title: ShortText
    role: ExperimentTaskRole
    instruction: LongText
    sensor: SensorKind
    analyzer_id: Identifier
    recommended_phyphox_experiment: ShortText
    condition_id: Identifier
    parameter_definitions: list[ExperimentParameterDefinition] = Field(
        default_factory=list,
        max_length=16,
    )
    parameter_targets: list[ExperimentParameterValue] = Field(default_factory=list, max_length=16)
    controls: list[ShortText] = Field(default_factory=list, max_length=16)
    tool_ids: list[Identifier] = Field(min_length=1, max_length=12)
    selection_source: Literal["protocol", "deterministic", "agent", "fallback"] = (
        "protocol"
    )
    selection_reason_code: Identifier = "protocol-step"
    selection_reason: ShortText = "由已验证实验协议生成。"
    selection_evidence_ids: list[Identifier] = Field(default_factory=list, max_length=16)
    status: ExperimentTaskStatus = "pending"

    @model_validator(mode="after")
    def validate_task_parameters(self) -> Self:
        definitions = {item.key: item for item in self.parameter_definitions}
        targets = {item.key: item for item in self.parameter_targets}
        if len(definitions) != len(self.parameter_definitions):
            raise ValueError("parameter_definitions must not contain duplicates")
        if len(targets) != len(self.parameter_targets):
            raise ValueError("parameter_targets must not contain duplicates")
        unknown_targets = set(targets) - set(definitions)
        if unknown_targets:
            raise ValueError(f"parameter_targets reference unknown definitions: {sorted(unknown_targets)}")
        for key, target in targets.items():
            definitions[key].validate_target(target)
        _ensure_unique(self.controls, field_name="controls")
        _ensure_unique(self.tool_ids, field_name="tool_ids")
        _ensure_unique(self.selection_evidence_ids, field_name="selection_evidence_ids")
        _ensure_safe_display_text(self.selection_reason, field_name="selection_reason")
        return self


class RecordingRef(StrictBase):
    recording_type: Literal["sensor_v2", "acceleration_v1"]
    recording_id: OpaqueRecordId
    sensor: SensorKind
    analyzer_id: Identifier
    analyzer_version: SemanticVersion
    source: Literal[
        "phyphox_remote",
        "phone_upload",
        "file_import",
        "public_replay",
        "test_fixture",
    ]
    config_sha256: Annotated[
        str,
        StringConstraints(pattern=r"^[0-9a-f]{64}$"),
    ] | None = None
    remote_session: ShortText | None = None

    @model_validator(mode="after")
    def validate_recording_type(self) -> Self:
        if self.recording_type == "acceleration_v1" and self.sensor != "accelerometer":
            raise ValueError("acceleration_v1 recording references must use accelerometer")
        return self


class MetricSnapshot(StrictBase):
    key: Identifier
    label: ShortText
    value: float
    unit: Unit


class SensorAnalysisSnapshot(StrictBase):
    schema_version: Literal["2.0"] = "2.0"
    sensor: SensorKind
    analyzer_id: Identifier
    analyzer_version: SemanticVersion
    sample_count: int = Field(ge=2, le=120_000)
    duration_s: float = Field(gt=0, le=86_400)
    sampling_rate_hz: float = Field(gt=0, le=100_000)
    sampling_jitter_ratio: float = Field(ge=0)
    max_sampling_gap_ratio: float = Field(gt=0)
    confidence: Literal["low", "medium", "high"]
    warnings: list[ShortText] = Field(default_factory=list, max_length=32)
    metrics: list[MetricSnapshot] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        _ensure_unique([item.key for item in self.metrics], field_name="metrics")
        return self

    @classmethod
    def from_sensor_analysis(cls, analysis: SensorAnalysis) -> SensorAnalysisSnapshot:
        return cls.model_validate(analysis.model_dump(mode="python"))


class ExperimentEvidence(StrictBase):
    evidence_id: Identifier
    task_id: Identifier
    condition_id: Identifier
    recording: RecordingRef
    role: ExperimentTaskRole
    parameters: list[ExperimentParameterValue] = Field(default_factory=list, max_length=16)
    quality: Literal["low", "medium", "high"]
    analysis: SensorAnalysisSnapshot | None = None
    observation_notes: str = Field(default="", max_length=800)
    valid: StrictBool
    rejection_reasons: list[ShortText] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _ensure_unique([item.key for item in self.parameters], field_name="parameters")
        _ensure_unique(self.rejection_reasons, field_name="rejection_reasons")
        if self.valid:
            if self.analysis is None:
                raise ValueError("valid evidence requires a SensorAnalysis snapshot")
            if self.rejection_reasons:
                raise ValueError("valid evidence cannot contain rejection reasons")
        elif not self.rejection_reasons:
            raise ValueError("invalid evidence requires at least one rejection reason")
        if self.analysis is not None and self.analysis.sensor != self.recording.sensor:
            raise ValueError("analysis sensor must match recording sensor")
        if self.analysis is not None and (
            self.analysis.analyzer_id != self.recording.analyzer_id
            or self.analysis.analyzer_version != self.recording.analyzer_version
        ):
            raise ValueError("analysis identity must match recording analyzer identity")
        return self


class ToolExecution(StrictBase):
    execution_id: Identifier
    sequence: int = Field(ge=1, le=128)
    task_id: Identifier
    tool_id: Identifier
    tool_version: SemanticVersion
    input_evidence_ids: list[Identifier] = Field(min_length=1, max_length=32)
    parameters: list[ExperimentParameterValue] = Field(default_factory=list, max_length=16)
    status: Literal["pending", "succeeded", "rejected", "failed"]
    result_metrics: list[MetricSnapshot] = Field(default_factory=list, max_length=64)
    error_code: Identifier | None = None
    error_message: ShortText | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        _ensure_unique(self.input_evidence_ids, field_name="input_evidence_ids")
        _ensure_unique([item.key for item in self.parameters], field_name="parameters")
        _ensure_unique([item.key for item in self.result_metrics], field_name="result_metrics")
        if self.status == "succeeded" and (self.error_code or self.error_message):
            raise ValueError("succeeded tools cannot contain errors")
        if self.status in {"failed", "rejected"} and not self.error_code:
            raise ValueError("failed or rejected tools require error_code")
        if self.status == "pending" and (self.result_metrics or self.error_code or self.error_message):
            raise ValueError("pending tools cannot contain results or errors")
        return self


class LightPlannerCandidate(StrictBase):
    candidate_id: Identifier
    distance_m: float = Field(ge=0.1, le=4.0)
    unit: Literal["m"] = "m"
    projected_span_ratio: float = Field(ge=1.0)
    risk_codes: list[Identifier] = Field(default_factory=list, max_length=8)
    server_reason: ShortText

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        _ensure_unique(self.risk_codes, field_name="risk_codes")
        _ensure_safe_display_text(self.server_reason, field_name="server_reason")
        return self


class LightPlannerRequest(StrictBase):
    schema_version: Literal["1.0"] = "1.0"
    operation: Literal["select_next_design_point"] = "select_next_design_point"
    case_id: Identifier
    expected_revision: int = Field(ge=1)
    completed_task_id: Identifier
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    protocol_id: Identifier
    protocol_version: SemanticVersion
    research_question: LongText
    condition_number: int = Field(ge=1, le=3)
    background_lx: float = Field(ge=0)
    latest_distance_m: float = Field(ge=0.1, le=4.0)
    median_net_illuminance_lx: float = Field(gt=0)
    repeatability_ratio: float = Field(ge=0)
    upper_plateau_fraction: float = Field(ge=0, le=1)
    signal_to_background_ratio: float = Field(ge=0)
    context_untrusted: str = Field(default="", max_length=1200)
    observation_notes_untrusted: str = Field(default="", max_length=1600)
    execution_constraints: list[ExperimentParameterConstraint] = Field(
        default_factory=list,
        max_length=16,
    )
    input_evidence_ids: list[Identifier] = Field(min_length=2, max_length=16)
    candidates: list[LightPlannerCandidate] = Field(min_length=2, max_length=4)
    fallback_candidate_id: Identifier

    @model_validator(mode="after")
    def validate_request_graph(self) -> Self:
        candidate_ids = [item.candidate_id for item in self.candidates]
        _ensure_unique(candidate_ids, field_name="planner candidates")
        _ensure_unique(self.input_evidence_ids, field_name="input_evidence_ids")
        _ensure_unique(
            [item.key for item in self.execution_constraints],
            field_name="execution_constraints",
        )
        if self.fallback_candidate_id not in candidate_ids:
            raise ValueError("fallback_candidate_id must reference a planner candidate")
        distance_constraint = next(
            (item for item in self.execution_constraints if item.key == "distance_m"),
            None,
        )
        if distance_constraint is not None and any(
            not distance_constraint.allows(item.distance_m) for item in self.candidates
        ):
            raise ValueError("planner candidates must already satisfy execution constraints")
        return self


class LightPlannerDecision(StrictBase):
    schema_version: Literal["1.0"] = "1.0"
    case_id: Identifier
    expected_revision: int = Field(ge=1)
    completed_task_id: Identifier
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    selected_candidate_id: Identifier
    rationale_code: PlannerRationaleCode


class PlannerDecisionTrace(StrictBase):
    decision_id: Identifier
    sequence: int = Field(ge=1, le=32)
    source_task_id: Identifier
    planned_task_id: Identifier
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    candidate_ids: list[Identifier] = Field(min_length=2, max_length=4)
    selected_candidate_id: Identifier
    fallback_candidate_id: Identifier
    rationale_code: PlannerRationaleCode
    transport: PlannerTransport = "not_attempted"
    transport_fallback_reason: Identifier | None = None
    source: Literal["agent", "deterministic_fallback"]
    outcome: Literal["accepted", "fallback"]
    fallback_reason: Identifier | None = None
    input_evidence_ids: list[Identifier] = Field(min_length=2, max_length=16)
    revision_before: int = Field(ge=1)
    revision_after: int = Field(ge=2)
    run_id: Identifier | None = None
    model: ShortText | None = None
    attempts: int = Field(default=0, ge=0, le=4)
    model_requests: int = Field(default=0, ge=0, le=8)
    tool_calls: int = Field(default=0, ge=0, le=8)
    elapsed_s: float = Field(default=0.0, ge=0, le=600)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    token_budget_exceeded: StrictBool = False
    allowlist_respected: Literal[True] = True

    @model_validator(mode="after")
    def validate_trace(self) -> Self:
        _ensure_unique(self.candidate_ids, field_name="candidate_ids")
        _ensure_unique(self.input_evidence_ids, field_name="input_evidence_ids")
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected_candidate_id must reference a candidate")
        if self.fallback_candidate_id not in self.candidate_ids:
            raise ValueError("fallback_candidate_id must reference a candidate")
        if self.revision_after != self.revision_before + 1:
            raise ValueError("planner trace must describe a single-revision commit")
        if self.source == "agent":
            if self.outcome != "accepted" or self.fallback_reason is not None:
                raise ValueError("accepted Agent traces cannot contain a fallback reason")
        else:
            if self.outcome != "fallback" or self.fallback_reason is None:
                raise ValueError("fallback traces require a fallback reason")
            if self.selected_candidate_id != self.fallback_candidate_id:
                raise ValueError("fallback traces must select the frozen fallback candidate")
        return self


class VisualizationAxis(StrictBase):
    field_key: Identifier
    label: ShortText
    unit: Unit
    scale: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def validate_axis(self) -> Self:
        _ensure_safe_display_text(self.label, field_name="axis label")
        if self.field_key in _ABSOLUTE_LOCATION_FIELDS:
            raise ValueError("absolute latitude/longitude fields cannot be visualization axes")
        return self


class VisualizationPoint(StrictBase):
    x: float
    y: float
    x_error: float | None = Field(default=None, ge=0)
    y_error: float | None = Field(default=None, ge=0)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_point_references(self) -> Self:
        _ensure_unique(self.evidence_ids, field_name="evidence_ids")
        return self


class VisualizationSeries(StrictBase):
    series_id: Identifier
    label: ShortText
    series_type: Literal["observations", "fit", "reference"]
    points: list[VisualizationPoint] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_series(self) -> Self:
        _ensure_safe_display_text(self.label, field_name="series label")
        return self


class VisualizationArtifact(StrictBase):
    artifact_id: Identifier
    kind: Literal["line", "scatter", "scatter_with_fit", "bar"]
    title: ShortText
    x_axis: VisualizationAxis
    y_axis: VisualizationAxis
    series: list[VisualizationSeries] = Field(min_length=1, max_length=16)
    source_evidence_ids: list[Identifier] = Field(min_length=1, max_length=64)
    source_tool_execution_ids: list[Identifier] = Field(default_factory=list, max_length=32)
    warnings: list[ShortText] = Field(default_factory=list, max_length=16)
    claim_boundaries: list[ShortText] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_artifact_references(self) -> Self:
        _ensure_safe_display_text(self.title, field_name="artifact title")
        _ensure_unique([item.series_id for item in self.series], field_name="series")
        _ensure_unique(self.source_evidence_ids, field_name="source_evidence_ids")
        _ensure_unique(
            self.source_tool_execution_ids,
            field_name="source_tool_execution_ids",
        )
        source_evidence = set(self.source_evidence_ids)
        point_evidence = {
            evidence_id
            for series in self.series
            for point in series.points
            for evidence_id in point.evidence_ids
        }
        unknown = point_evidence - source_evidence
        if unknown:
            raise ValueError(f"visualization points reference unknown evidence: {sorted(unknown)}")
        for warning in (*self.warnings, *self.claim_boundaries):
            _ensure_safe_display_text(warning, field_name="visualization text")
        return self


class ExperimentProgress(StrictBase):
    measurements_used: int = Field(ge=0, le=32)
    corrections_used: int = Field(ge=0, le=16)
    valid_evidence_count: int = Field(ge=0, le=32)
    distinct_condition_count: int = Field(ge=0, le=32)
    condition_coverage_ratio: float = Field(ge=0, le=1)
    quality_pass_rate: float = Field(ge=0, le=1)
    recent_information_gain: float = Field(ge=0, le=1)
    conclusion_ready: StrictBool = False
    forced_stop: StrictBool = False
    decision: Literal["continue", "conclude", "inconclusive"] = "continue"
    blockers: list[ShortText] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.corrections_used > self.measurements_used:
            raise ValueError("corrections_used cannot exceed measurements_used")
        if self.valid_evidence_count > self.measurements_used:
            raise ValueError("valid_evidence_count cannot exceed measurements_used")
        if self.conclusion_ready and self.forced_stop:
            raise ValueError("conclusion_ready and forced_stop are mutually exclusive")
        if self.decision == "conclude" and not self.conclusion_ready:
            raise ValueError("conclude decision requires conclusion_ready")
        if self.decision == "inconclusive" and not self.forced_stop:
            raise ValueError("inconclusive decision requires forced_stop")
        if self.decision == "continue" and (self.conclusion_ready or self.forced_stop):
            raise ValueError("continue decision cannot already be terminal")
        if self.decision == "conclude" and self.blockers:
            raise ValueError("a conclusion-ready progress vector cannot retain blockers")
        return self


class ExperimentReport(StrictBase):
    outcome: Literal["completed_with_conclusion", "completed_inconclusive"]
    confidence: Literal["low", "medium", "high"]
    conclusion: LongText
    evidence_ids: list[Identifier] = Field(default_factory=list, max_length=32)
    tool_execution_ids: list[Identifier] = Field(default_factory=list, max_length=32)
    artifact_ids: list[Identifier] = Field(default_factory=list, max_length=16)
    summary_metrics: list[MetricSnapshot] = Field(default_factory=list, max_length=24)
    remaining_uncertainties: list[ShortText] = Field(default_factory=list, max_length=16)
    claim_boundaries: list[ShortText] = Field(min_length=1, max_length=12)
    stop_reason_code: Identifier = "protocol-termination"
    stop_reason: ShortText
    market_validated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report_references(self) -> Self:
        _ensure_unique(self.evidence_ids, field_name="evidence_ids")
        _ensure_unique(self.tool_execution_ids, field_name="tool_execution_ids")
        _ensure_unique(self.artifact_ids, field_name="artifact_ids")
        _ensure_unique([item.key for item in self.summary_metrics], field_name="summary_metrics")
        return self


class InvestigationCase(StrictBase):
    schema_version: Literal["2.0"] = "2.0"
    case_id: Identifier
    revision: int = Field(ge=1)
    title: ShortText
    research_question: LongText
    context: str = Field(default="", max_length=1200)
    mode: InvestigationMode
    status: InvestigationStatus = "planning"
    plan_source: Literal["validated_protocol", "agent_allowlisted"]
    planning_policy: InvestigationPlanningPolicy = "deterministic"
    protocol: ExperimentProtocol
    execution_constraints: list[ExperimentParameterConstraint] = Field(
        default_factory=list,
        max_length=16,
    )
    current_task: ExperimentTask | None = None
    completed_tasks: list[ExperimentTask] = Field(default_factory=list, max_length=32)
    evidence: list[ExperimentEvidence] = Field(default_factory=list, max_length=32)
    tool_trace: list[ToolExecution] = Field(default_factory=list, max_length=128)
    planner_trace: list[PlannerDecisionTrace] = Field(default_factory=list, max_length=32)
    artifacts: list[VisualizationArtifact] = Field(default_factory=list, max_length=16)
    progress: ExperimentProgress
    report: ExperimentReport | None = None

    @model_validator(mode="after")
    def validate_case_graph(self) -> Self:
        if self.mode != self.protocol.mode:
            raise ValueError("case mode must match protocol mode")
        tasks = [*self.completed_tasks]
        if self.current_task is not None:
            tasks.append(self.current_task)
        task_ids = [item.task_id for item in tasks]
        task_sequences = [item.sequence for item in tasks]
        _ensure_unique(task_ids, field_name="tasks")
        if len(task_sequences) != len(set(task_sequences)):
            raise ValueError("task sequence numbers must be unique")
        if len(tasks) > self.protocol.max_measurements:
            raise ValueError("case task count exceeds protocol max_measurements")
        if self.progress.measurements_used > self.protocol.max_measurements:
            raise ValueError("progress exceeds protocol max_measurements")
        if self.progress.corrections_used > self.protocol.max_corrections:
            raise ValueError("progress exceeds protocol max_corrections")
        allowed_tools = {item.tool_id: item for item in self.protocol.allowed_tools}
        protocol_parameters = {item.key: item for item in self.protocol.parameters}
        constraint_keys = [item.key for item in self.execution_constraints]
        _ensure_unique(constraint_keys, field_name="execution_constraints")
        constraints = {item.key: item for item in self.execution_constraints}
        for constraint in self.execution_constraints:
            definition = protocol_parameters.get(constraint.key)
            if definition is None:
                raise ValueError("execution constraint references an unknown protocol parameter")
            constraint.validate_definition(definition)
        for task in tasks:
            if task.sensor != self.protocol.primary_sensor:
                raise ValueError("task sensor must match protocol primary_sensor")
            if task.analyzer_id != self.protocol.required_analyzer_id:
                raise ValueError("task analyzer_id must match protocol required_analyzer_id")
            if not set(task.tool_ids).issubset(allowed_tools):
                raise ValueError("task references a tool outside the protocol allowlist")
            for definition in task.parameter_definitions:
                protocol_definition = protocol_parameters.get(definition.key)
                if protocol_definition is None or protocol_definition != definition:
                    raise ValueError("task parameter definition must match the protocol")
            for target in task.parameter_targets:
                constraint = constraints.get(target.key)
                if constraint is not None and not constraint.allows(float(target.value)):
                    raise ValueError("task parameter target violates an execution constraint")
        if self.current_task is not None and self.current_task.status in {"completed", "rejected"}:
            raise ValueError("current_task must still be actionable")
        if any(item.status not in {"completed", "rejected"} for item in self.completed_tasks):
            raise ValueError("completed_tasks may only contain completed or rejected tasks")

        known_tasks = set(task_ids)
        evidence_ids = [item.evidence_id for item in self.evidence]
        execution_ids = [item.execution_id for item in self.tool_trace]
        artifact_ids = [item.artifact_id for item in self.artifacts]
        decision_ids = [item.decision_id for item in self.planner_trace]
        _ensure_unique(evidence_ids, field_name="evidence")
        _ensure_unique(execution_ids, field_name="tool_trace")
        _ensure_unique(artifact_ids, field_name="artifacts")
        _ensure_unique(decision_ids, field_name="planner_trace")
        known_evidence = set(evidence_ids)
        known_executions = set(execution_ids)
        known_artifacts = set(artifact_ids)
        for item in self.evidence:
            if item.task_id not in known_tasks:
                raise ValueError("evidence references an unknown task")
            task = next(task for task in tasks if task.task_id == item.task_id)
            if (
                item.role != task.role
                or item.condition_id != task.condition_id
                or item.recording.sensor != task.sensor
            ):
                raise ValueError("evidence role, condition and sensor must match its task")
        for task in tasks:
            if not set(task.selection_evidence_ids).issubset(known_evidence):
                raise ValueError("task selection references unknown evidence")
        for execution in self.tool_trace:
            if execution.task_id not in known_tasks:
                raise ValueError("tool execution references an unknown task")
            if execution.tool_id not in allowed_tools:
                raise ValueError("tool execution is outside the protocol allowlist")
            execution_task = next(task for task in tasks if task.task_id == execution.task_id)
            if execution.tool_id not in execution_task.tool_ids:
                raise ValueError("tool execution is outside its task allowlist")
            if execution.tool_version != allowed_tools[execution.tool_id].version:
                raise ValueError("tool execution version does not match the allowlist")
            if not set(execution.input_evidence_ids).issubset(known_evidence):
                raise ValueError("tool execution references unknown evidence")
        for artifact in self.artifacts:
            if not set(artifact.source_evidence_ids).issubset(known_evidence):
                raise ValueError("artifact references unknown case evidence")
            if not set(artifact.source_tool_execution_ids).issubset(known_executions):
                raise ValueError("artifact references unknown tool executions")
        for decision in self.planner_trace:
            if decision.source_task_id not in known_tasks:
                raise ValueError("planner trace references an unknown source task")
            if decision.planned_task_id not in known_tasks:
                raise ValueError("planner trace references an unknown planned task")
            if not set(decision.input_evidence_ids).issubset(known_evidence):
                raise ValueError("planner trace references unknown evidence")
        terminal = self.status in {
            "completed_with_conclusion",
            "completed_inconclusive",
        }
        if terminal:
            if self.current_task is not None or self.report is None:
                raise ValueError("completed cases require a report and no current task")
            if self.report.outcome != self.status:
                raise ValueError("report outcome must match case status")
            expected_decision = (
                "conclude"
                if self.status == "completed_with_conclusion"
                else "inconclusive"
            )
            if self.progress.decision != expected_decision:
                raise ValueError("terminal case status must match the progress decision")
        elif self.report is not None:
            raise ValueError("non-completed cases cannot contain a final report")
        if self.report is not None:
            if not set(self.report.evidence_ids).issubset(known_evidence):
                raise ValueError("report references unknown evidence")
            if not set(self.report.tool_execution_ids).issubset(known_executions):
                raise ValueError("report references unknown tool executions")
            if not set(self.report.artifact_ids).issubset(known_artifacts):
                raise ValueError("report references unknown artifacts")
        return self


class InvestigationCaseCreate(StrictBase):
    title: ShortText
    research_question: LongText
    mode: InvestigationMode
    context: str = Field(default="", max_length=1200)
    planning_policy: InvestigationPlanningPolicy = "deterministic"
    protocol_id: Identifier | None = None
    protocol_version: SemanticVersion | None = None
    parameter_values: list[ExperimentParameterValue] = Field(default_factory=list, max_length=16)
    execution_constraints: list[ExperimentParameterConstraint] = Field(
        default_factory=list,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_protocol_selection(self) -> Self:
        if (self.protocol_id is None) != (self.protocol_version is None):
            raise ValueError("protocol_id and protocol_version must be supplied together")
        _ensure_unique([item.key for item in self.parameter_values], field_name="parameter_values")
        _ensure_unique(
            [item.key for item in self.execution_constraints],
            field_name="execution_constraints",
        )
        return self


class InvestigationCaseHistoryItem(StrictBase):
    case_id: Identifier
    revision: int = Field(ge=1)
    title: ShortText
    mode: InvestigationMode
    status: InvestigationStatus
    primary_sensor: SensorKind
    current_task_title: ShortText | None = None
    evidence_count: int = Field(ge=0, le=32)
    artifact_count: int = Field(ge=0, le=16)
    created_at: ShortText
    updated_at: ShortText


class InvestigationMeasurementSubmit(StrictBase):
    expected_revision: int = Field(ge=1)
    task_id: Identifier
    recording: RecordingRef
    parameters: list[ExperimentParameterValue] = Field(default_factory=list, max_length=16)
    controls_confirmed: StrictBool = False
    observation_notes: str = Field(default="", max_length=800)

    @model_validator(mode="after")
    def unique_parameters(self) -> Self:
        _ensure_unique([item.key for item in self.parameters], field_name="parameters")
        return self


class InvestigationPhyphoxCaptureRequest(StrictBase):
    expected_revision: int = Field(ge=1)
    task_id: Identifier
    base_url: Annotated[
        str,
        StringConstraints(min_length=10, max_length=200, pattern=r"^https?://[^\s]+$"),
    ]
    duration_s: float = Field(ge=1, le=300)
    parameters: list[ExperimentParameterValue] = Field(default_factory=list, max_length=16)
    controls_confirmed: StrictBool = False
    privacy_acknowledged: StrictBool = False
    observation_notes: str = Field(default="", max_length=800)

    @model_validator(mode="after")
    def unique_parameters(self) -> Self:
        _ensure_unique([item.key for item in self.parameters], field_name="parameters")
        return self


class InvestigationCaptureMetadata(StrictBase):
    source: Literal["phyphox_remote"] = "phyphox_remote"
    experiment_title: ShortText
    remote_session: ShortText | None = None
    config_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    requested_duration_s: float = Field(ge=1, le=300)
    actual_duration_s: float = Field(gt=0, le=360)
    sample_count: int = Field(ge=2, le=120_000)
    sensor: SensorKind
    analyzer_id: Identifier


class InvestigationPhyphoxCaptureResponse(StrictBase):
    case: InvestigationCase
    evidence: ExperimentEvidence
    capture: InvestigationCaptureMetadata

    @model_validator(mode="after")
    def validate_capture_binding(self) -> Self:
        if self.evidence.recording.recording_type != "sensor_v2":
            raise ValueError("Investigation phyphox capture must produce a sensor_v2 recording")
        if self.capture.sensor != self.evidence.recording.sensor:
            raise ValueError("capture sensor must match evidence recording sensor")
        if self.capture.analyzer_id != self.case.protocol.required_analyzer_id:
            raise ValueError("capture analyzer must match case protocol")
        if self.capture.analyzer_id != self.evidence.recording.analyzer_id:
            raise ValueError("capture analyzer must match evidence recording analyzer")
        if not any(item.evidence_id == self.evidence.evidence_id for item in self.case.evidence):
            raise ValueError("capture evidence must already be present in the returned case")
        return self
