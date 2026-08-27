from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pocketlab.public_replay_dataset import PublicDataClass
from pocketlab.sensor_models import SensorAnalysis, SensorKind

PublicSensorAgentKind = Literal[
    "accelerometer",
    "gyroscope",
    "magnetometer",
    "proximity",
    "microphone",
    "location",
]
PublicSensorPlannerOperation = Literal["select_evidence_route", "select_report_action"]

_ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,119}$"
_CODE_PATTERN = r"^[a-z][a-z0-9_]{2,79}$"
_ERROR_TYPE_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,119}$"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class PublicSensorExploreRequest(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    sensor: PublicSensorAgentKind
    protocol_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    research_question: str = Field(min_length=5, max_length=800)
    privacy_acknowledged: bool = False

    @field_validator("research_question")
    @classmethod
    def question_has_no_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\t" for character in value):
            raise ValueError("research_question contains unsupported control characters")
        return value


class PublicSensorPlanCandidate(_FrozenStrictModel):
    candidate_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=3, max_length=160)
    server_reason: str = Field(min_length=10, max_length=420)
    rationale_code: str = Field(pattern=_CODE_PATTERN)
    recording_ids: tuple[str, ...] = Field(default=(), max_length=3)
    tool_ids: tuple[str, ...] = Field(default=(), max_length=4)
    terminal: bool = False
    result_code: str = Field(pattern=_CODE_PATTERN)

    @field_validator("recording_ids", "tool_ids")
    @classmethod
    def identifiers_are_safe_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("candidate identifiers must be unique")
        if any(not value or len(value) > 120 for value in values):
            raise ValueError("candidate identifiers must be bounded")
        return values

    @model_validator(mode="after")
    def execution_contract_is_closed(self) -> Self:
        if self.terminal:
            if self.recording_ids or self.tool_ids:
                raise ValueError("terminal candidates cannot bind evidence or tools")
        elif not self.recording_ids or not self.tool_ids:
            raise ValueError("evidence candidates require recordings and tools")
        return self


class PublicSensorEvidenceView(_FrozenStrictModel):
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=3)
    confidence: Literal["low", "medium", "high"] | None = None
    quality_passed: bool | None = None
    result_codes: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def evidence_summary_is_complete(self) -> Self:
        if self.evidence_ids:
            if self.confidence is None or self.quality_passed is None:
                raise ValueError("present evidence requires confidence and quality state")
        elif (
            self.confidence is not None
            or self.quality_passed is not None
            or self.result_codes
        ):
            raise ValueError("empty evidence view cannot expose result state")
        return self


class PublicSensorPlannerRequest(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    sensor: PublicSensorAgentKind
    protocol_id: str = Field(pattern=_ID_PATTERN)
    protocol_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    operation: PublicSensorPlannerOperation
    run_id: str = Field(pattern=_ID_PATTERN)
    step: int = Field(ge=1, le=2)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_question_untrusted: str = Field(min_length=5, max_length=800)
    privacy_acknowledged: bool
    selection_policy: tuple[str, ...] = Field(min_length=1, max_length=12)
    evidence_view: PublicSensorEvidenceView
    candidates: tuple[PublicSensorPlanCandidate, ...] = Field(min_length=2, max_length=5)
    fallback_candidate_id: str = Field(pattern=_ID_PATTERN)

    @model_validator(mode="after")
    def planner_graph_is_closed(self) -> Self:
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Planner candidate IDs must be unique")
        if self.fallback_candidate_id not in candidate_ids:
            raise ValueError("fallback must reference a frozen candidate")
        if not self.privacy_acknowledged:
            raise ValueError("public sensor Planner requires privacy acknowledgement")
        if self.step == 1:
            if self.operation != "select_evidence_route" or self.evidence_view.evidence_ids:
                raise ValueError("step one selects evidence without prior evidence")
        elif self.operation != "select_report_action" or not self.evidence_view.evidence_ids:
            raise ValueError("step two requires evidence")
        return self


class PublicSensorPlannerDecision(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(pattern=_ID_PATTERN)
    step: int = Field(ge=1, le=2)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str = Field(pattern=_ID_PATTERN)
    rationale_code: str = Field(pattern=_CODE_PATTERN)


class PublicSensorRuntimeToolEvent(_FrozenStrictModel):
    name: str = Field(pattern=_ID_PATTERN)
    status: Literal["called", "returned", "committed", "error"]


class PublicSensorRuntimeTrace(_FrozenStrictModel):
    run_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    operation: str | None = Field(default=None, pattern=_ID_PATTERN)
    model: str | None = Field(default=None, max_length=160)
    status: Literal["completed", "failed", "cancelled"] | None = None
    elapsed_s: float | None = Field(default=None, ge=0.0, le=120.0)
    timeout_s: float | None = Field(default=None, gt=0.0, le=30.0)
    max_turns: int | None = Field(default=None, ge=1, le=3)
    retry_limit: int | None = Field(default=None, ge=0, le=1)
    model_requests: int | None = Field(default=None, ge=0, le=4)
    tool_calls: int | None = Field(default=None, ge=0, le=2)
    tool_events: tuple[PublicSensorRuntimeToolEvent, ...] = Field(default=(), max_length=4)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_reported: bool | None = None
    token_budget: int | None = Field(default=None, ge=1, le=4_000)
    token_budget_exceeded: bool | None = None
    error_kind: str | None = Field(default=None, pattern=_ID_PATTERN)
    error_type: str | None = Field(default=None, pattern=_ERROR_TYPE_PATTERN)
    transport: Literal["function_tool", "validated_json_text"] | None = None
    transport_fallback_reason: str | None = Field(default=None, pattern=_ID_PATTERN)


class PublicSensorPlannerTrace(_FrozenStrictModel):
    step: int = Field(ge=1, le=2)
    operation: PublicSensorPlannerOperation
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ids: tuple[str, ...] = Field(min_length=2, max_length=5)
    selected_candidate_id: str = Field(pattern=_ID_PATTERN)
    fallback_candidate_id: str = Field(pattern=_ID_PATTERN)
    rationale_code: str = Field(pattern=_CODE_PATTERN)
    source: Literal["agent", "strong_workflow_fallback"]
    outcome: Literal["accepted", "fallback"]
    fallback_reason: str | None = Field(default=None, max_length=120)
    transport: Literal[
        "not_attempted", "function_tool", "validated_json_text"
    ] = "not_attempted"
    runtime_trace: PublicSensorRuntimeTrace | None = None

    @model_validator(mode="after")
    def selection_is_consistent(self) -> Self:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("Planner trace candidates must be unique")
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected candidate must be frozen")
        if self.fallback_candidate_id not in self.candidate_ids:
            raise ValueError("fallback candidate must be frozen")
        if self.source == "strong_workflow_fallback":
            if self.outcome != "fallback" or not self.fallback_reason:
                raise ValueError("workflow fallback requires a reason")
            if self.selected_candidate_id != self.fallback_candidate_id:
                raise ValueError("workflow fallback must select its frozen fallback")
        elif self.outcome != "accepted" or self.fallback_reason is not None:
            raise ValueError("accepted Agent trace cannot contain fallback state")
        return self


class PublicSensorToolExecution(_FrozenStrictModel):
    sequence: int = Field(ge=1, le=8)
    tool_id: str = Field(pattern=_ID_PATTERN)
    status: Literal["completed", "failed", "timeout", "rejected"] = "completed"
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=3)
    result_codes: tuple[str, ...] = Field(default=(), max_length=12)


class PublicSensorEvidenceSnapshot(_FrozenStrictModel):
    evidence_id: str = Field(pattern=_ID_PATTERN)
    dataset_id: str = Field(pattern=_ID_PATTERN)
    recording_id: str = Field(pattern=_ID_PATTERN)
    sensor: SensorKind
    data_class: PublicDataClass
    condition_label: str = Field(min_length=3, max_length=240)
    device_scope: str = Field(min_length=3, max_length=240)
    source_title: str = Field(min_length=3, max_length=240)
    source_url: str = Field(min_length=12, max_length=500)
    doi: str = Field(min_length=3, max_length=160)
    license_spdx: str = Field(min_length=2, max_length=80)
    analysis: SensorAnalysis
    processing_disclosures: tuple[str, ...] = Field(min_length=1, max_length=16)
    claim_boundary: tuple[str, ...] = Field(min_length=1, max_length=16)
    gate_c_eligible: Literal[False] = False

    @model_validator(mode="after")
    def evidence_identity_is_consistent(self) -> Self:
        if self.analysis.sensor != self.sensor:
            raise ValueError("evidence sensor must match its analysis")
        return self


class PublicSensorComparisonMetric(_FrozenStrictModel):
    key: str = Field(pattern=_CODE_PATTERN)
    label: str = Field(min_length=2, max_length=120)
    value: float
    unit: str = Field(max_length=24)


class PublicSensorComparison(_FrozenStrictModel):
    comparison_id: str = Field(pattern=_ID_PATTERN)
    sensor: PublicSensorAgentKind
    status: Literal["passed", "failed", "not_evaluable"]
    quality_passed: bool
    result_codes: tuple[str, ...] = Field(min_length=1, max_length=12)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=3)
    metrics: tuple[PublicSensorComparisonMetric, ...] = Field(default=(), max_length=12)
    interpretation: str = Field(min_length=10, max_length=800)
    gate_c_eligible: Literal[False] = False

    @model_validator(mode="after")
    def status_matches_quality(self) -> Self:
        if self.status == "passed" and not self.quality_passed:
            raise ValueError("passed comparison requires the quality gate")
        if self.status != "passed" and self.quality_passed:
            raise ValueError("non-passing comparison cannot pass quality")
        return self


class PublicSensorFinding(_FrozenStrictModel):
    finding_id: str = Field(pattern=_ID_PATTERN)
    text: str = Field(min_length=5, max_length=600)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=3)


class PublicSensorReportSource(_FrozenStrictModel):
    dataset_id: str = Field(pattern=_ID_PATTERN)
    data_class: PublicDataClass
    device_scope: str = Field(min_length=3, max_length=240)
    source_title: str = Field(min_length=3, max_length=240)
    source_url: str = Field(min_length=12, max_length=500)
    doi: str = Field(min_length=3, max_length=160)
    license_spdx: str = Field(min_length=2, max_length=80)
    gate_c_eligible: Literal[False] = False


class PublicSensorReport(_FrozenStrictModel):
    conclusion_kind: Literal[
        "supported_with_limits",
        "limited",
        "unsupported",
        "live_measurement_required",
        "privacy_acknowledgement_required",
    ]
    title: str = Field(min_length=3, max_length=180)
    summary: str = Field(min_length=10, max_length=1_500)
    supported_findings: tuple[PublicSensorFinding, ...] = Field(default=(), max_length=8)
    uncertainties: tuple[str, ...] = Field(min_length=1, max_length=16)
    forbidden_claims: tuple[str, ...] = Field(min_length=1, max_length=16)
    next_live_measurement: str | None = Field(default=None, max_length=1_200)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=3)
    source_ids: tuple[str, ...] = Field(default=(), max_length=3)
    sources: tuple[PublicSensorReportSource, ...] = Field(default=(), max_length=3)
    gate_c_credited_records: Literal[0] = 0
    gate_e_status: Literal["not_evaluated"] = "not_evaluated"
    gate_h_status: Literal["not_evaluated"] = "not_evaluated"
    public_replay_ready: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def report_provenance_is_closed(self) -> Self:
        if tuple(item.dataset_id for item in self.sources) != self.source_ids:
            raise ValueError("report sources must match source IDs")
        evidence = set(self.evidence_ids)
        if any(
            evidence_id not in evidence
            for finding in self.supported_findings
            for evidence_id in finding.evidence_ids
        ):
            raise ValueError("report finding references unknown evidence")
        if self.conclusion_kind == "supported_with_limits" and not self.evidence_ids:
            raise ValueError("supported reports require evidence")
        return self


class PublicSensorExploreResult(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    sensor: PublicSensorAgentKind
    protocol_id: str = Field(pattern=_ID_PATTERN)
    protocol_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(pattern=_ID_PATTERN)
    research_question: str = Field(min_length=5, max_length=800)
    execution_status: Literal["completed", "limited", "unsupported"]
    selected_route_id: str = Field(pattern=_ID_PATTERN)
    planner_status: Literal["accepted", "fallback", "mixed"]
    planner_trace: tuple[PublicSensorPlannerTrace, ...] = Field(min_length=1, max_length=2)
    tool_trace: tuple[PublicSensorToolExecution, ...] = Field(default=(), max_length=8)
    evidence: tuple[PublicSensorEvidenceSnapshot, ...] = Field(default=(), max_length=3)
    comparison: PublicSensorComparison | None = None
    report: PublicSensorReport
    public_replay_ready: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def result_graph_is_closed(self) -> Self:
        expected_planner = (
            "accepted"
            if all(item.source == "agent" for item in self.planner_trace)
            else "fallback"
            if all(item.source == "strong_workflow_fallback" for item in self.planner_trace)
            else "mixed"
        )
        if self.planner_status != expected_planner:
            raise ValueError("planner status does not match traces")
        if self.selected_route_id != self.planner_trace[-1].selected_candidate_id:
            raise ValueError("selected route must match the final Planner action")
        if tuple(item.step for item in self.planner_trace) != tuple(
            range(1, len(self.planner_trace) + 1)
        ):
            raise ValueError("Planner steps must be contiguous")
        if tuple(item.sequence for item in self.tool_trace) != tuple(
            range(1, len(self.tool_trace) + 1)
        ):
            raise ValueError("tool sequence must be contiguous")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        source_ids = tuple(dict.fromkeys(item.dataset_id for item in self.evidence))
        if self.report.evidence_ids != evidence_ids or self.report.source_ids != source_ids:
            raise ValueError("report provenance must match result evidence")
        if self.comparison is not None and self.comparison.evidence_ids != evidence_ids:
            raise ValueError("comparison must cover the exact result evidence")
        if any(
            evidence_id not in set(evidence_ids)
            for execution in self.tool_trace
            for evidence_id in execution.evidence_ids
        ):
            raise ValueError("tool trace references unknown evidence")
        expected_status = (
            "unsupported"
            if self.report.conclusion_kind == "unsupported"
            else "completed"
            if self.report.conclusion_kind == "supported_with_limits"
            else "limited"
        )
        if self.execution_status != expected_status:
            raise ValueError("execution status does not match report")
        return self
