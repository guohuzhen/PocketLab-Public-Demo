from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pocketlab.public_pressure_models import (
    PressureDirection,
    PublicPressureClaimAuditResult,
    PublicPressureHeightComparison,
    PublicPressureTraceResult,
)
from pocketlab.public_replay_dataset import PublicDataClass

PressurePlannerOperation = Literal["select_evidence_route", "select_report_action"]
PressurePlannerRationale = Literal[
    "match_elevator_goal",
    "match_stairwell_goal",
    "request_live_device_evidence",
    "unsupported_claim_boundary",
    "evidence_quality_sufficient",
    "evidence_quality_insufficient",
    "privacy_not_acknowledged",
    "strong_workflow_fallback",
]

_ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,119}$"
_ERROR_TYPE_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,119}$"


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class PublicPressureExploreRequest(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    research_question: str = Field(min_length=5, max_length=800)
    privacy_acknowledged: bool = False

    @field_validator("research_question")
    @classmethod
    def question_has_no_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\t" for character in value):
            raise ValueError("research_question contains unsupported control characters")
        return value


class PublicPressurePlanCandidate(_FrozenStrictModel):
    candidate_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=3, max_length=160)
    server_reason: str = Field(min_length=10, max_length=360)
    recording_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    tool_ids: tuple[str, ...] = Field(default=(), max_length=3)
    terminal: bool = False
    result_code: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)

    @field_validator("tool_ids")
    @classmethod
    def tools_are_safe_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("candidate tool_ids must be unique")
        if any(not value or len(value) > 120 for value in values):
            raise ValueError("candidate tool_ids must contain safe identifiers")
        return values

    @model_validator(mode="after")
    def execution_contract_is_closed(self) -> Self:
        if self.terminal:
            if self.recording_id is not None or self.tool_ids:
                raise ValueError("terminal candidates cannot bind recordings or tools")
        elif self.recording_id is None or not self.tool_ids:
            raise ValueError("evidence candidates require a recording and tool sequence")
        return self


class PublicPressureEvidenceView(_FrozenStrictModel):
    evidence_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    candidate_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    confidence: Literal["low", "medium", "high"] | None = None
    platforms_passed: bool | None = None
    pressure_direction: PressureDirection | None = None
    approximate_height_change_m: float | None = Field(default=None, ge=-1_000, le=1_000)
    warning_codes: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def evidence_fields_are_all_present_or_absent(self) -> Self:
        fields = (
            self.evidence_id,
            self.candidate_id,
            self.confidence,
            self.platforms_passed,
            self.pressure_direction,
            self.approximate_height_change_m,
        )
        if any(value is not None for value in fields) and any(
            value is None for value in fields
        ):
            raise ValueError("pressure evidence view must be complete when present")
        if self.evidence_id is None and self.warning_codes:
            raise ValueError("empty pressure evidence cannot contain warning codes")
        return self


class PublicPressurePlannerRequest(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    operation: PressurePlannerOperation
    run_id: str = Field(pattern=_ID_PATTERN)
    step: int = Field(ge=1, le=2)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_question_untrusted: str = Field(min_length=5, max_length=800)
    privacy_acknowledged: bool
    evidence_view: PublicPressureEvidenceView
    candidates: tuple[PublicPressurePlanCandidate, ...] = Field(min_length=2, max_length=4)
    fallback_candidate_id: str = Field(pattern=_ID_PATTERN)

    @model_validator(mode="after")
    def planner_graph_is_closed(self) -> Self:
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("planner candidate IDs must be unique")
        if self.fallback_candidate_id not in candidate_ids:
            raise ValueError("fallback_candidate_id must reference a frozen candidate")
        if not self.privacy_acknowledged:
            raise ValueError("Pressure replay Planner requires explicit privacy acknowledgement")
        if self.step == 1:
            if self.operation != "select_evidence_route" or self.evidence_view.evidence_id:
                raise ValueError("step one must select an evidence route without evidence")
        elif self.operation != "select_report_action" or not self.evidence_view.evidence_id:
            raise ValueError("step two requires a complete pressure evidence view")
        return self


class PublicPressurePlannerDecision(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(pattern=_ID_PATTERN)
    step: int = Field(ge=1, le=2)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str = Field(pattern=_ID_PATTERN)
    rationale_code: PressurePlannerRationale


class PublicPressureRuntimeToolEvent(_FrozenStrictModel):
    name: str = Field(pattern=_ID_PATTERN)
    status: Literal["called", "returned", "committed", "error"]


class PublicPressureRuntimeTrace(_FrozenStrictModel):
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
    tool_events: tuple[PublicPressureRuntimeToolEvent, ...] = Field(default=(), max_length=4)
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


class PublicPressurePlannerTrace(_FrozenStrictModel):
    step: int = Field(ge=1, le=2)
    operation: PressurePlannerOperation
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ids: tuple[str, ...] = Field(min_length=2, max_length=4)
    selected_candidate_id: str = Field(pattern=_ID_PATTERN)
    fallback_candidate_id: str = Field(pattern=_ID_PATTERN)
    rationale_code: PressurePlannerRationale
    source: Literal["agent", "strong_workflow_fallback"]
    outcome: Literal["accepted", "fallback"]
    fallback_reason: str | None = Field(default=None, max_length=120)
    transport: Literal[
        "not_attempted", "function_tool", "validated_json_text"
    ] = "not_attempted"
    runtime_trace: PublicPressureRuntimeTrace | None = None

    @model_validator(mode="after")
    def selection_is_consistent(self) -> Self:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("planner trace candidate IDs must be unique")
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected candidate must be present in planner trace")
        if self.fallback_candidate_id not in self.candidate_ids:
            raise ValueError("fallback candidate must be present in planner trace")
        if self.source == "strong_workflow_fallback":
            if self.outcome != "fallback" or not self.fallback_reason:
                raise ValueError("workflow fallback requires an explicit reason")
            if self.selected_candidate_id != self.fallback_candidate_id:
                raise ValueError("workflow fallback must use the frozen fallback")
        elif self.outcome != "accepted" or self.fallback_reason is not None:
            raise ValueError("accepted Agent traces cannot contain fallback state")
        return self


class PublicPressureToolExecution(_FrozenStrictModel):
    sequence: int = Field(ge=1, le=3)
    tool_id: str = Field(pattern=_ID_PATTERN)
    status: Literal["completed", "failed", "timeout", "rejected"] = "completed"
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=2)
    result_codes: tuple[str, ...] = Field(default=(), max_length=8)


class PublicPressureEvidenceSnapshot(_FrozenStrictModel):
    evidence_id: str = Field(pattern=_ID_PATTERN)
    dataset_id: str = Field(pattern=_ID_PATTERN)
    recording_id: str = Field(pattern=_ID_PATTERN)
    data_class: PublicDataClass
    device_scope: str = Field(min_length=3, max_length=240)
    source_title: str = Field(min_length=3, max_length=240)
    source_url: str = Field(min_length=12, max_length=500)
    doi: str = Field(min_length=3, max_length=160)
    license_spdx: str = Field(min_length=2, max_length=80)
    inspection: PublicPressureTraceResult
    comparison: PublicPressureHeightComparison
    claim_audit: PublicPressureClaimAuditResult
    processing_disclosures: tuple[str, ...] = Field(min_length=1, max_length=16)
    claim_boundary: tuple[str, ...] = Field(min_length=1, max_length=16)
    gate_c_eligible: Literal[False] = False

    @model_validator(mode="after")
    def evidence_identity_and_boundaries_are_consistent(self) -> Self:
        if self.recording_id != self.inspection.candidate_id:
            raise ValueError("evidence recording must match inspection candidate")
        if self.inspection.gate_c_eligible or self.comparison.gate_c_eligible:
            raise ValueError("public Pressure evidence cannot satisfy Gate C")
        if self.claim_audit.agent_ready or self.claim_audit.market_validated:
            raise ValueError("public Pressure evidence cannot claim production readiness")
        return self


class PublicPressureFinding(_FrozenStrictModel):
    finding_id: str = Field(pattern=_ID_PATTERN)
    text: str = Field(min_length=5, max_length=600)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=2)


class PublicPressureReportSource(_FrozenStrictModel):
    dataset_id: str = Field(pattern=_ID_PATTERN)
    data_class: PublicDataClass
    device_scope: str = Field(min_length=3, max_length=240)
    source_title: str = Field(min_length=3, max_length=240)
    source_url: str = Field(min_length=12, max_length=500)
    doi: str = Field(min_length=3, max_length=160)
    license_spdx: str = Field(min_length=2, max_length=80)
    gate_c_eligible: Literal[False] = False


class PublicPressureReport(_FrozenStrictModel):
    conclusion_kind: Literal[
        "supported_relative_height",
        "limited",
        "unsupported",
        "live_measurement_required",
        "privacy_acknowledgement_required",
    ]
    title: str = Field(min_length=3, max_length=160)
    summary: str = Field(min_length=10, max_length=1_200)
    supported_findings: tuple[PublicPressureFinding, ...] = Field(default=(), max_length=8)
    uncertainties: tuple[str, ...] = Field(min_length=1, max_length=16)
    forbidden_claims: tuple[str, ...] = Field(min_length=1, max_length=16)
    next_live_measurement: str | None = Field(default=None, max_length=1_000)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=2)
    source_ids: tuple[str, ...] = Field(default=(), max_length=2)
    sources: tuple[PublicPressureReportSource, ...] = Field(default=(), max_length=2)
    gate_c_credited_records: Literal[0] = 0
    gate_e_status: Literal["not_evaluated"] = "not_evaluated"
    gate_h_status: Literal["not_evaluated"] = "not_evaluated"
    public_replay_ready: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def report_provenance_is_closed(self) -> Self:
        if tuple(item.dataset_id for item in self.sources) != self.source_ids:
            raise ValueError("report source objects must exactly match source_ids")
        referenced = {
            evidence_id
            for finding in self.supported_findings
            for evidence_id in finding.evidence_ids
        }
        if not referenced.issubset(set(self.evidence_ids)):
            raise ValueError("report findings reference evidence outside the report")
        if self.conclusion_kind == "supported_relative_height" and not self.evidence_ids:
            raise ValueError("supported Pressure reports require evidence")
        return self


class PublicPressureExploreResult(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_id: Literal["pressure-public-exploration.v1"] = (
        "pressure-public-exploration.v1"
    )
    protocol_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(pattern=_ID_PATTERN)
    research_question: str = Field(min_length=5, max_length=800)
    execution_status: Literal["completed", "limited", "unsupported"]
    selected_route_id: str = Field(pattern=_ID_PATTERN)
    planner_status: Literal["accepted", "fallback", "mixed"]
    planner_trace: tuple[PublicPressurePlannerTrace, ...] = Field(min_length=1, max_length=2)
    tool_trace: tuple[PublicPressureToolExecution, ...] = Field(default=(), max_length=3)
    evidence: tuple[PublicPressureEvidenceSnapshot, ...] = Field(default=(), max_length=1)
    report: PublicPressureReport
    public_replay_ready: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def result_graph_is_closed(self) -> Self:
        expected_planner_status = (
            "accepted"
            if all(item.source == "agent" for item in self.planner_trace)
            else "fallback"
            if all(item.source == "strong_workflow_fallback" for item in self.planner_trace)
            else "mixed"
        )
        if self.planner_status != expected_planner_status:
            raise ValueError("planner_status does not match planner trace")
        if self.selected_route_id != self.planner_trace[-1].selected_candidate_id:
            raise ValueError("selected route must equal the final Planner selection")
        if tuple(item.step for item in self.planner_trace) != tuple(
            range(1, len(self.planner_trace) + 1)
        ):
            raise ValueError("planner steps must be contiguous")
        if tuple(item.sequence for item in self.tool_trace) != tuple(
            range(1, len(self.tool_trace) + 1)
        ):
            raise ValueError("tool sequences must be contiguous")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        source_ids = tuple(dict.fromkeys(item.dataset_id for item in self.evidence))
        if self.report.evidence_ids != evidence_ids or self.report.source_ids != source_ids:
            raise ValueError("report provenance must exactly match result evidence")
        if any(
            evidence_id not in set(evidence_ids)
            for execution in self.tool_trace
            for evidence_id in execution.evidence_ids
        ):
            raise ValueError("tool trace references evidence outside the result")
        expected_status = (
            "unsupported"
            if self.report.conclusion_kind == "unsupported"
            else "completed"
            if self.report.conclusion_kind == "supported_relative_height"
            else "limited"
        )
        if self.execution_status != expected_status:
            raise ValueError("execution status does not match report conclusion")
        return self
