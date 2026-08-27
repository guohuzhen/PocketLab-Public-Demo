from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pocketlab.public_replay_dataset import PublicDataClass
from pocketlab.sensor_models import SensorAnalysis

Identifier = str
PublicLightOperation = Literal["select_initial_evidence", "select_follow_up"]
PublicLightRationale = Literal[
    "match_temporal_perturbation_goal",
    "match_registered_condition_comparison",
    "match_naturalistic_context_goal",
    "add_phone_transfer_crosscheck",
    "request_missing_live_evidence",
    "minimal_sufficient_evidence",
    "unsupported_claim_boundary",
    "privacy_not_acknowledged",
    "strong_workflow_fallback",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not value or len(value) > 100:
        raise ValueError(f"{field_name} must contain 1-100 characters")
    if not value[0].isalnum() or any(
        not (character.isascii() and (character.isalnum() or character in "._:-"))
        for character in value
    ):
        raise ValueError(f"{field_name} must be a machine-safe ASCII identifier")
    return value


def _validate_evidence_ref(value: str) -> str:
    if not value or len(value) > 200:
        raise ValueError("evidence_refs must contain 1-200 characters")
    parts = value.split("/")
    if len(parts) not in {1, 2}:
        raise ValueError("evidence_refs must be dataset or dataset/recording")
    for part in parts:
        _validate_identifier(part, field_name="evidence_refs")
    return value


class PublicLightExploreRequest(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    research_question: str = Field(min_length=5, max_length=800)
    privacy_acknowledged: bool = False
    query_illuminance_lx: float | None = Field(
        default=None,
        ge=0.0,
        le=1_000_000_000.0,
    )

    @field_validator("research_question")
    @classmethod
    def question_has_no_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 and character not in "\n\t" for character in value):
            raise ValueError("research_question contains unsupported control characters")
        return value


class PublicLightFact(_StrictModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    value: float
    unit: str = Field(min_length=1, max_length=32)


class PublicLightPlanCandidate(_StrictModel):
    candidate_id: Identifier
    title: str = Field(min_length=3, max_length=160)
    server_reason: str = Field(min_length=10, max_length=360)
    tool_ids: list[Identifier] = Field(default_factory=list, max_length=4)
    evidence_refs: list[Identifier] = Field(default_factory=list, max_length=8)
    requires_privacy_acknowledgement: bool = False
    terminal: bool = False
    result_code: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)

    @field_validator("candidate_id")
    @classmethod
    def candidate_id_is_safe(cls, value: str) -> str:
        return _validate_identifier(value, field_name="candidate_id")

    @field_validator("tool_ids")
    @classmethod
    def tool_ids_are_safe_and_unique(cls, values: list[str]) -> list[str]:
        checked = [
            _validate_identifier(value, field_name="tool_ids")
            for value in values
        ]
        if len(checked) != len(set(checked)):
            raise ValueError("candidate tool_ids must be unique")
        return checked

    @field_validator("evidence_refs")
    @classmethod
    def evidence_refs_are_safe_and_unique(cls, values: list[str]) -> list[str]:
        checked = [_validate_evidence_ref(value) for value in values]
        if len(checked) != len(set(checked)):
            raise ValueError("candidate evidence_refs must be unique")
        return checked

    @model_validator(mode="after")
    def terminal_candidates_have_no_tools(self) -> Self:
        if self.terminal and self.tool_ids:
            raise ValueError("terminal candidates cannot request tools")
        if not self.terminal and not self.tool_ids:
            raise ValueError("non-terminal candidates require an allowlisted tool")
        return self


class PublicLightEvidenceView(_StrictModel):
    evidence_ids: list[Identifier] = Field(default_factory=list, max_length=8)
    result_codes: list[str] = Field(default_factory=list, max_length=16)
    facts: list[PublicLightFact] = Field(default_factory=list, max_length=24)
    limitations: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_safe_and_unique(cls, values: list[str]) -> list[str]:
        checked = [
            _validate_identifier(value, field_name="evidence_ids") for value in values
        ]
        if len(checked) != len(set(checked)):
            raise ValueError("evidence_ids must be unique")
        return checked


class PublicLightPlannerRequest(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    operation: PublicLightOperation
    run_id: Identifier
    step: int = Field(ge=1, le=2)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_question_untrusted: str = Field(min_length=5, max_length=800)
    privacy_acknowledged: bool
    evidence_view: PublicLightEvidenceView
    candidates: list[PublicLightPlanCandidate] = Field(min_length=2, max_length=4)
    fallback_candidate_id: Identifier

    @field_validator("run_id", "fallback_candidate_id")
    @classmethod
    def identifiers_are_safe(cls, value: str, info: object) -> str:
        return _validate_identifier(value, field_name=getattr(info, "field_name", "identifier"))

    @model_validator(mode="after")
    def candidate_graph_is_closed(self) -> Self:
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("planner candidates must be unique")
        if self.fallback_candidate_id not in candidate_ids:
            raise ValueError("fallback_candidate_id must reference a candidate")
        if any(
            item.requires_privacy_acknowledgement and not self.privacy_acknowledged
            for item in self.candidates
        ):
            raise ValueError("privacy-gated candidates require explicit acknowledgement")
        return self


class PublicLightPlannerDecision(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: Identifier
    step: int = Field(ge=1, le=2)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: Identifier
    rationale_code: PublicLightRationale

    @field_validator("run_id", "selected_candidate_id")
    @classmethod
    def identifiers_are_safe(cls, value: str, info: object) -> str:
        return _validate_identifier(value, field_name=getattr(info, "field_name", "identifier"))


class PublicLightRuntimeToolEvent(_StrictModel):
    name: Identifier
    status: Literal["called", "returned", "committed", "error"]

    @field_validator("name")
    @classmethod
    def name_is_safe(cls, value: str) -> str:
        return _validate_identifier(value, field_name="tool_event.name")


class PublicLightRuntimeTrace(_StrictModel):
    run_id: Identifier | None = None
    operation: Identifier | None = None
    model: str | None = Field(default=None, max_length=160)
    status: Literal["completed", "failed", "cancelled"] | None = None
    elapsed_s: float | None = Field(default=None, ge=0.0, le=120.0)
    timeout_s: float | None = Field(default=None, gt=0.0, le=30.0)
    max_turns: int | None = Field(default=None, ge=1, le=3)
    retry_limit: int | None = Field(default=None, ge=0, le=1)
    model_requests: int | None = Field(default=None, ge=0, le=4)
    tool_calls: int | None = Field(default=None, ge=0, le=2)
    tool_events: list[PublicLightRuntimeToolEvent] = Field(
        default_factory=list,
        max_length=4,
    )
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_reported: bool | None = None
    token_budget: int | None = Field(default=None, ge=1, le=4_000)
    token_budget_exceeded: bool | None = None
    error_kind: Identifier | None = None
    error_type: Identifier | None = None
    transport: Literal["function_tool", "validated_json_text"] | None = None
    transport_fallback_reason: Identifier | None = None

    @field_validator(
        "run_id",
        "operation",
        "error_kind",
        "error_type",
        "transport_fallback_reason",
    )
    @classmethod
    def optional_ids_are_safe(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return _validate_identifier(value, field_name=getattr(info, "field_name", "identifier"))


class PublicLightPlannerTrace(_StrictModel):
    step: int = Field(ge=1, le=2)
    operation: PublicLightOperation
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ids: list[Identifier] = Field(min_length=2, max_length=4)
    selected_candidate_id: Identifier
    fallback_candidate_id: Identifier
    rationale_code: PublicLightRationale
    source: Literal["agent", "strong_workflow_fallback"]
    outcome: Literal["accepted", "fallback"]
    fallback_reason: str | None = Field(default=None, max_length=120)
    transport: Literal[
        "not_attempted", "function_tool", "validated_json_text"
    ] = "not_attempted"
    runtime_trace: PublicLightRuntimeTrace | None = None

    @field_validator(
        "candidate_ids",
    )
    @classmethod
    def candidate_ids_are_safe(cls, values: list[str]) -> list[str]:
        return [
            _validate_identifier(value, field_name="candidate_ids") for value in values
        ]

    @field_validator("selected_candidate_id", "fallback_candidate_id")
    @classmethod
    def selected_ids_are_safe(cls, value: str, info: object) -> str:
        return _validate_identifier(value, field_name=getattr(info, "field_name", "identifier"))

    @model_validator(mode="after")
    def selected_ids_and_fallback_are_consistent(self) -> Self:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("planner trace candidate_ids must be unique")
        if self.selected_candidate_id not in self.candidate_ids:
            raise ValueError("selected candidate must be present in trace")
        if self.fallback_candidate_id not in self.candidate_ids:
            raise ValueError("fallback candidate must be present in trace")
        if self.source == "strong_workflow_fallback":
            if self.outcome != "fallback" or not self.fallback_reason:
                raise ValueError("workflow fallback trace requires an explicit reason")
            if self.selected_candidate_id != self.fallback_candidate_id:
                raise ValueError("workflow fallback must select the frozen fallback candidate")
        return self


class PublicLightToolExecution(_StrictModel):
    sequence: int = Field(ge=1, le=3)
    tool_id: Identifier
    status: Literal["completed", "failed", "timeout", "rejected"] = "completed"
    evidence_ids: list[Identifier] = Field(default_factory=list, max_length=8)
    result_codes: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("tool_id")
    @classmethod
    def tool_id_is_safe(cls, value: str) -> str:
        return _validate_identifier(value, field_name="tool_id")

    @field_validator("evidence_ids")
    @classmethod
    def tool_evidence_ids_are_safe_and_unique(cls, values: list[str]) -> list[str]:
        checked = [
            _validate_identifier(value, field_name="evidence_ids") for value in values
        ]
        if len(checked) != len(set(checked)):
            raise ValueError("tool evidence_ids must be unique")
        return checked


class PublicLightEvidenceSnapshot(_StrictModel):
    evidence_id: Identifier
    dataset_id: Identifier
    recording_ids: list[Identifier] = Field(default_factory=list, max_length=66)
    data_class: PublicDataClass
    device_scope: str = Field(min_length=3, max_length=240)
    source_title: str = Field(min_length=3, max_length=240)
    source_url: str = Field(min_length=12, max_length=500)
    doi: str | None = Field(default=None, max_length=160)
    license_spdx: str = Field(min_length=2, max_length=40)
    analyses: list[SensorAnalysis] = Field(default_factory=list, max_length=2)
    facts: list[PublicLightFact] = Field(default_factory=list, max_length=32)
    processing_disclosures: list[str] = Field(min_length=1, max_length=24)
    claim_boundary: list[str] = Field(min_length=1, max_length=24)
    gate_c_eligible: Literal[False] = False

    @field_validator("evidence_id", "dataset_id")
    @classmethod
    def identifiers_are_safe(cls, value: str, info: object) -> str:
        return _validate_identifier(value, field_name=getattr(info, "field_name", "identifier"))

    @field_validator("recording_ids")
    @classmethod
    def recording_ids_are_safe_and_unique(cls, values: list[str]) -> list[str]:
        checked = [
            _validate_identifier(value, field_name="recording_ids") for value in values
        ]
        if len(checked) != len(set(checked)):
            raise ValueError("recording_ids must be unique")
        return checked


class PublicLightFinding(_StrictModel):
    finding_id: Identifier
    text: str = Field(min_length=5, max_length=600)
    evidence_ids: list[Identifier] = Field(min_length=1, max_length=3)

    @field_validator("finding_id")
    @classmethod
    def finding_id_is_safe(cls, value: str) -> str:
        return _validate_identifier(value, field_name="finding_id")

    @field_validator("evidence_ids")
    @classmethod
    def finding_evidence_ids_are_safe_and_unique(cls, values: list[str]) -> list[str]:
        checked = [
            _validate_identifier(value, field_name="finding.evidence_ids")
            for value in values
        ]
        if len(checked) != len(set(checked)):
            raise ValueError("finding evidence_ids must be unique")
        return checked


class PublicLightReportSource(_StrictModel):
    dataset_id: Identifier
    data_class: PublicDataClass
    device_scope: str = Field(min_length=3, max_length=240)
    source_title: str = Field(min_length=3, max_length=240)
    source_url: str = Field(min_length=12, max_length=500)
    doi: str | None = Field(default=None, max_length=160)
    license_spdx: str = Field(min_length=2, max_length=40)
    gate_c_eligible: Literal[False] = False

    @field_validator("dataset_id")
    @classmethod
    def dataset_id_is_safe(cls, value: str) -> str:
        return _validate_identifier(value, field_name="dataset_id")


class PublicLightReport(_StrictModel):
    conclusion_kind: Literal[
        "supported_descriptive",
        "limited",
        "unsupported",
        "live_measurement_required",
        "privacy_acknowledgement_required",
    ]
    title: str = Field(min_length=3, max_length=160)
    summary: str = Field(min_length=10, max_length=1200)
    supported_findings: list[PublicLightFinding] = Field(default_factory=list, max_length=16)
    uncertainties: list[str] = Field(min_length=1, max_length=16)
    forbidden_claims: list[str] = Field(min_length=1, max_length=16)
    next_live_measurement: str | None = Field(default=None, max_length=1000)
    evidence_ids: list[Identifier] = Field(default_factory=list, max_length=8)
    source_ids: list[Identifier] = Field(default_factory=list, max_length=4)
    sources: list[PublicLightReportSource] = Field(default_factory=list, max_length=4)
    gate_c_credited_records: Literal[0] = 0
    gate_e_status: Literal["not_evaluated"] = "not_evaluated"
    gate_h_status: Literal["not_evaluated"] = "not_evaluated"
    public_replay_ready: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @field_validator("evidence_ids", "source_ids")
    @classmethod
    def report_ids_are_safe_and_unique(cls, values: list[str], info: object) -> list[str]:
        field_name = getattr(info, "field_name", "identifier")
        checked = [_validate_identifier(value, field_name=field_name) for value in values]
        if len(checked) != len(set(checked)):
            raise ValueError(f"{field_name} must be unique")
        return checked

    @model_validator(mode="after")
    def report_source_ids_are_closed(self) -> Self:
        if [item.dataset_id for item in self.sources] != self.source_ids:
            raise ValueError("report sources must exactly match source_ids in order")
        finding_evidence = {
            evidence_id
            for finding in self.supported_findings
            for evidence_id in finding.evidence_ids
        }
        if not finding_evidence.issubset(set(self.evidence_ids)):
            raise ValueError("report findings reference evidence outside the report")
        return self


class PublicLightExploreResult(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_id: Literal["light-public-exploration.v1"] = "light-public-exploration.v1"
    protocol_version: Literal["1.0.0"] = "1.0.0"
    run_id: Identifier
    research_question: str = Field(min_length=5, max_length=800)
    execution_status: Literal["completed", "limited", "unsupported"]
    selected_route_id: Identifier
    planner_status: Literal["accepted", "fallback", "mixed"]
    planner_trace: list[PublicLightPlannerTrace] = Field(min_length=1, max_length=2)
    tool_trace: list[PublicLightToolExecution] = Field(default_factory=list, max_length=3)
    evidence: list[PublicLightEvidenceSnapshot] = Field(default_factory=list, max_length=3)
    report: PublicLightReport
    public_replay_ready: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @field_validator("run_id", "selected_route_id")
    @classmethod
    def result_ids_are_safe(cls, value: str, info: object) -> str:
        return _validate_identifier(value, field_name=getattr(info, "field_name", "identifier"))

    @model_validator(mode="after")
    def graph_and_readiness_are_consistent(self) -> Self:
        expected_planner_status = (
            "accepted"
            if all(item.source == "agent" for item in self.planner_trace)
            else "fallback"
            if all(
                item.source == "strong_workflow_fallback"
                for item in self.planner_trace
            )
            else "mixed"
        )
        if self.planner_status != expected_planner_status:
            raise ValueError("planner_status does not match planner_trace")
        if self.selected_route_id != self.planner_trace[-1].selected_candidate_id:
            raise ValueError("selected_route_id must equal the final planner selection")
        if [item.step for item in self.planner_trace] != list(
            range(1, len(self.planner_trace) + 1)
        ):
            raise ValueError("planner trace steps must be contiguous")
        if [item.sequence for item in self.tool_trace] != list(
            range(1, len(self.tool_trace) + 1)
        ):
            raise ValueError("tool trace sequences must be contiguous")
        evidence_ids = [item.evidence_id for item in self.evidence]
        source_ids = list(dict.fromkeys(item.dataset_id for item in self.evidence))
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
            if self.report.conclusion_kind == "supported_descriptive"
            else "limited"
        )
        if self.execution_status != expected_status:
            raise ValueError("execution_status does not match report conclusion")
        return self
