from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RealityFeedbackType = Literal[
    "hypothesis_not_applicable",
    "hypothesis_needs_correction",
    "task_not_feasible",
    "instruction_unclear",
    "environment_fact",
]
RealityEvidenceReuseReason = Literal[
    "compatible-planning-context",
    "sensitive-reuse-not-confirmed",
    "low-quality-or-invalid",
    "non-user-evidence-source",
    "missing-structured-facts",
]


class RealityEvidenceReuseCandidate(BaseModel):
    """A minimal, provider-free view used to audit old evidence for replanning."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    evidence_id: str = Field(min_length=1, max_length=80)
    sensor: str = Field(min_length=1, max_length=40)
    planning_summary: str = Field(default="", max_length=240)
    eligible: bool
    exclusion_reason_code: RealityEvidenceReuseReason | None = None

    @model_validator(mode="after")
    def candidate_has_closed_eligibility(self) -> Self:
        if self.eligible and (not self.planning_summary or self.exclusion_reason_code is not None):
            raise ValueError("eligible evidence reuse candidates require a summary and no blocker")
        if not self.eligible and self.exclusion_reason_code is None:
            raise ValueError("ineligible evidence reuse candidates require a blocker")
        return self


class RealityEvidenceReuseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    evidence_id: str = Field(min_length=1, max_length=80)
    sensor: str = Field(min_length=1, max_length=40)
    disposition: Literal["planning_context", "archived_only"]
    reason_code: RealityEvidenceReuseReason
    planning_summary: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def decision_does_not_leak_archived_content(self) -> Self:
        if self.disposition == "planning_context":
            if self.reason_code != "compatible-planning-context" or not self.planning_summary:
                raise ValueError("planning reuse requires an explicit compatible summary")
        elif self.planning_summary:
            raise ValueError("archived-only evidence cannot expose planning content")
        return self


class RealityEvidenceReuseAudit(BaseModel):
    """Server-owned receipt: old facts may guide a plan but never count as new evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: Literal["none", "planning_only"] = "none"
    planning_context_evidence_ids: tuple[str, ...] = Field(default=(), max_length=256)
    archived_only_evidence_ids: tuple[str, ...] = Field(default=(), max_length=256)
    decisions: tuple[RealityEvidenceReuseDecision, ...] = Field(default=(), max_length=256)
    counts_toward_new_conclusion: Literal[False] = False

    @model_validator(mode="after")
    def audit_partitions_all_decisions(self) -> Self:
        planning = tuple(
            item.evidence_id for item in self.decisions if item.disposition == "planning_context"
        )
        archived = tuple(
            item.evidence_id for item in self.decisions if item.disposition == "archived_only"
        )
        if planning != self.planning_context_evidence_ids:
            raise ValueError("planning reuse IDs must match the ordered decision receipt")
        if archived != self.archived_only_evidence_ids:
            raise ValueError("archived reuse IDs must match the ordered decision receipt")
        all_ids = (*planning, *archived)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("evidence reuse decisions must be unique")
        if self.scope != ("planning_only" if planning else "none"):
            raise ValueError("evidence reuse scope must match accepted planning context")
        return self

    def planning_prompt(self) -> str:
        summaries = [
            item.planning_summary
            for item in self.decisions
            if item.disposition == "planning_context"
        ]
        if not summaries:
            return ""
        return (
            "经过服务端筛选的旧测量事实（只用于重新规划、避免无意义重复；"
            "不计入新案例证据或终止结论）：\n- "
            + "\n- ".join(summaries)
        )


def build_reality_evidence_reuse_audit(
    candidates: tuple[RealityEvidenceReuseCandidate, ...],
    *,
    confirm_sensitive_sensor_reuse: bool,
) -> RealityEvidenceReuseAudit:
    sensitive = {"microphone", "location"}
    decisions: list[RealityEvidenceReuseDecision] = []
    for candidate in candidates:
        if not candidate.eligible:
            disposition = "archived_only"
            reason = candidate.exclusion_reason_code
            summary = ""
        elif candidate.sensor in sensitive and not confirm_sensitive_sensor_reuse:
            disposition = "archived_only"
            reason = "sensitive-reuse-not-confirmed"
            summary = ""
        else:
            disposition = "planning_context"
            reason = "compatible-planning-context"
            summary = candidate.planning_summary
        decisions.append(
            RealityEvidenceReuseDecision(
                evidence_id=candidate.evidence_id,
                sensor=candidate.sensor,
                disposition=disposition,
                reason_code=reason,
                planning_summary=summary,
            )
        )
    planning_ids = tuple(
        item.evidence_id for item in decisions if item.disposition == "planning_context"
    )
    archived_ids = tuple(
        item.evidence_id for item in decisions if item.disposition == "archived_only"
    )
    return RealityEvidenceReuseAudit(
        scope="planning_only" if planning_ids else "none",
        planning_context_evidence_ids=planning_ids,
        archived_only_evidence_ids=archived_ids,
        decisions=tuple(decisions),
    )


class RealityFeedbackRequest(BaseModel):
    """A plain-language correction to assumptions about the user's real setting."""

    model_config = ConfigDict(extra="forbid")

    feedback_type: RealityFeedbackType
    message: str = Field(min_length=3, max_length=800)
    hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    expected_task_id: str | None = Field(default=None, max_length=80)
    expected_revision: int | None = Field(default=None, ge=1)
    confirm_sensitive_sensor_reuse: bool = False

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        return " ".join(value.strip().split())

    @field_validator("hypothesis_ids", mode="before")
    @classmethod
    def normalize_hypothesis_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def feedback_targets_are_closed(self) -> Self:
        if len(self.hypothesis_ids) != len(set(self.hypothesis_ids)):
            raise ValueError("反馈不能重复引用同一候选解释。")
        if (
            self.feedback_type
            in {"hypothesis_not_applicable", "hypothesis_needs_correction"}
            and not self.hypothesis_ids
        ):
            raise ValueError("请选择至少一个不符合现场实际的候选解释。")
        return self


class RealityFeedbackRecord(BaseModel):
    """Auditable lineage for one user-triggered plan revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feedback_id: str = Field(pattern=r"^feedback-[0-9a-f]{16}$")
    feedback_type: RealityFeedbackType
    message: str = Field(min_length=3, max_length=800)
    hypothesis_ids: tuple[str, ...] = Field(default=(), max_length=4)
    source_case_id: str = Field(min_length=1, max_length=80)
    source_task_id: str | None = Field(default=None, max_length=80)
    preserved_evidence_ids: tuple[str, ...] = Field(default=(), max_length=256)
    evidence_reuse: RealityEvidenceReuseAudit = Field(
        default_factory=RealityEvidenceReuseAudit
    )
    created_at: str = Field(min_length=10, max_length=64)

    @model_validator(mode="after")
    def reuse_receipt_only_references_preserved_evidence(self) -> Self:
        decision_ids = {item.evidence_id for item in self.evidence_reuse.decisions}
        if decision_ids and decision_ids != set(self.preserved_evidence_ids):
            raise ValueError("evidence reuse receipt must cover every preserved evidence ID")
        return self


def revised_context(
    *,
    original_context: str,
    feedback: RealityFeedbackRequest,
    rejected_hypotheses: tuple[str, ...],
    task_title: str | None,
    limit: int,
    evidence_reuse: RealityEvidenceReuseAudit | None = None,
) -> str:
    """Place the user's reality above model assumptions without overflowing schemas."""

    correction = [f"用户现场事实：{feedback.message}"]
    if rejected_hypotheses and feedback.feedback_type == "hypothesis_not_applicable":
        correction.append("用户明确指出以下旧假设整体不适用：" + "；".join(rejected_hypotheses))
    elif rejected_hypotheses and feedback.feedback_type == "hypothesis_needs_correction":
        correction.append(
            "用户指出以下旧假设只有部分内容不符合现场；必须按现场事实修正，"
            "只能保留不冲突的部分：" + "；".join(rejected_hypotheses)
        )
    if task_title and feedback.feedback_type in {"task_not_feasible", "instruction_unclear"}:
        correction.append(f"受影响的旧任务：{task_title}")
    correction.append(
        "重新规划时必须服从这条现场事实，不得恢复已排除的设备、结构或操作前提。"
    )
    if evidence_reuse is not None and evidence_reuse.planning_prompt():
        correction.append(evidence_reuse.planning_prompt())
    correction_text = "\n".join(correction)
    if not original_context.strip():
        return correction_text[:limit]
    original_prefix = "原先提供的其他现场信息："
    available = max(0, limit - len(correction_text) - len(original_prefix) - 2)
    return (
        f"{correction_text}\n{original_prefix}\n{original_context.strip()[:available]}"
    )[:limit]
