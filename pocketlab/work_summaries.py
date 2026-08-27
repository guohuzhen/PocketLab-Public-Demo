from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pocketlab.general_exploration_state import GeneralExperimentCase
from pocketlab.schemas import DiagnosticCaseSnapshot
from pocketlab.sensor_models import SensorKind


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkEvidenceReference(_StrictModel):
    evidence_id: str
    sensor: SensorKind
    source: str
    quality: Literal["low", "medium", "high"]


class WorkMeasurementCard(_StrictModel):
    task_id: str
    title: str
    instruction: str
    sensors: list[SensorKind] = Field(min_length=1, max_length=8)
    metric_keys: list[str] = Field(default_factory=list, max_length=8)
    controlled_variables: list[str] = Field(default_factory=list, max_length=16)
    source_options: list[str] = Field(default_factory=list, max_length=8)
    expected_revision: int | None = Field(default=None, ge=1)


class WorkReportEnvelope(_StrictModel):
    outcome: Literal["completed", "inconclusive"]
    headline: str
    confidence: Literal["low", "medium", "high"]
    mechanism_explanation: str
    evidence_ids: list[str] = Field(default_factory=list, max_length=256)
    boundaries: list[str] = Field(default_factory=list, max_length=24)
    next_actions: list[str] = Field(default_factory=list, max_length=16)


class WorkSummary(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["diagnostic", "general_exploration"]
    work_id: str
    title: str
    question: str
    status: Literal[
        "planning",
        "collecting",
        "awaiting_user_decision",
        "completed",
        "inconclusive",
    ]
    resumable: bool
    resume_path: str
    next_action: str
    revision_token: str
    sensors: list[SensorKind] = Field(default_factory=list, max_length=8)
    evidence_count: int = Field(ge=0, le=256)
    data_sources: list[str] = Field(default_factory=list, max_length=12)
    evidence: list[WorkEvidenceReference] = Field(default_factory=list, max_length=256)
    current_measurement: WorkMeasurementCard | None = None
    report: WorkReportEnvelope | None = None
    created_at: str
    updated_at: str


def diagnostic_work_summary(snapshot: DiagnosticCaseSnapshot) -> WorkSummary:
    case = snapshot.case
    status = {
        "planning": "planning",
        "collecting": "collecting",
        "awaiting_user_decision": "awaiting_user_decision",
        "completed_with_conclusion": "completed",
        "completed_inconclusive": "inconclusive",
    }[case.status]
    sensors = list(
        dict.fromkeys(
            [item.sensor for item in case.sensor_plan]
            + [item.sensor for item in case.evidence]
        )
    )
    evidence = [
        WorkEvidenceReference(
            evidence_id=item.evidence_id,
            sensor=item.sensor,
            source=_diagnostic_evidence_source(item.facts),
            quality=item.quality,
        )
        for item in case.evidence
    ]
    current_measurement = None
    if case.current_task is not None:
        task = case.current_task
        current_measurement = WorkMeasurementCard(
            task_id=task.task_id,
            title=task.title,
            instruction=task.instruction,
            sensors=[task.required_sensor],
            metric_keys=[task.target_metric_key] if task.target_metric_key else [],
            controlled_variables=list(task.controlled_variables),
            source_options=[
                "phyphox_live",
                "account_recording",
                "public_replay",
                "protocol_simulator",
            ],
        )
    report = _diagnostic_report(case)
    return WorkSummary(
        workflow="diagnostic",
        work_id=case.case_id,
        title=case.title,
        question=case.problem_statement,
        status=status,
        resumable=status in {"planning", "collecting", "awaiting_user_decision"},
        resume_path=f"/app/cases/{case.case_id}",
        next_action=_next_action(status, current_measurement is not None),
        revision_token=snapshot.updated_at,
        sensors=sensors,
        evidence_count=len(evidence),
        data_sources=list(dict.fromkeys(item.source for item in evidence)),
        evidence=evidence,
        current_measurement=current_measurement,
        report=report,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
    )


def general_work_summary(
    case: GeneralExperimentCase,
    *,
    created_at: str,
    updated_at: str,
) -> WorkSummary:
    status = {
        "collecting": "collecting",
        "awaiting_user_decision": "awaiting_user_decision",
        "completed_descriptive": "completed",
        "completed_inconclusive": "inconclusive",
    }[case.status]
    sensors = list(dict.fromkeys(item.sensor for item in case.protocol.sensors))
    evidence = [
        WorkEvidenceReference(
            evidence_id=item.evidence_id,
            sensor=item.sensor,
            source=item.lineage.source,
            quality=item.quality,
        )
        for item in case.evidence
    ]
    current_measurement = None
    if case.current_task is not None:
        task = case.current_task
        metric_keys = [
            item.metric_key
            for item in case.protocol.sensors
            if item.sensor in task.sensors
        ]
        current_measurement = WorkMeasurementCard(
            task_id=task.task_id,
            title=task.title,
            instruction=task.instruction,
            sensors=list(task.sensors),
            metric_keys=metric_keys,
            controlled_variables=list(case.protocol.controls),
            source_options=list(case.protocol.selected_sources),
            expected_revision=case.revision,
        )
    return WorkSummary(
        workflow="general_exploration",
        work_id=case.case_id,
        title=case.protocol.title,
        question=case.protocol.question,
        status=status,
        resumable=status in {"collecting", "awaiting_user_decision"},
        resume_path=f"/app/explore/general/runs/{case.case_id}",
        next_action=_next_action(status, current_measurement is not None),
        revision_token=f"revision:{case.revision}",
        sensors=sensors,
        evidence_count=len(evidence),
        data_sources=list(dict.fromkeys(item.source for item in evidence)),
        evidence=evidence,
        current_measurement=current_measurement,
        report=_general_report(case),
        created_at=created_at,
        updated_at=updated_at,
    )


def _diagnostic_evidence_source(facts: list[object]) -> str:
    sources = [str(getattr(item, "provenance_source", "")) for item in facts]
    sources = [item for item in sources if item]
    return "+".join(dict.fromkeys(sources)) if sources else "recording"


def _diagnostic_report(case) -> WorkReportEnvelope | None:
    report = case.final_report
    if report is None:
        return None
    next_actions = []
    if report.solution_plan is not None:
        next_actions.extend(item.title for item in report.solution_plan.actions)
        next_actions.extend(report.solution_plan.escalation_conditions)
    return WorkReportEnvelope(
        outcome=(
            "completed"
            if report.outcome == "completed_with_conclusion"
            else "inconclusive"
        ),
        headline=report.answer_headline or report.conclusion,
        confidence=report.confidence,
        mechanism_explanation=report.mechanism_explanation or report.conclusion,
        evidence_ids=report.source_fact_ids
        or [item.evidence_id for item in case.evidence],
        boundaries=[report.scope_boundary] if report.scope_boundary else [],
        next_actions=next_actions[:16],
    )


def _general_report(case: GeneralExperimentCase) -> WorkReportEnvelope | None:
    report = case.report
    if report is None:
        return None
    return WorkReportEnvelope(
        outcome=(
            "completed"
            if report.outcome == "completed_descriptive"
            else "inconclusive"
        ),
        headline=report.answer_headline or report.answer,
        confidence=report.confidence,
        mechanism_explanation=report.mechanism_explanation or report.answer,
        evidence_ids=list(report.evidence_ids),
        boundaries=list(report.claim_boundaries),
        next_actions=[],
    )


def _next_action(status: str, has_measurement: bool) -> str:
    if status == "planning":
        return "继续规划第一项实验"
    if status == "awaiting_user_decision":
        return "选择继续探求或按当前证据结束"
    if status == "collecting" and has_measurement:
        return "完成当前测量"
    if status == "collecting":
        return "刷新并恢复当前任务"
    return "查看最终报告"
