from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pocketlab.auth import get_current_user_id
from pocketlab.experiment_protocols import get_experiment_protocol
from pocketlab.experiment_tools import (
    LightConditionAggregate,
    LightDecayFit,
    LightMeasurement,
    aggregate_light_conditions,
    fit_light_distance_decay,
    sample_light_fit_series,
)
from pocketlab.investigation_models import (
    ExperimentEvidence,
    ExperimentParameterConstraint,
    ExperimentParameterValue,
    ExperimentProgress,
    ExperimentReport,
    ExperimentTask,
    InvestigationCase,
    InvestigationCaseCreate,
    InvestigationCaseHistoryItem,
    InvestigationMeasurementSubmit,
    LightPlannerCandidate,
    LightPlannerDecision,
    LightPlannerRequest,
    MetricSnapshot,
    PlannerDecisionTrace,
    RecordingRef,
    SensorAnalysisSnapshot,
    ToolExecution,
    VisualizationArtifact,
    VisualizationAxis,
    VisualizationPoint,
    VisualizationSeries,
)
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.store import SessionStore, StoredSensorRecording, session_store

_PROTOCOL_ID = "light-distance-law.v1"
_PROTOCOL_VERSION = "1.0.0"
_DISTANCE_KEY = "distance_m"
_DISTANCE_UNIT = "m"
_DISTANCE_TOLERANCE_RATIO = 0.05
_REPEATABILITY_LIMIT = 0.15
_TARGET_CONDITION_COUNT = 4
_TARGET_REPEATS = 2

_PLANNER_REASON_TEXT = {
    "maximize_log_span": "Agent 在安全候选中选择更大的对数距离跨度，以提高幂律拟合辨识度。",
    "preserve_signal_to_background": "Agent 选择较保守的距离增量，以避免净照度过早接近环境光背景。",
    "reduce_saturation_risk": "Agent 选择更大的距离增量，以降低近距离高照度平台风险。",
    "respect_user_constraint": "Agent 根据用户描述的现场空间约束，从服务端候选中选择可执行测点。",
    "prefer_protocol_default": "Agent 确认采用预注册协议的默认下一距离。",
}


@dataclass(frozen=True)
class PreparedInvestigationTransition:
    """A fully validated measurement transition that has not mutated storage yet."""

    case_id: str
    expected_revision: int
    provisional_case: InvestigationCase
    planner_request: LightPlannerRequest | None = None
    candidate_tasks: tuple[tuple[str, ExperimentTask], ...] = ()


class InvestigationNotFound(KeyError):
    pass


class InvestigationConflict(ValueError):
    pass


class InvestigationValidation(ValueError):
    pass


class InvestigationStore:
    """Persist and execute allowlisted experiment protocols for one active user."""

    def __init__(
        self,
        storage: SQLiteDatabase | None = None,
        recordings: SessionStore | None = None,
        *,
        user_id: str | None = DEFAULT_USER_ID,
    ) -> None:
        self._database = storage or SQLiteDatabase(":memory:")
        self._recordings = recordings or SessionStore(self._database, user_id=user_id)
        self._user_id = user_id

    @property
    def _active_user_id(self) -> str:
        return self._user_id or get_current_user_id()

    def create(self, request: InvestigationCaseCreate) -> InvestigationCase:
        if request.mode != "explore":
            raise InvestigationValidation("首版可执行协议只支持 explore 模式。")
        if request.protocol_id is None or request.protocol_version is None:
            raise InvestigationValidation("必须显式选择已验证的协议 ID 和版本。")
        try:
            protocol = get_experiment_protocol(request.protocol_id, request.protocol_version)
        except KeyError as exc:
            raise InvestigationValidation(str(exc)) from exc
        if (protocol.protocol_id, protocol.protocol_version) != (
            _PROTOCOL_ID,
            _PROTOCOL_VERSION,
        ):
            raise InvestigationValidation("该协议尚未接入可执行状态机。")

        if request.parameter_values:
            raise InvestigationValidation("首版协议从预注册的 0.5 m 条件开始，不接受起始距离覆盖。")
        protocol_parameters = {item.key: item for item in protocol.parameters}
        for constraint in request.execution_constraints:
            definition = protocol_parameters.get(constraint.key)
            if definition is None:
                raise InvestigationValidation("执行约束引用了协议之外的参数。")
            try:
                constraint.validate_definition(definition)
            except ValueError as exc:
                raise InvestigationValidation(str(exc)) from exc
        distance_constraint = next(
            (
                item
                for item in request.execution_constraints
                if item.key == _DISTANCE_KEY
            ),
            None,
        )
        if distance_constraint is not None and not distance_constraint.allows(0.5):
            raise InvestigationValidation("现场距离约束必须允许预注册的 0.5 m 起始测点。")

        case = InvestigationCase(
            case_id=f"inv-{uuid4().hex[:12]}",
            revision=1,
            title=request.title,
            research_question=request.research_question,
            context=request.context,
            mode="explore",
            status="collecting",
            plan_source="validated_protocol",
            planning_policy=request.planning_policy,
            protocol=protocol,
            execution_constraints=request.execution_constraints,
            current_task=self._background_task(sequence=1, ending=False),
            progress=self._progress([], corrections=0),
        )
        now = utc_now()
        self._database.execute(
            """
            INSERT INTO investigation_cases(
                investigation_id, user_id, revision, case_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                case.case_id,
                self._active_user_id,
                case.revision,
                case.model_dump_json(),
                now,
                now,
            ),
        )
        return case

    def get(self, case_id: str) -> InvestigationCase:
        row = self._database.fetch_one(
            """
            SELECT revision, case_json FROM investigation_cases
            WHERE investigation_id = ? AND user_id = ?
            """,
            (case_id, self._active_user_id),
        )
        if row is None:
            raise InvestigationNotFound(f"Unknown investigation: {case_id}")
        case = InvestigationCase.model_validate_json(row["case_json"])
        if case.revision != int(row["revision"]):
            raise RuntimeError("investigation revision column and JSON payload diverged")
        return case

    def list(self, *, limit: int = 100) -> list[InvestigationCaseHistoryItem]:
        rows = self._database.fetch_all(
            """
            SELECT revision, case_json, created_at, updated_at
            FROM investigation_cases
            WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?
            """,
            (self._active_user_id, limit),
        )
        result: list[InvestigationCaseHistoryItem] = []
        for row in rows:
            case = InvestigationCase.model_validate_json(row["case_json"])
            result.append(
                InvestigationCaseHistoryItem(
                    case_id=case.case_id,
                    revision=int(row["revision"]),
                    title=case.title,
                    mode=case.mode,
                    status=case.status,
                    primary_sensor=case.protocol.primary_sensor,
                    current_task_title=(case.current_task.title if case.current_task else None),
                    evidence_count=len(case.evidence),
                    artifact_count=len(case.artifacts),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return result

    def delete(self, case_id: str) -> None:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM investigation_cases WHERE investigation_id = ? AND user_id = ?",
                (case_id, self._active_user_id),
            )
        if cursor.rowcount != 1:
            raise InvestigationNotFound(f"Unknown investigation: {case_id}")

    def recording_is_referenced(self, recording_id: str) -> bool:
        rows = self._database.fetch_all(
            "SELECT case_json FROM investigation_cases WHERE user_id = ?",
            (self._active_user_id,),
        )
        for row in rows:
            case = InvestigationCase.model_validate_json(row["case_json"])
            if any(item.recording.recording_id == recording_id for item in case.evidence):
                return True
        return False

    def validate_capture_request(
        self,
        case_id: str,
        *,
        expected_revision: int,
        task_id: str,
        parameters: list[ExperimentParameterValue],
        controls_confirmed: bool,
    ) -> InvestigationCase:
        """Validate mutable state before a remote phone capture is started."""

        case = self.get(case_id)
        if expected_revision != case.revision:
            raise InvestigationConflict(
                f"版本冲突：当前 revision={case.revision}，请刷新后重试。"
            )
        if case.current_task is None or case.status != "collecting":
            raise InvestigationConflict("该实验已经结束，不能继续采集。")
        if task_id != case.current_task.task_id:
            raise InvestigationConflict("任务已变化，请刷新后按当前任务采集。")
        if not controls_confirmed:
            raise InvestigationValidation("采集前必须确认本任务的控制条件。")
        self._validate_parameters(case.current_task, parameters)
        return case

    def prepare_measurement_transition(
        self,
        case_id: str,
        request: InvestigationMeasurementSubmit,
    ) -> PreparedInvestigationTransition:
        case = self.get(case_id)
        if request.expected_revision != case.revision:
            raise InvestigationConflict(
                f"版本冲突：当前 revision={case.revision}，请刷新后重试。"
            )
        if case.current_task is None or case.status != "collecting":
            raise InvestigationConflict("该实验已经结束，不能继续绑定测量。")
        task = case.current_task
        if request.task_id != task.task_id:
            raise InvestigationConflict("任务已变化，请刷新后按当前任务提交。")
        if not request.controls_confirmed:
            raise InvestigationValidation("提交前必须确认本任务的控制条件。")
        if any(
            evidence.recording.recording_id == request.recording.recording_id
            for evidence in case.evidence
        ):
            raise InvestigationConflict("同一条测量记录不能重复绑定。")

        try:
            recording = self._recordings.get_sensor_recording(request.recording.recording_id)
        except KeyError as exc:
            raise InvestigationValidation("找不到当前用户的 v2 传感器记录。") from exc
        self._validate_recording_reference(request.recording, recording, task)
        parameters = self._validate_parameters(task, request.parameters)

        evidence, inspection = self._build_evidence(
            task,
            recording,
            parameters,
            request.observation_notes,
            sequence=len(case.evidence) + 1,
        )
        if evidence.valid and task.condition_id.startswith("condition-"):
            start_background = self._background_value(case.evidence, ending=False)
            observed = _snapshot_metric(evidence, "median_illuminance_lx")
            background_iqr = _snapshot_metric(case.evidence[0], "illuminance_iqr_lx")
            minimum_net_signal = max(1.0, start_background * 0.05, background_iqr * 3.0)
            if observed - start_background <= minimum_net_signal:
                payload = evidence.model_dump(mode="python")
                payload.update(
                    valid=False,
                    rejection_reasons=["净照度没有超过背景噪声工程门槛。"],
                )
                evidence = ExperimentEvidence.model_validate(payload)
        completed_task = task.model_copy(
            update={"status": "completed" if evidence.valid else "rejected"}
        )
        completed_tasks = [*case.completed_tasks, completed_task]
        evidence_items = [*case.evidence, evidence]
        tool_trace = [*case.tool_trace, inspection]
        corrections = case.progress.corrections_used

        next_task: ExperimentTask | None
        status = "collecting"
        report = None
        artifacts: list[VisualizationArtifact] = []
        terminal_reason: str | None = None
        terminal_reason_code = "protocol-termination"
        planner_request: LightPlannerRequest | None = None
        candidate_tasks: tuple[tuple[str, ExperimentTask], ...] = ()

        if not evidence.valid:
            if (
                corrections >= case.protocol.max_corrections
                or len(evidence_items) >= case.protocol.max_measurements
            ):
                next_task = None
                status = "completed_inconclusive"
                terminal_reason = "纠偏或测量预算已用尽，仍未获得可用证据。"
                terminal_reason_code = "correction-budget-exhausted"
            else:
                corrections += 1
                next_task = self._correction_task(
                    task,
                    evidence,
                    sequence=len(completed_tasks) + 1,
                )
                if task.condition_id.startswith("condition-"):
                    tool_trace.append(
                        self._design_execution(
                            task=completed_task,
                            evidence=evidence,
                            next_distance=self._target_distance(next_task),
                            sequence=len(tool_trace) + 1,
                            reason_code=2.0,
                        )
                    )
        elif task.condition_id == "background-start":
            next_task = self._condition_task(
                sequence=len(completed_tasks) + 1,
                condition_number=1,
                distance=0.5,
                role="condition",
                selection_reason_code="background-ready",
                selection_reason=(
                    "起始环境光已记录；先在 0.5 m 建立第一个光源距离条件。"
                ),
                selection_evidence_ids=[evidence.evidence_id],
            )
        elif task.condition_id.startswith("condition-"):
            group = self._valid_condition_evidence(evidence_items, task.condition_id)
            if len(group) < _TARGET_REPEATS:
                next_task = self._condition_task(
                    sequence=len(completed_tasks) + 1,
                    condition_number=int(task.condition_id.rsplit("-", 1)[1]),
                    distance=self._evidence_distance(evidence),
                    role="replication",
                    selection_reason_code="replication-required",
                    selection_reason="同一距离至少需要两条有效记录才能检查重复性。",
                    selection_evidence_ids=[evidence.evidence_id],
                )
            else:
                aggregate, aggregate_execution = self._aggregate_group(
                    evidence_items,
                    task.condition_id,
                    completed_task,
                    sequence=len(tool_trace) + 1,
                )
                tool_trace.append(aggregate_execution)
                relative_mad = (
                    aggregate.mad_net_illuminance_lx
                    / aggregate.median_net_illuminance_lx
                )
                if relative_mad > _REPEATABILITY_LIMIT and len(group) == 2:
                    if (
                        corrections >= case.protocol.max_corrections
                        or len(evidence_items) >= case.protocol.max_measurements
                    ):
                        next_task = None
                        status = "completed_inconclusive"
                        terminal_reason = "重复测量离散度过大，且纠偏预算已用尽。"
                        terminal_reason_code = "repeatability-budget-exhausted"
                    else:
                        corrections += 1
                        next_task = self._condition_task(
                            sequence=len(completed_tasks) + 1,
                            condition_number=int(task.condition_id.rsplit("-", 1)[1]),
                            distance=aggregate.distance_m,
                            role="correction",
                            selection_reason_code="repeatability-correction",
                            selection_reason=(
                                "同一距离的重复离散度超过 15%，保持距离和控制变量后复测。"
                            ),
                            selection_evidence_ids=list(aggregate.evidence_ids),
                        )
                elif relative_mad > _REPEATABILITY_LIMIT:
                    next_task = None
                    status = "completed_inconclusive"
                    terminal_reason = "第三次复测后离散度仍超过 15%，证据不足。"
                    terminal_reason_code = "repeatability-failed"
                else:
                    condition_number = int(task.condition_id.rsplit("-", 1)[1])
                    if condition_number < _TARGET_CONDITION_COUNT:
                        planner_request, candidate_tasks = self._prepare_design_choice(
                            case=case,
                            evidence=evidence_items,
                            aggregate=aggregate,
                            completed_task=completed_task,
                            condition_number=condition_number,
                            next_sequence=len(completed_tasks) + 1,
                        )
                        if planner_request is not None:
                            next_task = dict(candidate_tasks)[
                                planner_request.fallback_candidate_id
                            ]
                        elif candidate_tasks:
                            next_task = candidate_tasks[0][1]
                            constrained = self._distance_constraint(case) is not None
                            task_payload = next_task.model_dump(mode="python")
                            task_payload.update(
                                selection_source="deterministic",
                                selection_reason_code=(
                                    "only-feasible-design-point"
                                    if constrained
                                    else "only-safe-design-point"
                                ),
                                selection_reason=(
                                    "服务端执行约束过滤后只剩一个可执行距离，跳过模型选择。"
                                    if constrained
                                    else "服务端安全候选生成后只剩一个距离，直接确定性执行。"
                                ),
                            )
                            next_task = ExperimentTask.model_validate(task_payload)
                            tool_trace.append(
                                self._design_execution(
                                    task=completed_task,
                                    evidence=evidence,
                                    next_distance=self._target_distance(next_task),
                                    sequence=len(tool_trace) + 1,
                                    reason_code=6.0 if constrained else 1.0,
                                )
                            )
                        else:
                            next_task = None
                            status = "completed_inconclusive"
                            terminal_reason = (
                                "服务端约束过滤后没有可执行的下一距离，实验已安全停止。"
                            )
                            terminal_reason_code = "no-feasible-design-point"
                    else:
                        next_task = self._background_task(
                            sequence=len(completed_tasks) + 1,
                            ending=True,
                            evidence_ids=list(aggregate.evidence_ids),
                        )
        else:
            drift_problem = self._background_drift_problem(evidence_items)
            if drift_problem:
                next_task = None
                status = "completed_inconclusive"
                terminal_reason = drift_problem
                terminal_reason_code = "background-drift"
            else:
                fit, final_tools = self._fit(evidence_items, completed_task, tool_trace)
                tool_trace.extend(final_tools)
                artifact = self._artifact(case, evidence_items, tool_trace, fit)
                artifacts = [artifact]
                next_task = None
                if fit.classification == "inconclusive":
                    status = "completed_inconclusive"
                    terminal_reason = "拟合未同时满足首版预注册判据。"
                    terminal_reason_code = "fit-inconclusive"
                else:
                    status = "completed_with_conclusion"
                report = self._report(
                    case,
                    evidence_items,
                    tool_trace,
                    artifacts,
                    fit,
                    status,
                    terminal_reason,
                    terminal_reason_code,
                )

        if status == "completed_inconclusive" and report is None:
            report = self._inconclusive_report(
                case,
                evidence_items,
                tool_trace,
                terminal_reason or "证据不足。",
                reason_code=terminal_reason_code,
            )

        progress = self._progress(
            evidence_items,
            corrections=corrections,
            terminal_status=status,
            blocker=terminal_reason,
        )
        updated = InvestigationCase(
            case_id=case.case_id,
            revision=case.revision + 1,
            title=case.title,
            research_question=case.research_question,
            context=case.context,
            mode=case.mode,
            status=status,
            plan_source=case.plan_source,
            planning_policy=case.planning_policy,
            protocol=case.protocol,
            execution_constraints=case.execution_constraints,
            current_task=next_task,
            completed_tasks=completed_tasks,
            evidence=evidence_items,
            tool_trace=tool_trace,
            planner_trace=case.planner_trace,
            artifacts=artifacts,
            progress=progress,
            report=report,
        )
        return PreparedInvestigationTransition(
            case_id=case.case_id,
            expected_revision=case.revision,
            provisional_case=updated,
            planner_request=planner_request,
            candidate_tasks=candidate_tasks,
        )

    def submit_measurement(
        self,
        case_id: str,
        request: InvestigationMeasurementSubmit,
    ) -> InvestigationCase:
        """Backward-compatible deterministic entry point used by offline tests and tools."""

        prepared = self.prepare_measurement_transition(case_id, request)
        fallback_reason = (
            "planner-not-invoked"
            if prepared.planner_request is not None
            and prepared.provisional_case.planning_policy == "bounded_agent"
            else None
        )
        return self.commit_prepared_transition(
            prepared,
            fallback_reason=fallback_reason,
        )

    def commit_prepared_transition(
        self,
        prepared: PreparedInvestigationTransition,
        *,
        decision: LightPlannerDecision | None = None,
        runtime_trace: dict[str, Any] | None = None,
        fallback_reason: str | None = None,
    ) -> InvestigationCase:
        """Commit one prepared transition with one final revision CAS."""

        updated = prepared.provisional_case
        self._revalidate_prepared_transition(prepared)
        request = prepared.planner_request
        if request is None:
            if decision is not None or fallback_reason is not None:
                raise InvestigationValidation("当前状态没有 Agent 设计点可提交。")
            self._save(updated, expected_revision=prepared.expected_revision)
            return updated

        candidates = dict(prepared.candidate_tasks)
        if set(candidates) != {item.candidate_id for item in request.candidates}:
            raise RuntimeError("prepared planner candidates diverged from request")

        use_agent = decision is not None and fallback_reason is None
        if use_agent:
            self._validate_planner_decision(request, decision)
            selected_id = decision.selected_candidate_id
            rationale_code = decision.rationale_code
            selection_source = "agent"
        else:
            selected_id = request.fallback_candidate_id
            rationale_code = "prefer_protocol_default"
            selection_source = (
                "fallback"
                if updated.planning_policy == "bounded_agent"
                else "deterministic"
            )

        try:
            selected_task = candidates[selected_id]
        except KeyError as exc:  # pragma: no cover - guarded by strict request/decision validation
            raise InvestigationValidation("Agent 选择了未注册候选。") from exc
        task_payload = selected_task.model_dump(mode="python")
        task_payload.update(
            selection_source=selection_source,
            selection_reason_code=rationale_code.replace("_", "-"),
            selection_reason=_PLANNER_REASON_TEXT[rationale_code],
            selection_evidence_ids=list(request.input_evidence_ids),
        )
        selected_task = ExperimentTask.model_validate(task_payload)

        source_task = next(
            item
            for item in updated.completed_tasks
            if item.task_id == request.completed_task_id
        )
        source_evidence = next(
            item
            for item in reversed(updated.evidence)
            if item.evidence_id in request.input_evidence_ids
        )
        trace = [
            *updated.tool_trace,
            self._design_execution(
                task=source_task,
                evidence=source_evidence,
                input_evidence_ids=list(request.input_evidence_ids),
                next_distance=self._target_distance(selected_task),
                sequence=len(updated.tool_trace) + 1,
                reason_code=self._planner_reason_number(rationale_code),
            ),
        ]
        planner_trace = list(updated.planner_trace)
        plan_source = updated.plan_source
        if updated.planning_policy == "bounded_agent":
            safe_runtime = runtime_trace or {}
            source = "agent" if use_agent else "deterministic_fallback"
            if use_agent:
                plan_source = "agent_allowlisted"
            planner_trace.append(
                PlannerDecisionTrace(
                    decision_id=f"decision-{uuid4().hex[:12]}",
                    sequence=len(planner_trace) + 1,
                    source_task_id=source_task.task_id,
                    planned_task_id=selected_task.task_id,
                    request_sha256=request.request_sha256,
                    candidate_ids=[item.candidate_id for item in request.candidates],
                    selected_candidate_id=selected_id,
                    fallback_candidate_id=request.fallback_candidate_id,
                    rationale_code=rationale_code,
                    source=source,
                    outcome="accepted" if use_agent else "fallback",
                    fallback_reason=None if use_agent else (fallback_reason or "planner-unavailable"),
                    transport=safe_runtime.get("transport", "not_attempted"),
                    transport_fallback_reason=safe_runtime.get(
                        "transport_fallback_reason"
                    ),
                    input_evidence_ids=list(request.input_evidence_ids),
                    revision_before=prepared.expected_revision,
                    revision_after=updated.revision,
                    run_id=safe_runtime.get("run_id"),
                    model=safe_runtime.get("model"),
                    attempts=len(safe_runtime.get("attempts", [])),
                    model_requests=int(safe_runtime.get("model_requests", 0)),
                    tool_calls=int(safe_runtime.get("tool_calls", 0)),
                    elapsed_s=float(safe_runtime.get("elapsed_s", 0.0)),
                    input_tokens=safe_runtime.get("input_tokens"),
                    output_tokens=safe_runtime.get("output_tokens"),
                    total_tokens=safe_runtime.get("total_tokens"),
                    token_budget_exceeded=bool(
                        safe_runtime.get("token_budget_exceeded", False)
                    ),
                )
            )

        payload = updated.model_dump(mode="python")
        payload.update(
            plan_source=plan_source,
            current_task=selected_task,
            tool_trace=trace,
            planner_trace=planner_trace,
        )
        updated = InvestigationCase.model_validate(payload)
        self._save(updated, expected_revision=prepared.expected_revision)
        return updated

    def _revalidate_prepared_transition(
        self,
        prepared: PreparedInvestigationTransition,
    ) -> None:
        """Recheck mutable state and the new recording after an Agent wait."""

        current = self.get(prepared.case_id)
        updated = prepared.provisional_case
        if current.revision != prepared.expected_revision:
            raise InvestigationConflict(
                f"版本冲突：当前 revision={current.revision}，请刷新后重试。"
            )
        if (
            updated.case_id != current.case_id
            or updated.revision != current.revision + 1
            or len(updated.evidence) != len(current.evidence) + 1
            or len(updated.completed_tasks) != len(current.completed_tasks) + 1
        ):
            raise InvestigationValidation("预备提交与当前实验状态不一致。")
        if current.current_task is None or current.status != "collecting":
            raise InvestigationConflict("该实验已经结束，不能提交预备测量。")

        completed_task = updated.completed_tasks[-1]
        new_evidence = updated.evidence[-1]
        if (
            completed_task.task_id != current.current_task.task_id
            or new_evidence.task_id != completed_task.task_id
            or any(
                item.recording.recording_id == new_evidence.recording.recording_id
                for item in current.evidence
            )
        ):
            raise InvestigationValidation("预备提交的任务或证据引用已经失效。")
        try:
            actual = self._recordings.get_sensor_recording(
                new_evidence.recording.recording_id
            )
        except KeyError as exc:
            raise InvestigationValidation(
                "预备提交引用的测量记录已不存在或不属于当前用户。"
            ) from exc
        self._validate_recording_reference(
            new_evidence.recording,
            actual,
            completed_task,
        )

    @staticmethod
    def _validate_planner_decision(
        request: LightPlannerRequest,
        decision: LightPlannerDecision,
    ) -> None:
        if (
            decision.case_id != request.case_id
            or decision.expected_revision != request.expected_revision
            or decision.completed_task_id != request.completed_task_id
            or decision.request_sha256 != request.request_sha256
        ):
            raise InvestigationValidation("Agent 决策与当前实验版本或任务不匹配。")
        if decision.selected_candidate_id not in {
            item.candidate_id for item in request.candidates
        }:
            raise InvestigationValidation("Agent 决策引用了候选集外的测点。")
        selected = next(
            item
            for item in request.candidates
            if item.candidate_id == decision.selected_candidate_id
        )
        minimum = min(item.distance_m for item in request.candidates)
        maximum = max(item.distance_m for item in request.candidates)
        if (
            decision.rationale_code == "prefer_protocol_default"
            and decision.selected_candidate_id != request.fallback_candidate_id
        ):
            raise InvestigationValidation("Agent 的默认协议理由与所选候选不一致。")
        if (
            decision.rationale_code == "preserve_signal_to_background"
            and selected.distance_m != minimum
        ):
            raise InvestigationValidation("Agent 的保留净信号理由必须选择最近安全候选。")
        if (
            decision.rationale_code in {"maximize_log_span", "reduce_saturation_risk"}
            and selected.distance_m != maximum
        ):
            raise InvestigationValidation("Agent 的扩大距离理由必须选择最远安全候选。")
        if decision.rationale_code == "respect_user_constraint":
            distance_constraint = next(
                (
                    item
                    for item in request.execution_constraints
                    if item.key == _DISTANCE_KEY
                ),
                None,
            )
            if distance_constraint is None or selected.distance_m != maximum:
                raise InvestigationValidation(
                    "Agent 的现场约束理由必须引用服务端可信距离约束并选择最远可执行候选。"
                )

    @staticmethod
    def _planner_reason_number(reason: str) -> float:
        return float(
            {
                "prefer_protocol_default": 1,
                "maximize_log_span": 3,
                "preserve_signal_to_background": 4,
                "reduce_saturation_risk": 5,
                "respect_user_constraint": 6,
            }[reason]
        )

    def _save(self, case: InvestigationCase, *, expected_revision: int) -> None:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE investigation_cases
                SET revision = ?, case_json = ?, updated_at = ?
                WHERE investigation_id = ? AND user_id = ? AND revision = ?
                """,
                (
                    case.revision,
                    case.model_dump_json(),
                    utc_now(),
                    case.case_id,
                    self._active_user_id,
                    expected_revision,
                ),
            )
        if cursor.rowcount != 1:
            raise InvestigationConflict("实验已被其他请求更新，请刷新后重试。")

    @staticmethod
    def _validate_recording_reference(
        claimed: RecordingRef,
        actual: StoredSensorRecording,
        task: ExperimentTask,
    ) -> None:
        provenance = actual.upload.provenance
        expected = RecordingRef(
            recording_type="sensor_v2",
            recording_id=actual.session_id,
            sensor=actual.upload.sensor,
            analyzer_id=actual.analysis.analyzer_id,
            analyzer_version=actual.analysis.analyzer_version,
            source=provenance.source,
            config_sha256=provenance.config_sha256,
            remote_session=provenance.remote_session or None,
        )
        if claimed != expected:
            raise InvestigationValidation("记录元数据与服务端真实记录不一致。")
        if expected.source == "public_replay":
            raise InvestigationValidation(
                "当前距离协议不接受公开环境光回放；该数据缺少本任务的距离、背景与重复条件。"
            )
        if expected.sensor != task.sensor or expected.analyzer_id != task.analyzer_id:
            raise InvestigationValidation("传感器或分析器与当前任务不匹配。")

    @staticmethod
    def _validate_parameters(
        task: ExperimentTask,
        supplied: list[ExperimentParameterValue],
    ) -> list[ExperimentParameterValue]:
        definitions = {item.key: item for item in task.parameter_definitions}
        values = {item.key: item for item in supplied}
        if set(values) != set(definitions):
            raise InvestigationValidation("参数必须与当前任务要求完全一致。")
        for key, definition in definitions.items():
            try:
                definition.validate_target(values[key])
            except ValueError as exc:
                raise InvestigationValidation(str(exc)) from exc
        targets = {item.key: item for item in task.parameter_targets}
        if _DISTANCE_KEY in targets:
            measured = float(values[_DISTANCE_KEY].value)
            target = float(targets[_DISTANCE_KEY].value)
            if abs(measured - target) / target > _DISTANCE_TOLERANCE_RATIO:
                raise InvestigationValidation("实际距离与任务目标相差超过 5%。")
        return list(supplied)

    @staticmethod
    def _build_evidence(
        task: ExperimentTask,
        recording: StoredSensorRecording,
        parameters: list[ExperimentParameterValue],
        observation_notes: str,
        *,
        sequence: int,
    ) -> tuple[ExperimentEvidence, ToolExecution]:
        analysis = recording.analysis
        snapshot = SensorAnalysisSnapshot.from_sensor_analysis(analysis)
        reasons: list[str] = []
        if analysis.confidence == "low":
            reasons.append("分析器质量门禁为 low。")
        plateau = _metric_value(recording, "upper_plateau_fraction")
        median = _metric_value(recording, "median_illuminance_lx")
        if plateau >= 0.5 and median > 1_000_000.0:
            reasons.append("高照度记录的上限平台比例达到 50%，疑似饱和。")
        evidence = ExperimentEvidence(
            evidence_id=f"evidence-{uuid4().hex[:12]}",
            task_id=task.task_id,
            condition_id=task.condition_id,
            recording=RecordingRef(
                recording_type="sensor_v2",
                recording_id=recording.session_id,
                sensor=recording.upload.sensor,
                analyzer_id=analysis.analyzer_id,
                analyzer_version=analysis.analyzer_version,
                source=recording.upload.provenance.source,
                config_sha256=recording.upload.provenance.config_sha256,
                remote_session=recording.upload.provenance.remote_session or None,
            ),
            role=task.role,
            parameters=parameters,
            quality=analysis.confidence,
            analysis=snapshot,
            observation_notes=observation_notes,
            valid=not reasons,
            rejection_reasons=reasons,
        )
        execution = ToolExecution(
            execution_id=f"exec-{uuid4().hex[:12]}",
            sequence=sequence,
            task_id=task.task_id,
            tool_id="sensor_analysis.light.v2",
            tool_version="2.0.0",
            input_evidence_ids=[evidence.evidence_id],
            status="succeeded",
            result_metrics=snapshot.metrics,
        )
        return evidence, execution

    def _correction_task(
        self,
        task: ExperimentTask,
        evidence: ExperimentEvidence,
        *,
        sequence: int,
    ) -> ExperimentTask:
        if not task.condition_id.startswith("condition-"):
            return self._task_copy(task, sequence=sequence, role="correction")
        target = self._evidence_distance(evidence)
        if any("上限平台" in reason for reason in evidence.rejection_reasons):
            target = min(4.0, target * 1.5)
            reason_code = "possible-saturation-correction"
            reason = "高照度记录疑似出现上限平台；增大距离后重新测量。"
        else:
            target = max(0.1, target / 1.5)
            reason_code = "low-net-signal-correction"
            reason = "净照度没有稳定超过背景门槛；缩短距离后重新测量。"
        number = int(task.condition_id.rsplit("-", 1)[1])
        return self._condition_task(
            sequence=sequence,
            condition_number=number,
            distance=target,
            role="correction",
            selection_reason_code=reason_code,
            selection_reason=reason,
            selection_evidence_ids=[evidence.evidence_id],
        )

    @staticmethod
    def _task_copy(task: ExperimentTask, *, sequence: int, role: str) -> ExperimentTask:
        payload = task.model_dump(mode="python")
        payload.update(
            task_id=f"task-{sequence}",
            sequence=sequence,
            role=role,
            title=f"纠偏复测：{task.title}",
            selection_source="deterministic",
            selection_reason_code="quality-correction",
            selection_reason="当前证据未通过质量门禁，保持条件后重新测量。",
            status="in_progress",
        )
        return ExperimentTask.model_validate(payload)

    @staticmethod
    def _background_task(
        *,
        sequence: int,
        ending: bool,
        evidence_ids: list[str] | None = None,
    ) -> ExperimentTask:
        suffix = "end" if ending else "start"
        return ExperimentTask(
            task_id=f"task-{sequence}",
            sequence=sequence,
            title="结束环境光对照" if ending else "建立环境光基线",
            role="background",
            instruction=(
                "关闭或完全遮挡待测光源，保持手机位置和环境不变，采集稳定环境光。"
            ),
            sensor="light",
            analyzer_id="pocketlab.light.v2",
            recommended_phyphox_experiment="Light（光照度）",
            condition_id=f"background-{suffix}",
            controls=["待测光源关闭或被完全遮挡。", "手机位置和朝向保持不变。"],
            tool_ids=(
                [
                    "sensor_analysis.light.v2",
                    "fit_light_distance_decay",
                    "sample_light_fit_series",
                ]
                if ending
                else ["sensor_analysis.light.v2"]
            ),
            selection_source="protocol" if not ending else "deterministic",
            selection_reason_code=(
                "ending-background-check" if ending else "initial-background"
            ),
            selection_reason=(
                "距离条件已经完成；再次测量环境光以检查实验期间的背景漂移。"
                if ending
                else "先测量环境光基线，后续距离条件必须扣除同一背景。"
            ),
            selection_evidence_ids=evidence_ids or [],
            status="in_progress",
        )

    @staticmethod
    def _condition_task(
        *,
        sequence: int,
        condition_number: int,
        distance: float,
        role: str,
        selection_reason_code: str = "protocol-step",
        selection_reason: str = "由已验证实验协议生成。",
        selection_evidence_ids: list[str] | None = None,
        selection_source: str = "deterministic",
    ) -> ExperimentTask:
        protocol = get_experiment_protocol(_PROTOCOL_ID, _PROTOCOL_VERSION)
        definition = protocol.parameters[0]
        distance = float(round(distance, 4))
        return ExperimentTask(
            task_id=f"task-{sequence}",
            sequence=sequence,
            title=f"距离条件 {condition_number}：{distance:g} m",
            role=role,
            instruction=(
                f"将传感器受光面置于距光源参考点 {distance:g} m，保持朝向一致，"
                "等读数稳定后采集。"
            ),
            sensor="light",
            analyzer_id="pocketlab.light.v2",
            recommended_phyphox_experiment="Light（光照度）",
            condition_id=f"condition-{condition_number}",
            parameter_definitions=[definition],
            parameter_targets=[
                ExperimentParameterValue(key=_DISTANCE_KEY, value=distance, unit=_DISTANCE_UNIT)
            ],
            controls=protocol.controls,
            tool_ids=[
                "sensor_analysis.light.v2",
                "aggregate_light_conditions",
                "select_next_design_point",
            ],
            selection_source=selection_source,
            selection_reason_code=selection_reason_code,
            selection_reason=selection_reason,
            selection_evidence_ids=selection_evidence_ids or [],
            status="in_progress",
        )

    @staticmethod
    def _valid_condition_evidence(
        evidence: Iterable[ExperimentEvidence], condition_id: str
    ) -> list[ExperimentEvidence]:
        return [
            item
            for item in evidence
            if item.valid
            and item.task_id
            and item.parameters
            and item.role in {"condition", "replication", "correction"}
            and item.condition_id == condition_id
        ]

    def _aggregate_group(
        self,
        evidence: list[ExperimentEvidence],
        condition_id: str,
        task: ExperimentTask,
        *,
        sequence: int,
    ) -> tuple[LightConditionAggregate, ToolExecution]:
        background = self._background_value(evidence, ending=False)
        members = [
            item
            for item in evidence
            if item.valid
            and item.parameters
            and item.condition_id == condition_id
        ]
        measurements = [self._light_measurement(item, condition_id) for item in members]
        aggregate = aggregate_light_conditions(background, measurements, min_repeats=2)[0]
        execution = ToolExecution(
            execution_id=f"exec-{uuid4().hex[:12]}",
            sequence=sequence,
            task_id=task.task_id,
            tool_id="aggregate_light_conditions",
            tool_version="1.0.0",
            input_evidence_ids=list(aggregate.evidence_ids),
            status="succeeded",
            result_metrics=[
                MetricSnapshot(key="distance_m", label="条件距离", value=aggregate.distance_m, unit="m"),
                MetricSnapshot(
                    key="median_net_illuminance_lx",
                    label="净照度中位数",
                    value=aggregate.median_net_illuminance_lx,
                    unit="lx",
                ),
                MetricSnapshot(
                    key="mad_net_illuminance_lx",
                    label="净照度中位绝对偏差",
                    value=aggregate.mad_net_illuminance_lx,
                    unit="lx",
                ),
                MetricSnapshot(
                    key="repeat_count",
                    label="重复次数",
                    value=float(aggregate.repeat_count),
                    unit="count",
                ),
            ],
        )
        return aggregate, execution

    def _prepare_design_choice(
        self,
        *,
        case: InvestigationCase,
        evidence: list[ExperimentEvidence],
        aggregate: LightConditionAggregate,
        completed_task: ExperimentTask,
        condition_number: int,
        next_sequence: int,
    ) -> tuple[
        LightPlannerRequest | None,
        tuple[tuple[str, ExperimentTask], ...],
    ]:
        """Freeze safe next-distance candidates without giving the model raw control."""

        minimum_distance = min(
            self._evidence_distance(item)
            for item in evidence
            if item.valid and item.parameters
        )
        default_distance = min(4.0, aggregate.distance_m * 2.0)
        candidate_rows: list[tuple[LightPlannerCandidate, ExperimentTask]] = []
        seen_distances: set[float] = set()
        factor_specs = (
            (
                1.5,
                "较小步长优先保留净信号相对背景的余量。",
                ["preserve-signal"],
            ),
            (
                2.0,
                "采用预注册协议的默认倍增距离。",
                ["protocol-default"],
            ),
            (
                2.5,
                "较大步长优先扩大对数距离跨度。",
                ["maximize-span"],
            ),
        )
        for factor, server_reason, risk_codes in factor_specs:
            target = float(round(min(4.0, aggregate.distance_m * factor), 4))
            if target <= aggregate.distance_m * 1.05 or target in seen_distances:
                continue
            if not self._distance_is_allowed(case, target):
                continue
            projected_span = target / minimum_distance
            if condition_number + 1 == _TARGET_CONDITION_COUNT and projected_span < 3.0:
                continue
            seen_distances.add(target)
            candidate_id = f"distance-{round(target * 1000)}mm"
            candidate = LightPlannerCandidate(
                candidate_id=candidate_id,
                distance_m=target,
                projected_span_ratio=projected_span,
                risk_codes=risk_codes,
                server_reason=server_reason,
            )
            task = self._condition_task(
                sequence=next_sequence,
                condition_number=condition_number + 1,
                distance=target,
                role="condition",
                selection_reason_code="distance-span-expansion",
                selection_reason="当前距离重复性已通过，下一步扩展距离跨度。",
                selection_evidence_ids=list(aggregate.evidence_ids),
            )
            candidate_rows.append((candidate, task))

        if len(candidate_rows) < 2:
            return None, tuple(
                (candidate.candidate_id, task) for candidate, task in candidate_rows
            )
        fallback = min(
            candidate_rows,
            key=lambda item: abs(item[0].distance_m - default_distance),
        )[0]
        group = [
            item
            for item in evidence
            if item.valid and item.condition_id == completed_task.condition_id
        ]
        background = self._background_value(evidence, ending=False)
        plateau = max(_snapshot_metric(item, "upper_plateau_fraction") for item in group)
        signal_ratio = min(
            1_000_000_000.0,
            aggregate.median_net_illuminance_lx / max(background, 1.0),
        )
        request_payload: dict[str, Any] = {
            "schema_version": "1.0",
            "operation": "select_next_design_point",
            "case_id": case.case_id,
            "expected_revision": case.revision,
            "completed_task_id": completed_task.task_id,
            "protocol_id": case.protocol.protocol_id,
            "protocol_version": case.protocol.protocol_version,
            "research_question": case.research_question,
            "condition_number": condition_number,
            "background_lx": background,
            "latest_distance_m": aggregate.distance_m,
            "median_net_illuminance_lx": aggregate.median_net_illuminance_lx,
            "repeatability_ratio": (
                aggregate.mad_net_illuminance_lx
                / aggregate.median_net_illuminance_lx
            ),
            "upper_plateau_fraction": plateau,
            "signal_to_background_ratio": signal_ratio,
            "context_untrusted": case.context,
            "observation_notes_untrusted": "\n".join(
                item.observation_notes for item in group if item.observation_notes
            )[:1600],
            "execution_constraints": [
                item.model_dump(mode="json") for item in case.execution_constraints
            ],
            "input_evidence_ids": list(aggregate.evidence_ids),
            "candidates": [item.model_dump(mode="json") for item, _ in candidate_rows],
            "fallback_candidate_id": fallback.candidate_id,
        }
        request_hash = hashlib.sha256(
            json.dumps(
                request_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        request = LightPlannerRequest.model_validate(
            {**request_payload, "request_sha256": request_hash}
        )
        return request, tuple(
            (candidate.candidate_id, task) for candidate, task in candidate_rows
        )

    @staticmethod
    def _distance_constraint(
        case: InvestigationCase,
    ) -> ExperimentParameterConstraint | None:
        return next(
            (
                item
                for item in case.execution_constraints
                if item.key == _DISTANCE_KEY and item.unit == _DISTANCE_UNIT
            ),
            None,
        )

    @classmethod
    def _distance_is_allowed(cls, case: InvestigationCase, distance_m: float) -> bool:
        constraint = cls._distance_constraint(case)
        return constraint is None or constraint.allows(distance_m)

    def _fit(
        self,
        evidence: list[ExperimentEvidence],
        task: ExperimentTask,
        existing_trace: list[ToolExecution],
    ) -> tuple[LightDecayFit, list[ToolExecution]]:
        background = (
            self._background_value(evidence, ending=False)
            + self._background_value(evidence, ending=True)
        ) / 2.0
        measurements = [
            self._light_measurement(item, item.condition_id)
            for item in evidence
            if item.valid and item.parameters
        ]
        aggregates = aggregate_light_conditions(background, measurements, min_repeats=2)
        fit = fit_light_distance_decay(aggregates)
        all_ids = [item.evidence_id for item in evidence if item.valid]
        fit_execution = ToolExecution(
            execution_id=f"exec-{uuid4().hex[:12]}",
            sequence=len(existing_trace) + 1,
            task_id=task.task_id,
            tool_id="fit_light_distance_decay",
            tool_version="1.0.0",
            input_evidence_ids=all_ids,
            status="succeeded",
            result_metrics=self._fit_metrics(fit),
        )
        sample_execution = ToolExecution(
            execution_id=f"exec-{uuid4().hex[:12]}",
            sequence=len(existing_trace) + 2,
            task_id=task.task_id,
            tool_id="sample_light_fit_series",
            tool_version="1.0.0",
            input_evidence_ids=all_ids,
            status="succeeded",
            result_metrics=[
                MetricSnapshot(
                    key="sample_point_count",
                    label="可视化采样点数",
                    value=48.0,
                    unit="count",
                )
            ],
        )
        return fit, [fit_execution, sample_execution]

    def _artifact(
        self,
        case: InvestigationCase,
        evidence: list[ExperimentEvidence],
        trace: list[ToolExecution],
        fit: LightDecayFit,
    ) -> VisualizationArtifact:
        background = (
            self._background_value(evidence, ending=False)
            + self._background_value(evidence, ending=True)
        ) / 2.0
        condition_measurements = [
            self._light_measurement(item, item.condition_id)
            for item in evidence
            if item.valid and item.parameters
        ]
        aggregates = aggregate_light_conditions(background, condition_measurements, 2)
        valid_ids = [item.evidence_id for item in evidence if item.valid]
        sampled = sample_light_fit_series(fit, 48)
        return VisualizationArtifact(
            artifact_id=f"artifact-{uuid4().hex[:12]}",
            kind="scatter_with_fit",
            title="净照度—距离幂律拟合",
            x_axis=VisualizationAxis(field_key="distance_m", label="距离", unit="m", scale="log"),
            y_axis=VisualizationAxis(
                field_key="net_illuminance_lx",
                label="扣除背景后的照度",
                unit="lx",
                scale="log",
            ),
            series=[
                VisualizationSeries(
                    series_id="observations",
                    label="重复测量聚合",
                    series_type="observations",
                    points=[
                        VisualizationPoint(
                            x=item.distance_m,
                            y=item.median_net_illuminance_lx,
                            y_error=item.mad_net_illuminance_lx,
                            evidence_ids=list(item.evidence_ids),
                        )
                        for item in aggregates
                    ],
                ),
                VisualizationSeries(
                    series_id="free-fit",
                    label="自由幂律拟合",
                    series_type="fit",
                    points=[
                        VisualizationPoint(
                            x=item["distance_m"],
                            y=item["free_model_net_illuminance_lx"],
                            evidence_ids=valid_ids,
                        )
                        for item in sampled
                    ],
                ),
                VisualizationSeries(
                    series_id="inverse-square-reference",
                    label="n=2 参考模型",
                    series_type="reference",
                    points=[
                        VisualizationPoint(
                            x=item["distance_m"],
                            y=item["inverse_square_net_illuminance_lx"],
                            evidence_ids=valid_ids,
                        )
                        for item in sampled
                    ],
                ),
            ],
            source_evidence_ids=valid_ids,
            source_tool_execution_ids=[
                item.execution_id
                for item in trace
                if item.tool_id in {"fit_light_distance_decay", "sample_light_fit_series"}
            ],
            warnings=["误差棒为条件内中位绝对偏差，不是仪器校准不确定度。"],
            claim_boundaries=case.protocol.claim_boundaries,
        )

    @staticmethod
    def _fit_metrics(fit: LightDecayFit) -> list[MetricSnapshot]:
        return [
            MetricSnapshot(
                key="condition_count",
                label="有效距离条件数",
                value=float(fit.condition_count),
                unit="count",
            ),
            MetricSnapshot(
                key="minimum_distance_m",
                label="最小距离",
                value=fit.minimum_distance_m,
                unit="m",
            ),
            MetricSnapshot(
                key="maximum_distance_m",
                label="最大距离",
                value=fit.maximum_distance_m,
                unit="m",
            ),
            MetricSnapshot(
                key="distance_span_ratio",
                label="距离跨度",
                value=fit.distance_span_ratio,
                unit="ratio",
            ),
            MetricSnapshot(key="free_exponent", label="自由幂指数", value=fit.free_exponent, unit="1"),
            MetricSnapshot(
                key="free_exponent_ci95_low",
                label="幂指数区间下界",
                value=fit.free_exponent_ci95_low,
                unit="1",
            ),
            MetricSnapshot(
                key="free_exponent_ci95_high",
                label="幂指数区间上界",
                value=fit.free_exponent_ci95_high,
                unit="1",
            ),
            MetricSnapshot(
                key="free_model_r_squared",
                label="自由模型 R²",
                value=fit.free_model_r_squared,
                unit="ratio",
            ),
            MetricSnapshot(
                key="inverse_square_relative_rmse",
                label="n=2 模型 RMS 相对误差",
                value=fit.inverse_square_relative_rmse,
                unit="ratio",
            ),
        ]

    @staticmethod
    def _report(
        case: InvestigationCase,
        evidence: list[ExperimentEvidence],
        trace: list[ToolExecution],
        artifacts: list[VisualizationArtifact],
        fit: LightDecayFit,
        status: str,
        terminal_reason: str | None,
        terminal_reason_code: str,
    ) -> ExperimentReport:
        if fit.classification == "consistent_with_inverse_square":
            conclusion = (
                f"在本次已测试的 {fit.minimum_distance_m:g}–{fit.maximum_distance_m:g} m "
                f"范围内，净照度与距离平方反比近似一致：自由幂指数 n={fit.free_exponent:.3f}，"
                f"95% 区间 [{fit.free_exponent_ci95_low:.3f}, {fit.free_exponent_ci95_high:.3f}]，"
                f"R²={fit.free_model_r_squared:.3f}。"
            )
        elif fit.classification == "not_supported_in_tested_range":
            conclusion = (
                f"在本次已测试范围内，数据不支持 n=2 近似：自由幂指数 n={fit.free_exponent:.3f}，"
                f"95% 区间 [{fit.free_exponent_ci95_low:.3f}, {fit.free_exponent_ci95_high:.3f}]，"
                "这只描述当前光源、几何和手机条件。"
            )
        else:
            conclusion = "本次数据可拟合，但没有达到预注册的确认或否定判据，结论不充分。"
        valid = [item for item in evidence if item.valid]
        confidence = "high" if valid and all(item.quality == "high" for item in valid) else "medium"
        return ExperimentReport(
            outcome=status,
            confidence=confidence,
            conclusion=conclusion,
            evidence_ids=[item.evidence_id for item in valid],
            tool_execution_ids=[item.execution_id for item in trace],
            artifact_ids=[item.artifact_id for item in artifacts],
            summary_metrics=InvestigationStore._fit_metrics(fit),
            remaining_uncertainties=[
                "手机光线传感器没有可追溯校准链。",
                "有限尺寸光源、近场、反射和入射角仍可能影响关系。",
            ],
            claim_boundaries=case.protocol.claim_boundaries,
            stop_reason_code=terminal_reason_code,
            stop_reason=terminal_reason or "已满足重复数、距离跨度、质量和拟合终止判据。",
            market_validated=False,
        )

    @staticmethod
    def _inconclusive_report(
        case: InvestigationCase,
        evidence: list[ExperimentEvidence],
        trace: list[ToolExecution],
        reason: str,
        *,
        reason_code: str,
    ) -> ExperimentReport:
        return ExperimentReport(
            outcome="completed_inconclusive",
            confidence="low",
            conclusion=f"实验已安全停止，但当前证据不足以回答研究问题：{reason}",
            evidence_ids=[item.evidence_id for item in evidence if item.valid],
            tool_execution_ids=[item.execution_id for item in trace],
            remaining_uncertainties=[reason],
            claim_boundaries=case.protocol.claim_boundaries,
            stop_reason_code=reason_code,
            stop_reason=reason,
            market_validated=False,
        )

    @staticmethod
    def _progress(
        evidence: list[ExperimentEvidence],
        corrections: int,
        terminal_status: str = "collecting",
        blocker: str | None = None,
    ) -> ExperimentProgress:
        valid = [item for item in evidence if item.valid]
        conditions = {item.condition_id for item in valid if item.parameters}
        if terminal_status == "completed_with_conclusion":
            decision = "conclude"
            ready = True
            forced = False
        elif terminal_status == "completed_inconclusive":
            decision = "inconclusive"
            ready = False
            forced = True
        else:
            decision = "continue"
            ready = False
            forced = False
        blockers: list[str] = []
        if decision == "continue":
            background_ids = {item.condition_id for item in valid if not item.parameters}
            if "background-start" not in background_ids:
                blockers.append("尚未完成起始环境光基线。")
            if len(conditions) < _TARGET_CONDITION_COUNT:
                blockers.append(
                    f"有效距离条件覆盖 {len(conditions)}/{_TARGET_CONDITION_COUNT}。"
                )
            repeat_counts = {
                condition_id: sum(
                    item.valid and item.condition_id == condition_id for item in evidence
                )
                for condition_id in conditions
            }
            incomplete = sorted(
                condition_id
                for condition_id, count in repeat_counts.items()
                if count < _TARGET_REPEATS
            )
            if incomplete:
                blockers.append("仍需完成同距离重复：" + "、".join(incomplete) + "。")
            if "background-end" not in background_ids:
                blockers.append("尚未完成结束环境光漂移检查。")
            blockers.append("尚未运行最终距离衰减拟合。")
        elif blocker and decision != "conclude":
            blockers = [blocker]
        return ExperimentProgress(
            measurements_used=len(evidence),
            corrections_used=corrections,
            valid_evidence_count=len(valid),
            distinct_condition_count=min(_TARGET_CONDITION_COUNT, len(conditions)),
            condition_coverage_ratio=min(1.0, len(conditions) / _TARGET_CONDITION_COUNT),
            quality_pass_rate=(len(valid) / len(evidence) if evidence else 0.0),
            recent_information_gain=(0.0 if not evidence else (1.0 if evidence[-1].valid else 0.0)),
            conclusion_ready=ready,
            forced_stop=forced,
            decision=decision,
            blockers=blockers,
        )

    def _background_drift_problem(self, evidence: list[ExperimentEvidence]) -> str | None:
        start = self._background_value(evidence, ending=False)
        end = self._background_value(evidence, ending=True)
        condition_values = [
            _snapshot_metric(item, "median_illuminance_lx")
            for item in evidence
            if item.valid and item.parameters
        ]
        weakest_net = min(condition_values) - (start + end) / 2.0
        tolerance = max(1.0, abs(weakest_net) * 0.1)
        if abs(end - start) > tolerance:
            return "前后环境光漂移超过最弱净信号的 10%（且至少 1 lx）。"
        return None

    @staticmethod
    def _background_value(evidence: list[ExperimentEvidence], *, ending: bool) -> float:
        expected_sequence = -1 if ending else 0
        backgrounds = [item for item in evidence if item.valid and not item.parameters]
        if not backgrounds:
            raise RuntimeError("background evidence is missing")
        selected = backgrounds[expected_sequence]
        return _snapshot_metric(selected, "median_illuminance_lx")

    @staticmethod
    def _evidence_distance(evidence: ExperimentEvidence) -> float:
        for item in evidence.parameters:
            if item.key == _DISTANCE_KEY and item.unit == _DISTANCE_UNIT:
                return float(item.value)
        raise RuntimeError("light condition evidence has no distance_m")

    @staticmethod
    def _target_distance(task: ExperimentTask) -> float:
        for item in task.parameter_targets:
            if item.key == _DISTANCE_KEY:
                return float(item.value)
        return 0.0

    @staticmethod
    def _light_measurement(
        evidence: ExperimentEvidence,
        condition_id: str,
    ) -> LightMeasurement:
        return LightMeasurement(
            evidence_id=evidence.evidence_id,
            condition_id=condition_id,
            distance_m=InvestigationStore._evidence_distance(evidence),
            observed_illuminance_lx=_snapshot_metric(evidence, "median_illuminance_lx"),
        )

    @staticmethod
    def _design_execution(
        *,
        task: ExperimentTask,
        evidence: ExperimentEvidence,
        input_evidence_ids: list[str] | None = None,
        next_distance: float,
        sequence: int,
        reason_code: float,
    ) -> ToolExecution:
        return ToolExecution(
            execution_id=f"exec-{uuid4().hex[:12]}",
            sequence=sequence,
            task_id=task.task_id,
            tool_id="select_next_design_point",
            tool_version="1.0.0",
            input_evidence_ids=input_evidence_ids or [evidence.evidence_id],
            status="succeeded",
            result_metrics=[
                MetricSnapshot(
                    key="next_distance_m",
                    label="下一设计距离",
                    value=float(next_distance),
                    unit="m",
                ),
                MetricSnapshot(
                    key="design_reason_code",
                    label="设计原因代码",
                    value=float(reason_code),
                    unit="code",
                ),
            ],
        )


def _metric_value(recording: StoredSensorRecording, key: str) -> float:
    return float(recording.analysis.metric_value(key))


def _snapshot_metric(evidence: ExperimentEvidence, key: str) -> float:
    if evidence.analysis is None:
        raise RuntimeError("evidence has no analysis snapshot")
    for metric in evidence.analysis.metrics:
        if metric.key == key:
            return float(metric.value)
    raise RuntimeError(f"evidence metric is missing: {key}")
investigation_store = InvestigationStore(database, session_store, user_id=None)
