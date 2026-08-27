from __future__ import annotations

import re
from threading import RLock
from uuid import uuid4

from pocketlab.auth import get_current_user_id
from pocketlab.diagnostic_evidence import build_measurement_fact, get_diagnostic_recording
from pocketlab.experiment_guidance import (
    QUALITY_CORRECTION_CONTROLS,
    QUALITY_CORRECTION_CORE_INSTRUCTION,
    QUALITY_CORRECTION_VARIABLE,
    STABILITY_OBSERVATION_CORE_INSTRUCTION,
    assert_experiment_operation_guide,
    build_experiment_operation_guide,
    operation_text_is_single_record,
)
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.reality_feedback import (
    RealityEvidenceReuseCandidate,
    RealityFeedbackRecord,
    RealityFeedbackRequest,
    build_reality_evidence_reuse_audit,
    revised_context,
)
from pocketlab.schemas import (
    DiagnosticCase,
    DiagnosticCaseCreate,
    DiagnosticCaseHistoryItem,
    DiagnosticCaseSnapshot,
    DiagnosticControlEffect,
    DiagnosticEvidence,
    DiagnosticFinalReport,
    DiagnosticHypothesis,
    DiagnosticHypothesisDraft,
    DiagnosticMeasurementTask,
    DiagnosticReasoningReceipt,
    DiagnosticSensorPlanDraft,
    DiagnosticSensorPlanItem,
    DiagnosticTerminationVector,
    HypothesisAssessmentDraft,
    MeasurementTaskDraft,
)
from pocketlab.sensor_requirements import (
    explicit_sensor_preference,
    infer_task_sensor,
    sensor_requirement,
)
from pocketlab.solutions import build_solution_plan
from pocketlab.store import SessionStore, session_store

_BASELINE_NO_CHANGE_MARKERS = (
    "不改变",
    "不改动",
    "保持当前",
    "保持现状",
    "维持当前",
    "仅记录",
    "只记录",
    "无变量更改",
    "无状态改变",
    "建立基线",
    "采集基线",
)
_BASELINE_CHANGE_PATTERN = re.compile(
    r"(?:→|->|依次|切换|改为|调整为|提升|降低|重新分布|从.{0,30}(?:到|至))"
)


def _diagnostic_reuse_candidates(case: DiagnosticCase) -> tuple[RealityEvidenceReuseCandidate, ...]:
    candidates: list[RealityEvidenceReuseCandidate] = []
    user_sources = {"phyphox_remote", "phone_upload", "file_import", "legacy_session"}
    for evidence in case.evidence:
        facts = tuple(evidence.facts)
        sources = {item.provenance_source for item in facts}
        eligible = True
        blocker = None
        if evidence.quality == "low" or any(item.quality == "low" for item in facts):
            eligible = False
            blocker = "low-quality-or-invalid"
        elif not facts:
            eligible = False
            blocker = "missing-structured-facts"
        elif not sources or not sources <= user_sources:
            eligible = False
            blocker = "non-user-evidence-source"
        summaries = [
            f"{item.metric_label}为 {item.value:.4g} {item.metric_unit or '（无单位）'}"
            for item in facts[:3]
        ]
        candidates.append(
            RealityEvidenceReuseCandidate(
                evidence_id=evidence.evidence_id,
                sensor=evidence.sensor,
                planning_summary=("；".join(summaries) + "。") if summaries else "",
                eligible=eligible,
                exclusion_reason_code=blocker,
            )
        )
    return tuple(candidates)


def is_observational_baseline_task(task: DiagnosticMeasurementTask | MeasurementTaskDraft) -> bool:
    """Return whether a baseline observes one unchanged condition only."""

    variable = task.variable_to_change.strip()
    has_no_change_marker = any(marker in variable for marker in _BASELINE_NO_CHANGE_MARKERS)
    contains_transition = bool(_BASELINE_CHANGE_PATTERN.search(variable))
    return has_no_change_marker and not contains_transition


def build_diagnostic_retest_request(case: DiagnosticCase) -> DiagnosticCaseCreate:
    """Turn a finished report's optional retest into a new executable diagnosis."""

    report = case.final_report
    retest = report.solution_plan.optional_retest if report and report.solution_plan else None
    if report is None or retest is None:
        raise ValueError("该案例没有可执行的处理后复测方案。")
    controls = "、".join(retest.controlled_variables) or "原案例记录的测点与工况"
    criteria = "；".join(retest.success_criteria) or "按原报告的主要表征判断是否改善"
    context = "\n".join(
        [
            f"来源案例：{case.case_id} · {case.title}",
            f"原问题：{case.problem_statement}",
            f"原结论：{report.answer_headline or report.conclusion}",
            f"复测目的：{retest.purpose}",
            f"建议操作：{retest.instruction}",
            f"必须保持：{controls}",
            f"成功标准：{criteria}",
            "这是独立的新诊断。先规划当前处理后状态的可重复基线，再根据安全性选择匹配对照；不得把原报告文字当作新证据。",
        ]
    )[:1000]
    return DiagnosticCaseCreate(
        title=f"{case.title[:62]} · 处理后复测",
        problem_statement=(
            f"复核原案例“{case.title}”的处理是否有效，并用同一测点、同一传感器和"
            "受控条件给出普通用户可执行的改善判断。"
        )[:1000],
        context=context,
    )


class DiagnosticCaseStore:
    """Thread-safe diagnostic state backed by SQLite case snapshots."""

    def __init__(
        self,
        storage: SQLiteDatabase | None = None,
        sessions: SessionStore | None = None,
        *,
        user_id: str | None = DEFAULT_USER_ID,
    ) -> None:
        self._database = storage or SQLiteDatabase(":memory:")
        self._session_store = sessions or session_store
        self._user_id = user_id
        self._lock = RLock()

    @property
    def _active_user_id(self) -> str:
        return self._user_id or get_current_user_id()

    def create(self, request: DiagnosticCaseCreate) -> DiagnosticCase:
        user_id = self._active_user_id
        case = DiagnosticCase(
            case_id=uuid4().hex[:12],
            title=request.title,
            problem_statement=request.problem_statement,
            context=request.context,
        )
        with self._lock:
            now = utc_now()
            self._database.execute(
                """
                INSERT INTO diagnostic_cases(
                    case_id, user_id, case_json, latest_agent_message, created_at, updated_at
                ) VALUES (?, ?, ?, '', ?, ?)
                """,
                (case.case_id, user_id, case.model_dump_json(), now, now),
            )
        return case.model_copy(deep=True)

    def get(self, case_id: str) -> DiagnosticCase:
        with self._lock:
            return self._require_case(case_id).model_copy(deep=True)

    def get_snapshot(self, case_id: str) -> DiagnosticCaseSnapshot:
        with self._lock:
            row = self._case_row(case_id)
            return DiagnosticCaseSnapshot(
                case=self._load_case(row["case_json"]),
                latest_agent_message=row["latest_agent_message"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def replay_committed_measurement(
        self,
        case_id: str,
        *,
        task_id: str,
        session_id: str,
    ) -> DiagnosticCase | None:
        """Return current state when an identical measurement was already committed."""

        with self._lock:
            case = self._require_case(case_id)
            if any(
                item.task_id == task_id and item.session_id == session_id
                for item in case.evidence
            ):
                return case.model_copy(deep=True)
            return None

    def list(self, *, limit: int = 100) -> list[DiagnosticCaseHistoryItem]:
        user_id = self._active_user_id
        rows = self._database.fetch_all(
            """
            SELECT * FROM diagnostic_cases WHERE user_id = ?
            ORDER BY updated_at DESC LIMIT ?
            """,
            (user_id, limit),
        )
        items = []
        for row in rows:
            case = self._load_case(row["case_json"])
            items.append(
                DiagnosticCaseHistoryItem(
                    case_id=case.case_id,
                    title=case.title,
                    problem_statement=case.problem_statement,
                    status=case.status,
                    current_task_title=case.current_task.title if case.current_task else None,
                    evidence_count=len(case.evidence),
                    superseded_by_case_id=case.superseded_by_case_id,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return items

    def set_latest_agent_message(self, case_id: str, message: str) -> None:
        user_id = self._active_user_id
        with self._lock:
            self._case_row(case_id)
            self._database.execute(
                """
                UPDATE diagnostic_cases
                SET latest_agent_message = ?, updated_at = ?
                WHERE case_id = ? AND user_id = ?
                """,
                (message, utc_now(), case_id, user_id),
            )

    def set_intake_runtime(
        self,
        case_id: str,
        *,
        transport: str,
        model: str,
        model_requests: int,
        elapsed_ms: int,
        fallback_reason: str | None,
    ) -> DiagnosticCase:
        with self._lock:
            case = self._require_case(case_id)
            case.intake_transport = transport
            case.intake_model = model
            case.intake_model_requests = model_requests
            case.intake_elapsed_ms = elapsed_ms
            case.intake_fallback_reason = fallback_reason
            self._persist_case(case)
            return case.model_copy(deep=True)

    def set_final_report(
        self,
        case_id: str,
        report: DiagnosticFinalReport,
    ) -> DiagnosticCase:
        """Replace only a finished case's report after model finalization.

        Evidence, hypotheses, tasks and the server termination vector remain
        immutable. This makes a retry safe: it can improve explanatory prose and
        the solution playbook without rerunning or rewriting the experiment.
        """

        with self._lock:
            case = self._require_case(case_id)
            if case.final_report is None or not case.status.startswith("completed_"):
                raise ValueError("only a completed diagnostic case can be finalized")
            if report.outcome != case.final_report.outcome:
                raise ValueError("finalization cannot change the diagnostic outcome")
            if report.vector != case.termination_vector:
                raise ValueError("finalization cannot change the termination vector")
            if report.leading_hypothesis_id != case.final_report.leading_hypothesis_id:
                raise ValueError("finalization cannot change the leading hypothesis")
            case.final_report = report.model_copy(deep=True)
            self._persist_case(case)
            return case.model_copy(deep=True)

    def delete(self, case_id: str) -> None:
        user_id = self._active_user_id
        with self._lock:
            self._database.execute(
                "DELETE FROM diagnostic_cases WHERE case_id = ? AND user_id = ?",
                (case_id, user_id),
            )

    def create_reality_feedback_revision(
        self,
        case_id: str,
        request: RealityFeedbackRequest,
    ) -> DiagnosticCase:
        """Branch a fresh plan while retaining the source case and all of its evidence."""

        user_id = self._active_user_id
        with self._lock:
            source_row = self._case_row(case_id)
            source = self._load_case(source_row["case_json"])
            if source.superseded_by_case_id is not None:
                raise ValueError("该案例已经根据现场反馈生成过新版本。")
            if source.final_report is not None:
                raise ValueError("已结束案例请使用复测入口；现场反馈用于修正进行中的计划。")
            current_task_id = source.current_task.task_id if source.current_task else None
            if (
                request.expected_task_id is not None
                and request.expected_task_id != current_task_id
            ):
                raise ValueError("当前任务已经变化，请刷新后再提交现场反馈。")
            known_hypotheses = {item.hypothesis_id: item for item in source.hypotheses}
            unknown_ids = set(request.hypothesis_ids) - set(known_hypotheses)
            if unknown_ids:
                raise ValueError("反馈引用了当前计划中不存在的候选解释。")
            rejected_statements = tuple(
                known_hypotheses[item].statement for item in request.hypothesis_ids
            )
            evidence_reuse = build_reality_evidence_reuse_audit(
                _diagnostic_reuse_candidates(source),
                confirm_sensitive_sensor_reuse=request.confirm_sensitive_sensor_reuse,
            )
            new_case_id = uuid4().hex[:12]
            now = utc_now()
            feedback = RealityFeedbackRecord(
                feedback_id=f"feedback-{uuid4().hex[:16]}",
                feedback_type=request.feedback_type,
                message=request.message,
                hypothesis_ids=request.hypothesis_ids,
                source_case_id=source.case_id,
                source_task_id=current_task_id,
                preserved_evidence_ids=tuple(item.evidence_id for item in source.evidence),
                evidence_reuse=evidence_reuse,
                created_at=now,
            )
            revised = DiagnosticCase(
                case_id=new_case_id,
                title=source.title,
                problem_statement=source.problem_statement,
                context=revised_context(
                    original_context=source.context,
                    feedback=request,
                    rejected_hypotheses=rejected_statements,
                    task_title=source.current_task.title if source.current_task else None,
                    limit=1000,
                    evidence_reuse=evidence_reuse,
                ),
                revision_parent_case_id=source.case_id,
                revision_feedback=feedback,
            )
            source.superseded_by_case_id = revised.case_id
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO diagnostic_cases(
                        case_id, user_id, case_json, latest_agent_message, created_at, updated_at
                    ) VALUES (?, ?, ?, '', ?, ?)
                    """,
                    (revised.case_id, user_id, revised.model_dump_json(), now, now),
                )
                cursor = connection.execute(
                    """
                    UPDATE diagnostic_cases SET case_json = ?, updated_at = ?
                    WHERE case_id = ? AND user_id = ?
                    """,
                    (source.model_dump_json(), now, source.case_id, user_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Unknown case_id: {source.case_id}")
            return revised.model_copy(deep=True)

    def rollback_reality_feedback_revision(self, case_id: str) -> None:
        """Restore the source only when a brand-new revision failed before planning."""

        user_id = self._active_user_id
        with self._lock:
            revised = self._require_case(case_id)
            source_id = revised.revision_parent_case_id
            if (
                source_id is None
                or revised.hypotheses
                or revised.current_task is not None
                or revised.evidence
            ):
                raise ValueError("只有尚未生成计划的新修订可以回滚。")
            source = self._require_case(source_id)
            if source.superseded_by_case_id != revised.case_id:
                raise ValueError("修订链已经变化，不能回滚。")
            source.superseded_by_case_id = None
            now = utc_now()
            with self._database.transaction() as connection:
                connection.execute(
                    "DELETE FROM diagnostic_cases WHERE case_id = ? AND user_id = ?",
                    (revised.case_id, user_id),
                )
                connection.execute(
                    """
                    UPDATE diagnostic_cases SET case_json = ?, updated_at = ?
                    WHERE case_id = ? AND user_id = ?
                    """,
                    (source.model_dump_json(), now, source.case_id, user_id),
                )

    def commit_initial_plan(
        self,
        case_id: str,
        hypotheses: list[DiagnosticHypothesisDraft],
        task: MeasurementTaskDraft,
        sensor_plan: list[DiagnosticSensorPlanDraft] | None = None,
    ) -> DiagnosticCase:
        if not 2 <= len(hypotheses) <= 3:
            raise ValueError("初始诊断计划必须包含 2 到 3 个候选假设。")

        with self._lock:
            case = self._require_case(case_id)
            if case.hypotheses or case.current_task is not None:
                raise ValueError("该诊断案例已经生成过初始计划。")

            case_text = f"{case.title} {case.problem_statement} {case.context}"
            user_primary = explicit_sensor_preference(case_text)
            if sensor_plan is not None and user_primary is not None:
                model_primary = next(
                    (item.sensor for item in sensor_plan if item.role == "primary"),
                    None,
                )
                if task.required_sensor != user_primary or model_primary != user_primary:
                    raise ValueError(
                        "model sensor plan contradicts the user's explicit primary sensor"
                    )

            case.hypotheses = [
                DiagnosticHypothesis(
                    hypothesis_id=f"h{index}",
                    statement=draft.statement,
                    rationale=draft.rationale,
                    critical_prediction=draft.critical_prediction,
                    critical_sensor=draft.critical_sensor,
                    critical_expected_effect=draft.critical_expected_effect,
                )
                for index, draft in enumerate(hypotheses, start=1)
            ]
            if sensor_plan is None:
                task.required_sensor = infer_task_sensor(
                    task.required_sensor,
                    task_text=f"{task.title} {task.instruction} {task.variable_to_change}",
                    case_text=case_text,
                )
            else:
                self._resolve_task_sensor(case, task)
            case.sensor_plan = self._build_sensor_plan(sensor_plan, task)
            self._validate_task_draft(case, task, initial=True)
            case.current_task = self._build_task(task, number=1)
            self._apply_task_role(case, case.current_task)
            case.status = "collecting"
            self._persist_case(case)
            return case.model_copy(deep=True)

    def commit_measurement(
        self,
        case_id: str,
        task_id: str,
        session_id: str,
        observation_notes: str,
        evidence_summary: str,
        assessments: list[HypothesisAssessmentDraft],
        next_task: MeasurementTaskDraft,
        reasoning_receipt: DiagnosticReasoningReceipt | None = None,
        stop_inconclusive: bool = False,
    ) -> DiagnosticCase:
        session = get_diagnostic_recording(self._session_store, session_id)

        with self._lock:
            case = self._require_case(case_id)
            if case.superseded_by_case_id is not None:
                raise ValueError("该案例已被现场反馈重规划，请进入新版本继续。")
            current_task = case.current_task
            if current_task is None or current_task.task_id != task_id:
                expected = current_task.task_id if current_task else None
                raise ValueError(
                    f"task_id 不是当前待执行任务；expected={expected!r}, received={task_id!r}"
                )
            if current_task.analyzer_status != "ready":
                raise ValueError(
                    f"task analyzer is not Agent-ready for {current_task.required_sensor}"
                )
            if session.sensor != current_task.required_sensor:
                raise ValueError(
                    "session sensor does not match the current task: "
                    f"expected={current_task.required_sensor!r}, "
                    f"received={session.sensor!r}"
                )

            expected_ids = {item.hypothesis_id for item in case.hypotheses}
            received_ids = {item.hypothesis_id for item in assessments}
            if received_ids != expected_ids or len(assessments) != len(expected_ids):
                raise ValueError("必须且只能评估案例中的每一个候选假设。")

            target_ids = set(current_task.target_hypothesis_ids)
            assessments = [
                item.model_copy(
                    update={
                        "critical_prediction_tested": (
                            item.critical_prediction_tested and item.hypothesis_id in target_ids
                        )
                    }
                )
                for item in assessments
            ]

            baseline = self._comparison_recording(case, current_task)
            fact, control_effect = build_measurement_fact(
                task=current_task,
                recording=session,
                baseline=baseline,
            )
            known_fact_ids = {
                item.fact_id for evidence in case.evidence for item in evidence.facts
            } | {fact.fact_id}
            if reasoning_receipt is not None:
                if set(reasoning_receipt.source_fact_ids) - known_fact_ids:
                    raise ValueError("reasoning receipt references unknown deterministic facts")
                expected_hypotheses = {item.hypothesis_id for item in case.hypotheses}
                if set(reasoning_receipt.ranked_hypothesis_ids) != expected_hypotheses:
                    raise ValueError("reasoning receipt must rank every candidate hypothesis")

            evidence_id = uuid4().hex[:12]
            if session.analysis.confidence == "low":
                assessments = [
                    HypothesisAssessmentDraft(
                        hypothesis_id=item.hypothesis_id,
                        status="inconclusive",
                        reasoning="测量质量门禁未通过，当前数据不能用于增强或削弱该假设。",
                        critical_prediction_tested=False,
                    )
                    for item in assessments
                ]
                evidence_summary = f"质量门禁未通过：{evidence_summary}"
            elif control_effect is not None and not control_effect.comparable:
                warning_text = "；".join(control_effect.comparison_warnings)
                assessments = [
                    HypothesisAssessmentDraft(
                        hypothesis_id=item.hypothesis_id,
                        status="inconclusive",
                        reasoning=(
                            "对照协议可比性门禁未通过，当前数据不能增强或削弱该假设。"
                            f"原因：{warning_text}"
                        ),
                        critical_prediction_tested=False,
                    )
                    for item in assessments
                ]
                evidence_summary = (
                    f"对照协议不可比（{warning_text}），本轮只保留单次测量描述：{evidence_summary}"
                )
            elif control_effect is not None:
                assessments = self._calibrate_structured_predictions(
                    case,
                    current_task,
                    control_effect,
                    assessments,
                )

            evidence = DiagnosticEvidence(
                evidence_id=evidence_id,
                task_id=task_id,
                session_id=session_id,
                quality=session.analysis.confidence,
                summary=evidence_summary,
                observation_notes=observation_notes,
                hypothesis_assessments=assessments,
                control_effect=control_effect,
                sensor=session.sensor,
                facts=[fact],
                reasoning_receipt=reasoning_receipt,
            )
            assessment_by_id = {item.hypothesis_id: item for item in assessments}
            for hypothesis in case.hypotheses:
                assessment = assessment_by_id[hypothesis.hypothesis_id]
                hypothesis.status = assessment.status
                hypothesis.latest_reasoning = assessment.reasoning
                hypothesis.evidence_ids.append(evidence_id)

            current_task.status = "completed"
            case.completed_tasks.append(current_task)
            case.evidence.append(evidence)
            case.termination_vector = self._evaluate_termination(case)
            if case.termination_vector.hypothesis_revision_required:
                next_task = next_task.model_copy(
                    update={
                        "title": f"重规划判别：{next_task.title}"[:100],
                        "instruction": (
                            "当前候选解释已被本轮对照同时削弱。不要重复上一项条件；"
                            "请按下面步骤采集能扩大或重建原因范围的新表征："
                            + next_task.instruction
                        )[:800],
                        "task_kind": "exploration",
                        "comparison_task_id": None,
                        "target_hypothesis_ids": [
                            item.hypothesis_id for item in case.hypotheses
                        ],
                        "expected_effect": "unknown",
                    }
                )
            if case.termination_vector.conclusion_ready:
                case.current_task = None
                case.status = "completed_with_conclusion"
                case.final_report = self._build_final_report(case, conclusive=True)
            # A provider proposal cannot end a live diagnosis before the server-owned
            # user checkpoint.  The only short bounded stop is an explicitly labelled
            # finite public-replay rehearsal, which is not household evidence.
            elif self._can_stop_public_rehearsal(case):
                case.termination_vector.bounded_rehearsal_stop = True
                case.termination_vector.stop_reason_code = "public-replay-evidence-exhausted"
                case.current_task = None
                case.status = "completed_inconclusive"
                case.final_report = self._build_final_report(case, conclusive=False)
            elif case.termination_vector.user_decision_required:
                self._resolve_task_sensor(case, next_task)
                self._validate_task_draft(case, next_task, initial=False)
                case.checkpoint_next_task = self._build_task(
                    next_task,
                    number=len(case.completed_tasks) + 1,
                )
                self._apply_task_role(case, case.checkpoint_next_task)
                case.current_task = None
                case.status = "awaiting_user_decision"
                case.checkpoint_pending = True
            elif case.termination_vector.forced_stop:
                case.current_task = None
                case.status = "completed_inconclusive"
                case.final_report = self._build_final_report(case, conclusive=False)
            else:
                self._resolve_task_sensor(case, next_task)
                self._validate_task_draft(case, next_task, initial=False)
                case.current_task = self._build_task(
                    next_task,
                    number=len(case.completed_tasks) + 1,
                )
                self._apply_task_role(case, case.current_task)
                case.status = "collecting"
            self._persist_case(case)
            return case.model_copy(deep=True)

    @staticmethod
    def _calibrate_structured_predictions(
        case: DiagnosticCase,
        task: DiagnosticMeasurementTask,
        effect: DiagnosticControlEffect,
        assessments: list[HypothesisAssessmentDraft],
    ) -> list[HypothesisAssessmentDraft]:
        if task.task_kind == "replication":
            return [
                HypothesisAssessmentDraft(
                    hypothesis_id=item.hypothesis_id,
                    status="inconclusive",
                    reasoning=(
                        "本轮只检验相同条件能否重复；重复性可以提高事实可信度，"
                        "但不能单独增强或削弱任何原因假设。"
                    ),
                    critical_prediction_tested=False,
                )
                for item in assessments
            ]
        hypotheses = {item.hypothesis_id: item for item in case.hypotheses}
        targets = set(task.target_hypothesis_ids)
        calibrated = []
        for assessment in assessments:
            hypothesis = hypotheses[assessment.hypothesis_id]
            expected = hypothesis.critical_expected_effect
            if (
                assessment.hypothesis_id not in targets
                or hypothesis.critical_sensor != task.required_sensor
                or expected == "unknown"
            ):
                calibrated.append(assessment)
                continue
            matches = (
                effect.observed_effect in {"increase", "decrease"}
                if expected == "change"
                else effect.observed_effect == expected
            )
            calibrated.append(
                HypothesisAssessmentDraft(
                    hypothesis_id=assessment.hypothesis_id,
                    status="supported" if matches else "weakened",
                    reasoning=(
                        "服务端结构化关键预测与确定性对照方向一致。"
                        if matches
                        else "服务端结构化关键预测与确定性对照方向冲突。"
                    ),
                    critical_prediction_tested=True,
                )
            )
        return calibrated

    @staticmethod
    def _can_stop_public_rehearsal(case: DiagnosticCase) -> bool:
        consumed = list(case.evidence)
        if len(consumed) < 2:
            return False
        provenances = {fact.provenance_source for evidence in consumed for fact in evidence.facts}
        if provenances != {"public_replay"}:
            return False
        planned_sensors = {
            item.sensor
            for item in case.sensor_plan
            if item.sensor != "bluetooth" and item.analyzer_status == "ready"
        }
        if not planned_sensors:
            planned_sensors = {item.sensor for item in consumed}
        evidence_count_by_sensor = {
            sensor: sum(item.sensor == sensor for item in consumed) for sensor in planned_sensors
        }
        primary_sensors = {
            item.sensor
            for item in case.sensor_plan
            if item.role == "primary"
            and item.sensor != "bluetooth"
            and item.analyzer_status == "ready"
        }
        if not primary_sensors:
            primary_sensors = {next(iter(planned_sensors))}
        # Reviewed replay packs are intentionally finite and do not represent the
        # user's home.  Rehearse one baseline/control pair on the primary sensor,
        # then route through each useful auxiliary sensor once.  Requiring a full
        # pair for every auxiliary can consume the finite pack without adding site
        # validity and used to leave the case stranded on a third unavailable item.
        return all(
            count >= (2 if sensor in primary_sensors else 1)
            for sensor, count in evidence_count_by_sensor.items()
        )

    def _comparison_recording(
        self,
        case: DiagnosticCase,
        task: DiagnosticMeasurementTask,
    ):
        if task.task_kind not in {"control", "replication"} or not task.comparison_task_id:
            return None
        baseline_evidence = next(
            (item for item in case.evidence if item.task_id == task.comparison_task_id),
            None,
        )
        if baseline_evidence is None:
            raise ValueError("当前对照任务找不到对应的基线证据。")
        return get_diagnostic_recording(
            self._session_store,
            baseline_evidence.session_id,
        )

    def decide_checkpoint(
        self,
        case_id: str,
        *,
        decision: str,
        expected_completed_task_count: int,
        next_task: MeasurementTaskDraft | None = None,
    ) -> DiagnosticCase:
        with self._lock:
            case = self._require_case(case_id)
            if not case.checkpoint_pending or case.status != "awaiting_user_decision":
                raise ValueError("case is not waiting at the diagnostic checkpoint")
            if len(case.completed_tasks) != expected_completed_task_count:
                raise ValueError("diagnostic checkpoint changed; refresh before deciding")
            if decision == "stop":
                case.checkpoint_pending = False
                case.current_task = None
                case.checkpoint_next_task = None
                case.status = "completed_inconclusive"
                case.final_report = self._build_final_report(case, conclusive=False)
            elif decision == "continue":
                if case.checkpoint_next_task is None:
                    if next_task is None:
                        raise ValueError("continue requires a validated next task")
                    self._resolve_task_sensor(case, next_task)
                    self._validate_task_draft(case, next_task, initial=False)
                    case.current_task = self._build_task(
                        next_task,
                        number=len(case.completed_tasks) + 1,
                    )
                    self._apply_task_role(case, case.current_task)
                else:
                    case.current_task = case.checkpoint_next_task
                case.checkpoint_next_task = None
                case.checkpoint_pending = False
                case.continued_after_checkpoint = True
                case.status = "collecting"
            else:
                raise ValueError("checkpoint decision must be continue or stop")
            self._persist_case(case)
            return case.model_copy(deep=True)

    def clear(self) -> None:
        user_id = self._active_user_id
        with self._lock:
            self._database.execute(
                "DELETE FROM diagnostic_cases WHERE user_id = ?",
                (user_id,),
            )

    def _require_case(self, case_id: str) -> DiagnosticCase:
        row = self._case_row(case_id)
        return self._load_case(row["case_json"])

    @staticmethod
    def _load_case(case_json: str) -> DiagnosticCase:
        case = DiagnosticCase.model_validate_json(case_json)
        for task in [*case.completed_tasks, case.current_task]:
            if task is None:
                continue
            sensor = (
                task.required_sensor
                if case.sensor_plan
                else infer_task_sensor(
                    task.required_sensor,
                    task_text=f"{task.title} {task.instruction} {task.variable_to_change}",
                    case_text=f"{case.title} {case.problem_statement} {case.context}",
                )
            )
            requirement = sensor_requirement(sensor)
            task.required_sensor = sensor
            task.measurement_quantity = requirement.measurement_quantity
            task.recommended_phyphox_experiment = requirement.recommended_phyphox_experiment
            task.analyzer_status = requirement.analyzer_status
            task.target_metric_key = task.target_metric_key or requirement.default_metric_key
            task.sensor_role = next(
                (item.role for item in case.sensor_plan if item.sensor == task.required_sensor),
                "primary",
            )
        current_task = case.current_task
        if current_task is not None and current_task.status == "pending":
            task_text = f"{current_task.instruction} {current_task.variable_to_change}"
            legacy_correction = current_task.task_kind == "correction" and (
                "单一变量：" in current_task.instruction
                or "select…" in current_task.instruction
                or not operation_text_is_single_record(current_task.instruction)
            )
            legacy_unknown_factor = any(
                marker in task_text
                for marker in (
                    "一个尚未检验的安全因素",
                    "某个尚未检验的安全因素",
                    "一个安全设置内容",
                    "某个安全设置内容",
                )
            )
            if legacy_correction:
                current_task.instruction = build_experiment_operation_guide(
                    core_instruction=QUALITY_CORRECTION_CORE_INSTRUCTION,
                    sensors=(current_task.required_sensor,),
                    variable_to_change=QUALITY_CORRECTION_VARIABLE,
                    controlled_variables=QUALITY_CORRECTION_CONTROLS,
                    default_duration_s=5,
                    task_kind="correction",
                )
                current_task.variable_to_change = QUALITY_CORRECTION_VARIABLE
                current_task.controlled_variables = list(QUALITY_CORRECTION_CONTROLS)
            elif legacy_unknown_factor:
                current_task.instruction = build_experiment_operation_guide(
                    core_instruction=STABILITY_OBSERVATION_CORE_INSTRUCTION,
                    sensors=(current_task.required_sensor,),
                    variable_to_change="不引入新变量，仅观察当前已定义对照条件",
                    controlled_variables=("手机位置与姿态", "记录时长", "当前已定义工况"),
                    default_duration_s=5,
                    task_kind="exploration",
                )
                current_task.variable_to_change = "不引入新变量，仅观察当前已定义对照条件"
                current_task.controlled_variables = [
                    "手机位置与姿态",
                    "记录时长",
                    "当前已定义工况",
                ]
        report = case.final_report
        false_legacy_conclusion = bool(
            report is not None
            and report.outcome == "completed_with_conclusion"
            and case.hypotheses
            and all(item.status == "weakened" for item in case.hypotheses)
        )
        if false_legacy_conclusion:
            reason = (
                "旧版本曾把单变量干预效果误当成原因结论；当前候选假设实际全部被削弱，"
                "严格终止门已使该结论失效。"
            )
            case.status = "completed_inconclusive"
            case.termination_invalidated = True
            case.termination_invalidation_reason = reason
            case.termination_vector = case.termination_vector.model_copy(
                update={
                    "leading_hypothesis_id": None,
                    "runner_up_hypothesis_id": None,
                    "leading_support": 0.5,
                    "runner_up_support": 0.5,
                    "leading_margin": 0.0,
                    "leading_positive_weight": 0.0,
                    "hypothesis_set_state": "all_weakened",
                    "hypothesis_revision_required": True,
                    "conclusion_ready": False,
                    "forced_stop": False,
                    "blockers": ["候选假设已全部被削弱，需要重规划"],
                }
            )
            report = report.model_copy(
                update={
                    "outcome": "completed_inconclusive",
                    "confidence": "low",
                    "leading_hypothesis_id": None,
                    "conclusion": reason,
                    "termination_reason": reason,
                    "vector": case.termination_vector,
                    "answer_headline": "旧版原因结论已失效，需要建立新的判别诊断",
                    "mechanism_explanation": (
                        "所有注册候选都与当前受控观测冲突，因此不能按排序顺序强选一个原因。"
                        "已观测到的干预效果仍可保留为事实，但必须扩大候选原因范围后再测。"
                    ),
                    "ranked_hypothesis_ids": [],
                    "user_takeaway": (
                        "不要按旧版高置信结论处理；可使用下方按钮建立独立的新诊断，"
                        "原始证据仍会保留。"
                    ),
                    "scope_boundary": (report.scope_boundary + " " + reason).strip(),
                }
            )
            case.final_report = report
            report.solution_plan = build_solution_plan(case, conclusive=False)
        if report is not None and report.solution_plan is None:
            report.solution_plan = build_solution_plan(
                case,
                conclusive=report.outcome == "completed_with_conclusion",
            )
        return case

    def _case_row(self, case_id: str) -> object:
        user_id = self._active_user_id
        row = self._database.fetch_one(
            "SELECT * FROM diagnostic_cases WHERE case_id = ? AND user_id = ?",
            (case_id, user_id),
        )
        if row is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        return row

    def _persist_case(self, case: DiagnosticCase) -> None:
        user_id = self._active_user_id
        self._database.execute(
            """
            UPDATE diagnostic_cases SET case_json = ?, updated_at = ?
            WHERE case_id = ? AND user_id = ?
            """,
            (case.model_dump_json(), utc_now(), case.case_id, user_id),
        )

    @staticmethod
    def _build_task(draft: MeasurementTaskDraft, number: int) -> DiagnosticMeasurementTask:
        requirement = sensor_requirement(draft.required_sensor)
        return DiagnosticMeasurementTask(
            task_id=f"task-{number}",
            title=draft.title,
            instruction=draft.instruction,
            variable_to_change=draft.variable_to_change,
            controlled_variables=draft.controlled_variables,
            required_sensor=requirement.sensor,
            target_metric_key=draft.target_metric_key or requirement.default_metric_key,
            measurement_quantity=requirement.measurement_quantity,
            recommended_phyphox_experiment=requirement.recommended_phyphox_experiment,
            analyzer_status=requirement.analyzer_status,
            task_kind=draft.task_kind,
            comparison_task_id=draft.comparison_task_id,
            target_hypothesis_ids=draft.target_hypothesis_ids,
            expected_effect=draft.expected_effect,
            effect_metric=draft.effect_metric,
        )

    @staticmethod
    def _build_sensor_plan(
        drafts: list[DiagnosticSensorPlanDraft] | None,
        first_task: MeasurementTaskDraft,
    ) -> list[DiagnosticSensorPlanItem]:
        active = drafts or [
            DiagnosticSensorPlanDraft(
                sensor=first_task.required_sensor,
                role="primary",
                rationale="首项测量直接表征用户报告的主要物理现象。",
                target_metric_key=first_task.target_metric_key,
            )
        ]
        if not 1 <= len(active) <= 4:
            raise ValueError("diagnostic sensor plan requires one to four sensors")
        sensors = [item.sensor for item in active]
        if len(sensors) != len(set(sensors)):
            raise ValueError("diagnostic sensor plan cannot duplicate sensors")
        if sum(item.role == "primary" for item in active) != 1:
            raise ValueError("diagnostic sensor plan requires exactly one primary sensor")
        if "bluetooth" in sensors and sensors != ["bluetooth"]:
            raise ValueError("Bluetooth detection cannot be mixed with numeric evidence")
        if first_task.required_sensor not in sensors:
            raise ValueError("first task sensor must be declared in the sensor plan")
        result = []
        for item in active:
            requirement = sensor_requirement(item.sensor)
            metric_key = item.target_metric_key or requirement.default_metric_key
            if item.sensor == "bluetooth":
                raise ValueError(
                    "Bluetooth is detection-only and cannot start a numeric diagnostic case"
                )
            if requirement.analyzer_status != "ready":
                raise ValueError(f"diagnostic analyzer is unavailable for {item.sensor}")
            if metric_key not in requirement.accepted_metric_keys:
                raise ValueError(f"unregistered diagnostic metric for {item.sensor}: {metric_key}")
            result.append(
                DiagnosticSensorPlanItem(
                    **item.model_dump(exclude={"target_metric_key"}),
                    target_metric_key=metric_key,
                    analyzer_status=requirement.analyzer_status,
                    measurement_quantity=requirement.measurement_quantity,
                    recommended_phyphox_experiment=requirement.recommended_phyphox_experiment,
                )
            )
        return result

    @staticmethod
    def _apply_task_role(
        case: DiagnosticCase,
        task: DiagnosticMeasurementTask,
    ) -> None:
        task.sensor_role = next(
            (item.role for item in case.sensor_plan if item.sensor == task.required_sensor),
            "primary",
        )

    @staticmethod
    def _resolve_task_sensor(case: DiagnosticCase, draft: MeasurementTaskDraft) -> None:
        # New diagnostic plans carry an explicit, model-authored sensor graph.  Lexical
        # keyword routing is retained only while loading old cases; it must not override
        # a current model decision.
        if case.sensor_plan and draft.required_sensor not in {
            item.sensor for item in case.sensor_plan
        }:
            raise ValueError("task sensor is outside the frozen sensor plan")

    @staticmethod
    def _validate_task_draft(
        case: DiagnosticCase,
        draft: MeasurementTaskDraft,
        *,
        initial: bool,
    ) -> None:
        hypothesis_ids = {item.hypothesis_id for item in case.hypotheses}
        unknown_targets = set(draft.target_hypothesis_ids) - hypothesis_ids
        if unknown_targets:
            raise ValueError(f"任务引用了未知假设：{sorted(unknown_targets)}")
        if not draft.target_hypothesis_ids:
            raise ValueError("任务必须至少指定一个 target_hypothesis_id。")
        if initial and draft.task_kind != "baseline":
            raise ValueError("第一项诊断任务必须是 baseline。")
        if initial and not is_observational_baseline_task(draft):
            raise ValueError(
                "基线任务只能记录当前未改变的状态；variable_to_change 必须明确写明"
                "不改变变量，不能包含切换条件、前后对照或多个测量阶段。"
            )
        if not initial and draft.task_kind == "baseline":
            raise ValueError("后续任务不能再次声明为 baseline。")
        if (
            draft.task_kind == "replication"
            and case.completed_tasks
            and case.completed_tasks[-1].task_kind == "replication"
            and case.completed_tasks[-1].required_sensor == draft.required_sensor
        ):
            raise ValueError("同一传感器不得连续追加重复验证；应切换计划内判别量或进入检查点。")
        if draft.comparison_task_id:
            completed_ids = {item.task_id for item in case.completed_tasks}
            if draft.comparison_task_id not in completed_ids:
                raise ValueError("comparison_task_id 必须引用已经完成的任务。")
            baseline_task = next(
                item for item in case.completed_tasks if item.task_id == draft.comparison_task_id
            )
            if baseline_task.required_sensor != draft.required_sensor:
                raise ValueError("numeric control tasks must compare the same sensor")
        planned = {item.sensor for item in case.sensor_plan}
        if planned and draft.required_sensor not in planned:
            raise ValueError("task sensor is outside the model-declared sensor plan")
        requirement = sensor_requirement(draft.required_sensor)
        if requirement.analyzer_status != "ready":
            raise ValueError(f"{draft.required_sensor} is not eligible for numeric diagnosis")
        metric_key = draft.target_metric_key or requirement.default_metric_key
        if metric_key not in requirement.accepted_metric_keys:
            raise ValueError(
                f"unregistered diagnostic metric for {draft.required_sensor}: {metric_key}"
            )
        draft.target_metric_key = metric_key
        if "准备：" in draft.instruction or "操作：" in draft.instruction:
            assert_experiment_operation_guide(
                draft.instruction,
                sensors=(draft.required_sensor,),
            )

    @staticmethod
    def _quality_weight(quality: str) -> float:
        return {"high": 1.0, "medium": 0.6, "low": 0.0}[quality]

    def _support_state(
        self,
        case: DiagnosticCase,
        evidence: list[DiagnosticEvidence],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        positive = {item.hypothesis_id: 0.0 for item in case.hypotheses}
        negative = {item.hypothesis_id: 0.0 for item in case.hypotheses}
        for item in evidence:
            weight = self._quality_weight(item.quality)
            task = next(
                (
                    candidate
                    for candidate in case.completed_tasks
                    if candidate.task_id == item.task_id
                ),
                None,
            )
            if task is not None and task.task_kind == "replication":
                continue
            for assessment in item.hypothesis_assessments:
                if assessment.status == "supported":
                    positive[assessment.hypothesis_id] += weight
                    if (
                        item.control_effect
                        and item.control_effect.matches_expected_effect
                        and task
                        and assessment.hypothesis_id in task.target_hypothesis_ids
                        and assessment.critical_prediction_tested
                    ):
                        # A matched, pre-declared contrast is a second independent signal:
                        # one semantic assessment plus one deterministic effect check.
                        positive[assessment.hypothesis_id] += weight
                elif assessment.status == "weakened":
                    negative[assessment.hypothesis_id] += weight
                    if (
                        item.control_effect
                        and item.control_effect.matches_expected_effect
                        and task
                        and assessment.hypothesis_id in task.target_hypothesis_ids
                        and assessment.critical_prediction_tested
                    ):
                        negative[assessment.hypothesis_id] += weight
        scores = {
            hypothesis_id: round(
                (1.0 + positive[hypothesis_id])
                / (2.0 + positive[hypothesis_id] + negative[hypothesis_id]),
                4,
            )
            for hypothesis_id in positive
        }
        return scores, positive, negative

    def _evaluate_termination(self, case: DiagnosticCase) -> DiagnosticTerminationVector:
        effective = [item for item in case.evidence if self._quality_weight(item.quality) > 0]
        task_by_id = {item.task_id: item for item in case.completed_tasks}
        distinct_sensors = {item.sensor for item in effective}
        # A supporting sensor is an available diagnostic option, not a checklist item.
        # One well-controlled direct sensor can close a household diagnosis on its own;
        # the reasoner should request a second sensor only when it resolves a live rival.
        required_sensor_diversity = 1
        controls = [
            item
            for item in effective
            if item.control_effect is not None
            and task_by_id.get(item.task_id) is not None
            and task_by_id[item.task_id].task_kind == "control"
        ]
        matched_controls = [
            item
            for item in controls
            if item.control_effect and item.control_effect.matches_expected_effect
        ]
        direct_interventions = [
            item
            for item in controls
            if item.quality == "high"
            and item.control_effect is not None
            and item.control_effect.comparable
            and item.control_effect.observed_effect in {"increase", "decrease"}
            and (
                (
                    item.control_effect.metric_unit.strip().casefold() in {"db", "dbfs"}
                    and abs(item.control_effect.absolute_delta) >= 3.0
                )
                or (
                    item.control_effect.relative_change_ratio is not None
                    and abs(item.control_effect.relative_change_ratio) >= 0.20
                )
            )
            and all(fact.provenance_source != "public_replay" for fact in item.facts)
        ]
        scores, positive, negative = self._support_state(case, effective)
        ranked = sorted(scores, key=lambda item: scores[item], reverse=True)
        raw_leader = ranked[0] if ranked else None
        raw_runner_up = ranked[1] if len(ranked) > 1 else None
        current_statuses = [item.status for item in case.hypotheses]
        all_weakened = bool(current_statuses) and all(
            status == "weakened" for status in current_statuses
        )
        exact_tie = bool(
            raw_leader
            and raw_runner_up
            and abs(scores[raw_leader] - scores[raw_runner_up]) < 1e-9
        )
        leader = None if all_weakened or exact_tie else raw_leader
        runner_up = raw_runner_up if leader is not None else None
        leading_support = scores.get(leader, 0.5) if leader else 0.5
        runner_up_support = scores.get(runner_up, 0.5) if runner_up else 0.5
        supported_count = sum(status == "supported" for status in current_statuses)
        if all_weakened:
            hypothesis_set_state = "all_weakened"
        elif not any(status in {"supported", "weakened"} for status in current_statuses):
            hypothesis_set_state = "unverified"
        elif exact_tie:
            hypothesis_set_state = "tied"
        elif leader is not None and supported_count:
            hypothesis_set_state = "active_leader"
        else:
            hypothesis_set_state = "mixed"
        tested_ids = {
            assessment.hypothesis_id
            for item in effective
            for assessment in item.hypothesis_assessments
            if task_by_id.get(item.task_id) is not None
            and task_by_id[item.task_id].task_kind != "replication"
            if assessment.status in {"supported", "weakened"}
        }
        coverage = len(tested_ids) / max(len(case.hypotheses), 1)
        contradictions = sum(
            1
            for item in case.evidence
            if item.quality == "high"
            and task_by_id.get(item.task_id) is not None
            and task_by_id[item.task_id].task_kind != "replication"
            for assessment in item.hypothesis_assessments
            if assessment.hypothesis_id == leader and assessment.status == "weakened"
        )
        low_streak = 0
        for item in reversed(case.evidence):
            if item.quality != "low":
                break
            low_streak += 1
        previous_scores, _, _ = self._support_state(case, effective[:-1])
        recent_gain = max(
            (abs(scores[item] - previous_scores.get(item, 0.5)) for item in scores),
            default=0.0,
        )

        checks = {
            "至少两项有效证据": len(effective) >= 2,
            "至少一项符合预期的有效对照": len(matched_controls) >= 1,
            "至少覆盖一类直接传感器": len(distinct_sensors) >= required_sensor_diversity,
            "假设区分覆盖率至少三分之二": coverage >= (2.0 / 3.0),
            "领先假设支持度至少 0.72": leading_support >= 0.72,
            "领先假设与第二名差值至少 0.25": leading_support - runner_up_support >= 0.25,
            "领先假设正向证据权重至少 1.6": positive.get(leader, 0.0) >= 1.6,
            "领先假设没有高质量反向证据": contradictions == 0,
        }
        direct_intervention_ready = bool(direct_interventions)
        # A material intervention effect is a useful fact, not automatically a
        # diagnosis.  It may close the case only when the hypothesis ranking also
        # passes coverage, support, margin, positive-evidence and contradiction gates.
        conclusion_ready = all(checks.values())
        completed_task_count = len(case.completed_tasks)
        soft_checkpoint = (
            not conclusion_ready
            and completed_task_count >= 20
            and not case.continued_after_checkpoint
        )
        hard_stop = not conclusion_ready and completed_task_count >= 32
        forced_stop = hard_stop
        return DiagnosticTerminationVector(
            effective_evidence_count=len(effective),
            effective_control_count=len(controls),
            matched_control_count=len(matched_controls),
            hypothesis_coverage_ratio=round(coverage, 4),
            support_scores=scores,
            leading_hypothesis_id=leader,
            runner_up_hypothesis_id=runner_up,
            leading_support=leading_support,
            runner_up_support=runner_up_support,
            leading_margin=round(leading_support - runner_up_support, 4),
            leading_positive_weight=round(positive.get(leader, 0.0), 4),
            leading_negative_weight=round(negative.get(leader, 0.0), 4),
            high_quality_contradictions=contradictions,
            consecutive_low_quality_count=low_streak,
            recent_information_gain=round(recent_gain, 4),
            hypothesis_set_state=hypothesis_set_state,
            hypothesis_revision_required=all_weakened,
            intervention_effect_ready=direct_intervention_ready,
            conclusion_ready=conclusion_ready,
            forced_stop=forced_stop,
            completed_task_count=completed_task_count,
            distinct_sensor_count=len(distinct_sensors),
            required_sensor_diversity=required_sensor_diversity,
            soft_checkpoint_reached=soft_checkpoint,
            hard_stop_reached=hard_stop,
            user_decision_required=soft_checkpoint,
            blockers=(
                []
                if conclusion_ready
                else (
                    (["候选假设已全部被削弱，需要重规划"] if all_weakened else [])
                    + [label for label, passed in checks.items() if not passed]
                )
            ),
        )

    @staticmethod
    def _build_final_report(case: DiagnosticCase, *, conclusive: bool) -> DiagnosticFinalReport:
        vector = case.termination_vector
        task_by_id = {item.task_id: item for item in case.completed_tasks}
        leader = next(
            (
                item
                for item in case.hypotheses
                if item.hypothesis_id == vector.leading_hypothesis_id
            ),
            None,
        )
        if conclusive and (not vector.conclusion_ready or leader is None):
            raise ValueError("结论报告必须绑定通过严格终止门的唯一领先假设。")
        matched_effect = next(
            (
                item.control_effect
                for item in reversed(case.evidence)
                if item.control_effect
                and item.control_effect.matches_expected_effect
                and task_by_id.get(item.task_id) is not None
                and task_by_id[item.task_id].task_kind == "control"
            ),
            None,
        )
        reasoning = next(
            (
                item.reasoning_receipt
                for item in reversed(case.evidence)
                if item.reasoning_receipt is not None
                and (
                    leader is None
                    or item.reasoning_receipt.ranked_hypothesis_ids[0]
                    == leader.hypothesis_id
                )
            ),
            None,
        )
        if conclusive and leader:
            effect_text = ""
            if matched_effect:
                effect_text = (
                    f"有效对照中 {matched_effect.metric_key} 从 "
                    f"{matched_effect.baseline_value:.4g} {matched_effect.metric_unit} 变为 "
                    f"{matched_effect.current_value:.4g} {matched_effect.metric_unit}。"
                )
            if reasoning is not None:
                conclusion = (
                    f"{reasoning.answer_headline}。{reasoning.mechanism_explanation} {effect_text}"
                ).strip()
                confidence = reasoning.confidence
                mechanism_explanation = reasoning.mechanism_explanation
            else:
                conclusion = (
                    f"在当前实验条件下，证据更支持“{leader.statement}”。"
                    f"{effect_text}该结论仅适用于已记录的设备、测点和控制条件。"
                )
                confidence = "high" if vector.leading_support >= 0.80 else "medium"
                mechanism_explanation = (
                    f"“{leader.statement}”的关键预测是：{leader.critical_prediction}。"
                    "当前受控对照在覆盖竞争解释的同时，使该预测获得最高支持，"
                    "且领先差值与反证门均通过。"
                )
            reason = "终止向量进入结论区域：有效对照、覆盖率、支持度和领先差值均达标。"
        else:
            conclusion = (
                (
                    f"当前最可能的解释是：{reasoning.answer_headline}。"
                    f"{reasoning.mechanism_explanation} 当前证据尚未达到唯一结论门槛，"
                    "因此请把它作为有置信度的首选解释，而不是已确认故障。"
                ).strip()
                if reasoning is not None
                else "当前证据不足以可靠区分候选假设，案例在用户检查点或硬停止门槛处结束。"
            )
            confidence = reasoning.confidence if reasoning is not None else "low"
            mechanism_explanation = reasoning.mechanism_explanation if reasoning else ""
            if vector.bounded_rehearsal_stop:
                reason = (
                    "两条独立公开回放仍未形成可比的现场对照；演练证据边界已经用尽，"
                    "系统停止索取不存在的第三条同类记录，并保留当前排序与现场验证路径。"
                )
            else:
                reason = (
                    "用户在 20 次测量后的检查点选择停止；报告保留当前最可能解释与不确定性。"
                    if not vector.hard_stop_reached
                    else "达到 32 次硬停止边界，系统不再自动增加测量负担。"
                )
        uncertainties = [
            f"{item.hypothesis_id.upper()}：{item.latest_reasoning}"
            for item in case.hypotheses
            if item.hypothesis_id != vector.leading_hypothesis_id
        ]
        solution_plan = build_solution_plan(case, conclusive=conclusive)
        evidence_explanation: list[str] = []
        for evidence in case.evidence:
            if evidence.quality == "low":
                continue
            for fact in evidence.facts:
                value_text = f"{fact.value:.4g} {fact.metric_unit}".strip()
                if fact.baseline_value is None:
                    detail = (
                        f"{sensor_requirement(fact.sensor).label} · {fact.metric_label}为 "
                        f"{value_text}（{fact.quality}质量）"
                    )
                else:
                    ratio_text = ""
                    if fact.relative_delta_ratio is not None:
                        ratio_text = f"，相对变化 {fact.relative_delta_ratio * 100:+.1f}%"
                    relation_text = {
                        "increase": "上升",
                        "decrease": "下降",
                        "within_repeatability": "未分辨到稳定变化",
                    }.get(fact.relation, "变化")
                    detail = (
                        f"{sensor_requirement(fact.sensor).label} · {fact.metric_label}从 "
                        f"{fact.baseline_value:.4g} {fact.metric_unit} 变为 {value_text}，"
                        f"判定为{relation_text}{ratio_text}（{fact.quality}质量）"
                    )
                if fact.analysis_warnings:
                    detail += f"；分析器提醒：{'；'.join(fact.analysis_warnings)}"
                evidence_explanation.append(detail)
        if not evidence_explanation:
            evidence_explanation = [
                "当前没有通过质量门的数值事实；报告只能提供安全观察和升级路径。"
            ]
        confidence_explanation = (
            f"领先解释支持度 {vector.leading_support:.2f}，比第二名高 "
            f"{vector.leading_margin:.2f}；共有 {vector.effective_evidence_count} 条有效证据、"
            f"{vector.matched_control_count} 项方向匹配的受控对照。"
        )
        provenance = {
            fact.provenance_source for evidence in case.evidence for fact in evidence.facts
        }
        if "public_replay" in provenance or "test_fixture" in provenance:
            scope_boundary = (
                "本报告包含公开回放或软件演练证据，只证明 PocketLab 的分析与决策链可运行；"
                "它不能被当作你家现场已经完成的测量。"
            )
        else:
            scope_boundary = (
                "本报告适用于本次记录的设备、测点、手机姿态和工况；环境或设备状态改变后需重新核对。"
            )
        public_only = bool(provenance) and provenance == {"public_replay"}
        if public_only and not conclusive:
            sensor_labels = "、".join(
                sorted({sensor_requirement(item.sensor).label for item in case.evidence})
            )
            headline = (
                f"公开回放已完成{sensor_labels or '传感器'}分析演练，"
                f"但不能判断“{case.title}”的现场原因"
            )
            mechanism_detail = reasoning.mechanism_explanation if reasoning else ""
            conclusion = (
                f"{headline}。{mechanism_detail} 当前排序只描述这些公开记录本身；"
                "要诊断用户现场，仍需按行动计划完成同一测点、同一时长的单变量对照。"
            ).strip()
        else:
            headline = reasoning.answer_headline if reasoning else conclusion
        first_action = solution_plan.actions[0].title if solution_plan.actions else "先保留现场记录"
        user_takeaway = (
            f"{headline}。你现在可以先做“{first_action}”，完成后按报告中的验证方法判断是否有效。"
        )
        return DiagnosticFinalReport(
            outcome=("completed_with_conclusion" if conclusive else "completed_inconclusive"),
            confidence=confidence,
            leading_hypothesis_id=vector.leading_hypothesis_id,
            conclusion=conclusion,
            evidence_basis=[item.evidence_id for item in case.evidence if item.quality != "low"],
            remaining_uncertainties=uncertainties,
            termination_reason=reason,
            vector=vector,
            solution_plan=solution_plan,
            answer_headline=headline,
            mechanism_explanation=mechanism_explanation,
            ranked_hypothesis_ids=reasoning.ranked_hypothesis_ids if reasoning else [],
            source_fact_ids=reasoning.source_fact_ids if reasoning else [],
            user_takeaway=user_takeaway,
            evidence_explanation=evidence_explanation,
            confidence_explanation=confidence_explanation,
            scope_boundary=scope_boundary,
        )


diagnostic_case_store = DiagnosticCaseStore(database, session_store, user_id=None)
