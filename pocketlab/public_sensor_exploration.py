from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pocketlab.public_gyroscope_tools import (
    ANALYZE_TOOL_ID,
    ANCHOR_RECORDING_ID,
    COMPARE_TOOL_ID,
    DATASET_ID,
    TRANSITION_RECORDING_ID,
    load_public_gyroscope_evidence,
)
from pocketlab.public_sensor_agent_models import (
    PublicSensorComparison,
    PublicSensorEvidenceSnapshot,
    PublicSensorEvidenceView,
    PublicSensorExploreRequest,
    PublicSensorExploreResult,
    PublicSensorFinding,
    PublicSensorPlanCandidate,
    PublicSensorPlannerDecision,
    PublicSensorPlannerRequest,
    PublicSensorPlannerTrace,
    PublicSensorReport,
    PublicSensorReportSource,
    PublicSensorRuntimeTrace,
)

_PROTOCOL_ID = "gyroscope-public-exploration.v1"
_PROTOCOL_VERSION = "1.0.0"

_STATIONARY_ID = "inspect_stationary_gyroscope"
_HANDHELD_ID = "inspect_handheld_gyroscope"
_PAIR_ID = "compare_stationary_handheld_gyroscope"
_LIVE_ID = "request_live_gyroscope"
_UNSUPPORTED_ID = "stop_unsupported_gyroscope"
_FINISH_ID = "finish_gyroscope_report"
_PRIVACY_ID = "privacy_acknowledgement_required"

_SELECTION_POLICY = (
    "静止零偏、静置噪声或桌面不动问题选择 inspect_stationary_gyroscope。",
    "仅询问公开手持转动中是否有明显角运动可选择 inspect_handheld_gyroscope。",
    "比较静止与运动、验证传感器响应或未明确单一状态时选择 compare_stationary_handheld_gyroscope。",
    "询问用户当前手机、此刻现场或新实验读数时选择 request_live_gyroscope。",
    "精确转角、绝对姿态、航向、设备校准或候选外请求选择 stop_unsupported_gyroscope。",
)

_SAFE_RUNTIME_KEYS = frozenset(
    {
        "run_id",
        "operation",
        "model",
        "status",
        "elapsed_s",
        "timeout_s",
        "max_turns",
        "retry_limit",
        "model_requests",
        "tool_calls",
        "tool_events",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "usage_reported",
        "token_budget",
        "token_budget_exceeded",
        "error_kind",
        "error_type",
        "transport",
        "transport_fallback_reason",
    }
)


class PublicSensorPlannerOutcome(Protocol):
    decision: PublicSensorPlannerDecision
    runtime_trace: dict[str, Any]


PlannerCallable = Callable[
    [PublicSensorPlannerRequest], Awaitable[PublicSensorPlannerOutcome]
]


class PublicSensorExplorationUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"公开传感器闭环暂不可用（{reason}）。")
        self.reason = reason


class _PlannerDecisionRejected(ValueError):
    pass


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _gyroscope_question_family(question: str) -> str:
    lower = question.casefold()
    if _contains_any(
        lower,
        (
            "ignore previous",
            "ignore all",
            "system prompt",
            "developer message",
            "api key",
            "apikey",
            "读取密钥",
            "泄露提示",
            "忽略之前",
            "越过候选",
            "修改文件",
        ),
    ):
        return "unsupported"
    if _contains_any(
        lower,
        (
            "绝对姿态",
            "绝对方向",
            "航向",
            "heading",
            "orientation",
            "精确转角",
            "旋转了多少度",
            "calibration",
            "校准系数",
            "市场验证",
        ),
    ):
        return "unsupported"
    if _contains_any(
        lower,
        (
            "我的手机",
            "当前手机",
            "现在手机",
            "此刻",
            "现场",
            "刚才采集",
            "live",
            "real time",
            "realtime",
            "真机数据",
        ),
    ):
        return "live"
    if _contains_any(
        lower,
        (
            "比较",
            "区分",
            "对照",
            "两种状态",
            "静止和",
            "静止与",
            "stationary versus",
            "compare",
            "distinguish",
            "contrast",
        ),
    ):
        return "pair"
    if _contains_any(
        lower,
        (
            "静止",
            "静置",
            "不动",
            "零偏",
            "bias",
            "idle",
            "stationary",
            "resting",
        ),
    ):
        return "stationary"
    if _contains_any(
        lower,
        (
            "手持",
            "转动",
            "旋转响应",
            "角运动",
            "gyroscope response",
            "handheld",
            "angular motion",
        ),
    ):
        return "pair"
    return "pair"


def _initial_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id=_STATIONARY_ID,
            title="检查公开静止锚点与零偏候选",
            server_reason="只分析已注册的 NIST AS7 静止窗口，判断角速度是否处在预注册低值门内。",
            rationale_code="match_stationary_bias_goal",
            recording_ids=(ANCHOR_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="stationary_bias_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_HANDHELD_ID,
            title="检查公开手持转动响应",
            server_reason="只分析已注册的 NIST AS7 手持窗口，判断是否存在明显角运动响应。",
            rationale_code="match_handheld_response_goal",
            recording_ids=(TRANSITION_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="handheld_response_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_PAIR_ID,
            title="比较静止与手持角运动",
            server_reason="分析同一公开 acquisition 的静止和手持窗口，用预注册比值门检查状态分离。",
            rationale_code="compare_motion_states",
            recording_ids=(ANCHOR_RECORDING_ID, TRANSITION_RECORDING_ID),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="paired_motion_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox Gyroscope 真机测量",
            server_reason="公开数据不能回答当前用户手机或现场状态，转入受控真机测量建议。",
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止超出证据边界的请求",
            server_reason="精确角度、绝对姿态、航向、校准或越权请求不在当前分析器和证据范围内。",
            rationale_code="unsupported_claim_boundary",
            terminal=True,
            result_code="unsupported",
        ),
    )


def _follow_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id=_FINISH_ID,
            title="形成有边界的 Gyroscope 报告",
            server_reason="服务端确定性质量门已通过时，可以基于公开证据形成受限结论。",
            rationale_code="evidence_quality_sufficient",
            terminal=True,
            result_code="finish_report",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox Gyroscope 真机复核",
            server_reason="质量门不足或问题指向当前手机时，需要真机证据而不是扩大公开数据结论。",
            rationale_code="evidence_quality_insufficient",
            terminal=True,
            result_code="live_measurement_required",
        ),
    )


def _initial_fallback(family: str) -> str:
    return {
        "stationary": _STATIONARY_ID,
        "pair": _PAIR_ID,
        "live": _LIVE_ID,
        "unsupported": _UNSUPPORTED_ID,
    }[family]


def _planner_request(
    *,
    run_id: str,
    step: int,
    operation: str,
    request: PublicSensorExploreRequest,
    evidence_view: PublicSensorEvidenceView,
    candidates: tuple[PublicSensorPlanCandidate, ...],
    fallback_candidate_id: str,
    protocol_id: str = _PROTOCOL_ID,
    protocol_version: str = _PROTOCOL_VERSION,
    selection_policy: tuple[str, ...] = _SELECTION_POLICY,
) -> PublicSensorPlannerRequest:
    unsigned = {
        "sensor": request.sensor,
        "protocol_id": protocol_id,
        "protocol_version": protocol_version,
        "operation": operation,
        "run_id": run_id,
        "step": step,
        "research_question_untrusted": request.research_question,
        "privacy_acknowledged": request.privacy_acknowledged,
        "selection_policy": selection_policy,
        "evidence_view": evidence_view.model_dump(mode="json"),
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "fallback_candidate_id": fallback_candidate_id,
    }
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PublicSensorPlannerRequest(
        sensor=request.sensor,
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        operation=operation,  # type: ignore[arg-type]
        run_id=run_id,
        step=step,
        request_sha256=digest,
        research_question_untrusted=request.research_question,
        privacy_acknowledged=request.privacy_acknowledged,
        selection_policy=selection_policy,
        evidence_view=evidence_view,
        candidates=candidates,
        fallback_candidate_id=fallback_candidate_id,
    )


def _safe_runtime_trace(value: object) -> PublicSensorRuntimeTrace | None:
    if not isinstance(value, dict):
        return None
    safe: dict[str, object] = {
        key: item
        for key, item in value.items()
        if key in _SAFE_RUNTIME_KEYS
        and (item is None or isinstance(item, (str, int, float, bool)))
    }
    tool_events = value.get("tool_events")
    if isinstance(tool_events, list):
        safe_events: list[dict[str, str]] = []
        for item in tool_events[:4]:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            status = item.get("status")
            if isinstance(name, str) and isinstance(status, str):
                safe_events.append({"name": name[:120], "status": status[:40]})
        safe["tool_events"] = tuple(safe_events)
    return PublicSensorRuntimeTrace.model_validate(safe) if safe else None


def _fallback_reason(error: Exception) -> str:
    value = getattr(error, "reason", None)
    if not isinstance(value, str) or not value:
        value = "planner-unavailable"
    value = re.sub(r"[^a-z0-9-]+", "-", value.casefold()).strip("-")
    return (value or "planner-unavailable")[:100]


async def _select_candidate(
    request: PublicSensorPlannerRequest,
    *,
    planner: PlannerCallable | None,
) -> tuple[str, PublicSensorPlannerTrace]:
    candidate_ids = tuple(item.candidate_id for item in request.candidates)
    fallback_id = request.fallback_candidate_id
    try:
        active_planner = planner
        if active_planner is None:
            from pocketlab.public_sensor_planner import run_public_sensor_planner

            active_planner = run_public_sensor_planner
        outcome = await active_planner(request)
        decision = getattr(outcome, "decision", None)
        runtime_trace = getattr(outcome, "runtime_trace", None)
        if not isinstance(decision, PublicSensorPlannerDecision):
            raise _PlannerDecisionRejected
        selected = next(
            (item for item in request.candidates if item.candidate_id == decision.selected_candidate_id),
            None,
        )
        if (
            decision.run_id != request.run_id
            or decision.step != request.step
            or decision.request_sha256 != request.request_sha256
            or selected is None
            or decision.rationale_code != selected.rationale_code
        ):
            raise _PlannerDecisionRejected
        return decision.selected_candidate_id, PublicSensorPlannerTrace(
            step=request.step,
            operation=request.operation,
            request_sha256=request.request_sha256,
            candidate_ids=candidate_ids,
            selected_candidate_id=decision.selected_candidate_id,
            fallback_candidate_id=fallback_id,
            rationale_code=decision.rationale_code,
            source="agent",
            outcome="accepted",
            transport=(
                str(runtime_trace.get("transport"))
                if isinstance(runtime_trace, dict)
                and runtime_trace.get("transport")
                in {"function_tool", "validated_json_text"}
                else "not_attempted"
            ),
            runtime_trace=_safe_runtime_trace(runtime_trace),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        fallback = next(item for item in request.candidates if item.candidate_id == fallback_id)
        return fallback_id, PublicSensorPlannerTrace(
            step=request.step,
            operation=request.operation,
            request_sha256=request.request_sha256,
            candidate_ids=candidate_ids,
            selected_candidate_id=fallback_id,
            fallback_candidate_id=fallback_id,
            rationale_code=fallback.rationale_code,
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason=_fallback_reason(exc),
            runtime_trace=_safe_runtime_trace(getattr(exc, "runtime_trace", None)),
        )


def _planner_status(traces: Sequence[PublicSensorPlannerTrace]) -> str:
    sources = {item.source for item in traces}
    if sources == {"agent"}:
        return "accepted"
    if sources == {"strong_workflow_fallback"}:
        return "fallback"
    return "mixed"


def _privacy_result(
    request: PublicSensorExploreRequest,
    *,
    run_id: str,
) -> PublicSensorExploreResult:
    candidates = (_PRIVACY_ID, _UNSUPPORTED_ID)
    digest = hashlib.sha256(
        json.dumps(
            {
                "run_id": run_id,
                "sensor": request.sensor,
                "question": request.research_question,
                "privacy_acknowledged": False,
                "candidate_ids": candidates,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    trace = PublicSensorPlannerTrace(
        step=1,
        operation="select_evidence_route",
        request_sha256=digest,
        candidate_ids=candidates,
        selected_candidate_id=_PRIVACY_ID,
        fallback_candidate_id=_PRIVACY_ID,
        rationale_code="privacy_not_acknowledged",
        source="strong_workflow_fallback",
        outcome="fallback",
        fallback_reason="privacy-not-acknowledged",
    )
    report = PublicSensorReport(
        conclusion_kind="privacy_acknowledgement_required",
        title="需要确认本地公开 Gyroscope 回放边界",
        summary="确认前不会读取公开序列、调用 Planner、执行分析或写入账号。",
        uncertainties=("尚未读取任何公开 Gyroscope 证据。",),
        forbidden_claims=("不能把 NIST Pixel XL 数据冒充为你的 phyphox 真机数据。",),
        next_live_measurement="确认后可运行本地公开回放；用户真机 Gate C 仍需单独采集。",
    )
    return PublicSensorExploreResult(
        sensor=request.sensor,
        protocol_id=_PROTOCOL_ID,
        run_id=run_id,
        research_question=request.research_question,
        execution_status="limited",
        selected_route_id=_PRIVACY_ID,
        planner_status="fallback",
        planner_trace=(trace,),
        report=report,
    )


def _terminal_report(family: str) -> PublicSensorReport:
    if family == "live":
        return PublicSensorReport(
            conclusion_kind="live_measurement_required",
            title="这个问题需要你的手机 Gyroscope 数据",
            summary="公开 Pixel XL 回放不能回答当前手机或现场角运动状态，Agent 已停止证据替代。",
            uncertainties=("尚无当前手机的静止基线、受控转动条件和重复测量。",),
            forbidden_claims=("不得用公开记录推断你的手机此刻角速度或设备状态。",),
            next_live_measurement=(
                "在 phyphox 打开 Gyroscope：固定手机静置 5 秒，再按同一动作缓慢转动 5 秒；"
                "保持安装方向与动作幅度，并至少重复 3 次。"
            ),
        )
    return PublicSensorReport(
        conclusion_kind="unsupported",
        title="问题超出当前 Gyroscope 证据边界",
        summary="当前 Beta 只分析角速度模长、静止零偏候选和静止—手持状态分离，不能给出该请求。",
        uncertainties=("缺少标定转台、绝对姿态参考或当前设备真机证据。",),
        forbidden_claims=(
            "不得从当前公开切片积分并声称精确转角。",
            "不得声称绝对姿态、航向、设备校准、Gate C 或市场有效性。",
        ),
        next_live_measurement=(
            "若目标是角运动响应，可用 phyphox Gyroscope 采集静止基线与受控转动对照。"
        ),
    )


def _report_source(evidence: PublicSensorEvidenceSnapshot) -> PublicSensorReportSource:
    return PublicSensorReportSource(
        dataset_id=evidence.dataset_id,
        data_class=evidence.data_class,
        device_scope=evidence.device_scope,
        source_title=evidence.source_title,
        source_url=evidence.source_url,
        doi=evidence.doi,
        license_spdx=evidence.license_spdx,
    )


def _evidence_report(
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
    comparison: PublicSensorComparison,
    *,
    selected_action: str,
) -> PublicSensorReport:
    findings: list[PublicSensorFinding] = []
    metric_by_key = {item.key: item for item in comparison.metrics}
    if "stationary_bias_candidate_rad_s" in metric_by_key:
        value = metric_by_key["stationary_bias_candidate_rad_s"].value
        findings.append(
            PublicSensorFinding(
                finding_id="stationary-bias-candidate",
                text=f"公开静止窗口的零偏候选为 {value:.6f} rad/s。",
                evidence_ids=(evidence[0].evidence_id,),
            )
        )
    if "handheld_mean_rad_s" in metric_by_key:
        item = metric_by_key["handheld_mean_rad_s"]
        handheld = next(
            evidence_item
            for evidence_item in evidence
            if evidence_item.recording_id == TRANSITION_RECORDING_ID
        )
        findings.append(
            PublicSensorFinding(
                finding_id="handheld-angular-response",
                text=f"公开手持窗口的平均角速度模长为 {item.value:.6f} rad/s。",
                evidence_ids=(handheld.evidence_id,),
            )
        )
    if "motion_to_stationary_ratio" in metric_by_key:
        value = metric_by_key["motion_to_stationary_ratio"].value
        findings.append(
            PublicSensorFinding(
                finding_id="motion-state-separation",
                text=f"手持与静止平均角速度模长之比为 {value:.1f}，通过预注册状态分离检查。",
                evidence_ids=tuple(item.evidence_id for item in evidence),
            )
        )

    supported = selected_action == _FINISH_ID and comparison.quality_passed
    sources = (_report_source(evidence[0]),)
    if supported:
        conclusion = "supported_with_limits"
        title = "公开 Gyroscope 证据支持有边界的角运动响应结论"
        summary = (
            "Agent 在冻结候选中选择了与问题匹配的 NIST 公开手机记录；服务端专用分析器与"
            "预注册质量门已完成。结果只支持该记录中的静止零偏候选或角运动响应，不是用户真机结论。"
        )
    else:
        conclusion = "live_measurement_required"
        title = "公开 Gyroscope 证据不足，需要真机复核"
        summary = "服务端质量门未通过或 Planner 选择了真机复核，没有扩大公开数据结论。"
    return PublicSensorReport(
        conclusion_kind=conclusion,
        title=title,
        summary=summary,
        supported_findings=tuple(findings),
        uncertainties=(
            "两段切片来自同一台 Pixel XL、同一次 AS7 acquisition，不能视为独立手机重复。",
            "公开数据由 NIST 自定义 Android 应用采集，不是 phyphox 导出。",
            "未使用标定转台，不能验证角速度刻度或积分角度精度。",
        ),
        forbidden_claims=(
            "不得从这些切片声称精确转角、绝对姿态或航向。",
            "不得把公开回放计入用户真机 Gate C、市场验证或 production agent_ready。",
        ),
        next_live_measurement=(
            "最后一步是真机 Gate C：phyphox Gyroscope 中固定手机静置 5 秒，再执行受控缓慢转动 "
            "5 秒；保持安装与动作条件并至少重复 3 次。"
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        source_ids=(evidence[0].dataset_id,),
        sources=sources,
    )


async def run_public_sensor_exploration(
    request: PublicSensorExploreRequest,
    *,
    root: Path,
    planner: PlannerCallable | None = None,
) -> PublicSensorExploreResult:
    """Dispatch one stateless bounded public-sensor loop."""

    if request.sensor == "accelerometer":
        from pocketlab.public_accelerometer_exploration import (
            run_public_accelerometer_exploration,
        )

        return await run_public_accelerometer_exploration(
            request,
            root=root,
            planner=planner,
        )

    if request.sensor == "magnetometer":
        from pocketlab.public_magnetometer_exploration import (
            run_public_magnetometer_exploration,
        )

        return await run_public_magnetometer_exploration(
            request,
            root=root,
            planner=planner,
        )

    if request.sensor == "proximity":
        from pocketlab.public_proximity_exploration import (
            run_public_proximity_exploration,
        )

        return await run_public_proximity_exploration(
            request,
            root=root,
            planner=planner,
        )

    if request.sensor == "microphone":
        from pocketlab.public_microphone_exploration import (
            run_public_microphone_exploration,
        )

        return await run_public_microphone_exploration(
            request,
            root=root,
            planner=planner,
        )

    if request.sensor == "location":
        from pocketlab.public_location_exploration import (
            run_public_location_exploration,
        )

        return await run_public_location_exploration(
            request,
            root=root,
            planner=planner,
        )

    if request.sensor != "gyroscope":
        raise PublicSensorExplorationUnavailable("sensor-protocol-not-yet-registered")
    run_id = f"public-{request.sensor}-{uuid4().hex}"
    if not request.privacy_acknowledged:
        return _privacy_result(request, run_id=run_id)

    family = _gyroscope_question_family(request.research_question)
    initial_candidates = _initial_candidates()
    first_request = _planner_request(
        run_id=run_id,
        step=1,
        operation="select_evidence_route",
        request=request,
        evidence_view=PublicSensorEvidenceView(),
        candidates=initial_candidates,
        fallback_candidate_id=_initial_fallback(family),
    )
    if family in {"live", "unsupported"}:
        selected_id = first_request.fallback_candidate_id
        candidate = next(
            item for item in initial_candidates if item.candidate_id == selected_id
        )
        first_trace = PublicSensorPlannerTrace(
            step=1,
            operation="select_evidence_route",
            request_sha256=first_request.request_sha256,
            candidate_ids=tuple(item.candidate_id for item in initial_candidates),
            selected_candidate_id=selected_id,
            fallback_candidate_id=selected_id,
            rationale_code=candidate.rationale_code,
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason="policy-boundary",
        )
        report = _terminal_report(family)
        return PublicSensorExploreResult(
            sensor=request.sensor,
            protocol_id=_PROTOCOL_ID,
            run_id=run_id,
            research_question=request.research_question,
            execution_status="unsupported" if family == "unsupported" else "limited",
            selected_route_id=selected_id,
            planner_status="fallback",
            planner_trace=(first_trace,),
            report=report,
        )

    selected_id, first_trace = await _select_candidate(first_request, planner=planner)
    selected_candidate = next(
        item for item in initial_candidates if item.candidate_id == selected_id
    )
    if selected_candidate.terminal:
        terminal_family = "live" if selected_id == _LIVE_ID else "unsupported"
        report = _terminal_report(terminal_family)
        return PublicSensorExploreResult(
            sensor=request.sensor,
            protocol_id=_PROTOCOL_ID,
            run_id=run_id,
            research_question=request.research_question,
            execution_status=(
                "unsupported" if terminal_family == "unsupported" else "limited"
            ),
            selected_route_id=selected_id,
            planner_status=_planner_status((first_trace,)),
            planner_trace=(first_trace,),
            report=report,
        )

    try:
        evidence, comparison, tool_trace = load_public_gyroscope_evidence(
            root.resolve() / DATASET_ID,
            selected_candidate.recording_ids,
        )
    except (OSError, StopIteration, ValueError) as exc:
        raise PublicSensorExplorationUnavailable(
            "source-or-tool-validation-failed"
        ) from exc

    follow_candidates = _follow_candidates()
    follow_fallback = _FINISH_ID if comparison.quality_passed else _LIVE_ID
    minimum_confidence = (
        "low"
        if any(item.analysis.confidence == "low" for item in evidence)
        else "medium"
        if any(item.analysis.confidence == "medium" for item in evidence)
        else "high"
    )
    follow_request = _planner_request(
        run_id=run_id,
        step=2,
        operation="select_report_action",
        request=request,
        evidence_view=PublicSensorEvidenceView(
            evidence_ids=tuple(item.evidence_id for item in evidence),
            confidence=minimum_confidence,
            quality_passed=comparison.quality_passed,
            result_codes=comparison.result_codes,
        ),
        candidates=follow_candidates,
        fallback_candidate_id=follow_fallback,
    )
    follow_candidate = next(
        item
        for item in follow_candidates
        if item.candidate_id == follow_request.fallback_candidate_id
    )
    follow_id = follow_candidate.candidate_id
    follow_trace = PublicSensorPlannerTrace(
        step=2,
        operation="select_report_action",
        request_sha256=follow_request.request_sha256,
        candidate_ids=tuple(item.candidate_id for item in follow_candidates),
        selected_candidate_id=follow_id,
        fallback_candidate_id=follow_id,
        rationale_code=follow_candidate.rationale_code,
        source="strong_workflow_fallback",
        outcome="fallback",
        fallback_reason="server-owned-termination",
    )
    traces = (first_trace, follow_trace)
    report = _evidence_report(
        evidence,
        comparison,
        selected_action=follow_id,
    )
    return PublicSensorExploreResult(
        sensor=request.sensor,
        protocol_id=_PROTOCOL_ID,
        run_id=run_id,
        research_question=request.research_question,
        execution_status=(
            "completed"
            if report.conclusion_kind == "supported_with_limits"
            else "limited"
        ),
        selected_route_id=follow_id,
        planner_status=_planner_status(traces),
        planner_trace=traces,
        tool_trace=tool_trace,
        evidence=evidence,
        comparison=comparison,
        report=report,
    )


def strong_public_sensor_workflow_route(sensor: str, question: str) -> tuple[str, str]:
    """Expose the frozen strong baseline route for paired Harness evaluation."""

    if sensor == "magnetometer":
        from pocketlab.public_magnetometer_exploration import (
            strong_magnetometer_workflow_route,
        )

        return strong_magnetometer_workflow_route(question)
    if sensor == "proximity":
        from pocketlab.public_proximity_exploration import (
            strong_proximity_workflow_route,
        )

        return strong_proximity_workflow_route(question)
    if sensor == "microphone":
        from pocketlab.public_microphone_exploration import (
            strong_microphone_workflow_route,
        )

        return strong_microphone_workflow_route(question)
    if sensor == "location":
        from pocketlab.public_location_exploration import (
            strong_location_workflow_route,
        )

        return strong_location_workflow_route(question)
    if sensor != "gyroscope":
        return "unsupported", _UNSUPPORTED_ID
    family = _gyroscope_question_family(question)
    return family, _initial_fallback(family)
