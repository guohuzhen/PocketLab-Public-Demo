from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pocketlab.public_pressure_agent_models import (
    PublicPressureEvidenceSnapshot,
    PublicPressureEvidenceView,
    PublicPressureExploreRequest,
    PublicPressureExploreResult,
    PublicPressureFinding,
    PublicPressurePlanCandidate,
    PublicPressurePlannerDecision,
    PublicPressurePlannerRequest,
    PublicPressurePlannerTrace,
    PublicPressureReport,
    PublicPressureReportSource,
    PublicPressureRuntimeTrace,
    PublicPressureToolExecution,
)
from pocketlab.public_pressure_models import PublicPressureTraceResult
from pocketlab.public_pressure_replay import (
    PublicPressureReplayError,
    load_verified_public_pressure_evidence,
)
from pocketlab.public_pressure_tools import (
    audit_pressure_claim_support,
    compare_pressure_height_to_ground_truth,
    inspect_pressure_trace,
)
from pocketlab.public_replay_dataset import load_public_replay_dataset

_DATASET_ID = "pressure-nist-perfloc-pixel-20180516-v1"
_ELEVATOR_RECORDING_ID = "as5-elevator-stable-ascent"
_STAIR_RECORDING_ID = "as4-stairwell-stable-ascent"

_ELEVATOR_ID = "analyze_elevator_ascent"
_STAIR_ID = "analyze_stairwell_ascent"
_LIVE_ID = "request_live_pressure"
_UNSUPPORTED_ID = "stop_unsupported"
_FINISH_ID = "finish_relative_height_report"
_PRIVACY_ID = "privacy_acknowledgement_required"

_INSPECT_TOOL = "inspect_pressure_trace"
_COMPARE_TOOL = "compare_pressure_height_to_ground_truth"
_AUDIT_TOOL = "audit_pressure_claim_support"

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


class PublicPressurePlannerOutcome(Protocol):
    decision: PublicPressurePlannerDecision
    runtime_trace: dict[str, Any]


PlannerCallable = Callable[
    [PublicPressurePlannerRequest], Awaitable[PublicPressurePlannerOutcome]
]


class PublicPressureExplorationUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"公开 Pressure 闭环暂不可用（{reason}）。")
        self.reason = reason


class _PlannerDecisionRejected(ValueError):
    pass


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _question_family(question: str) -> str:
    lower = question.casefold()
    if _contains_any(
        lower,
        (
            "ignore previous",
            "ignore all",
            "system prompt",
            "api key",
            "shell",
            "higher priority",
            "override the protocol",
            "bypass the protocol",
            "ignore the protocol",
            "unlisted action",
            "expose secrets",
            "reveal secrets",
            "读取密钥",
            "忽略规则",
            "系统提示词",
            "执行命令",
            "高于协议",
            "绕过协议",
            "未列出的动作",
            "泄露秘密",
            "暴露密钥",
        ),
    ):
        return "unsupported"
    if _contains_any(
        lower,
        (
            "absolute altitude",
            "exact altitude",
            "绝对海拔",
            "精确海拔",
            "calibrate",
            "校准气压计",
            "market validation",
            "市场验证",
            "prove vertical motion",
            "证明一定是",
            "weather caused",
            "证明天气",
        ),
    ):
        return "unsupported"
    if _contains_any(
        lower,
        (
            "my phone",
            "right now",
            "current floor",
            "我的手机",
            "当前楼层",
            "现在几楼",
            "我现在",
        ),
    ):
        return "live"
    if _contains_any(lower, ("stair", "stairs", "stairwell", "楼梯", "爬楼")):
        return "stairwell"
    return "elevator"


def _evidence_candidate(
    *,
    candidate_id: str,
    recording_id: str,
    title: str,
    reason: str,
    result_code: str,
) -> PublicPressurePlanCandidate:
    return PublicPressurePlanCandidate(
        candidate_id=candidate_id,
        title=title,
        server_reason=reason,
        recording_id=recording_id,
        tool_ids=(_INSPECT_TOOL, _COMPARE_TOOL, _AUDIT_TOOL),
        result_code=result_code,
    )


def _terminal_candidate(
    candidate_id: str,
    title: str,
    reason: str,
    result_code: str,
) -> PublicPressurePlanCandidate:
    return PublicPressurePlanCandidate(
        candidate_id=candidate_id,
        title=title,
        server_reason=reason,
        terminal=True,
        result_code=result_code,
    )


def _initial_candidates() -> tuple[PublicPressurePlanCandidate, ...]:
    return (
        _evidence_candidate(
            candidate_id=_ELEVATOR_ID,
            recording_id=_ELEVATOR_RECORDING_ID,
            title="分析公开电梯上升记录",
            reason="使用 NIST PerfLoc AS5 电梯上升的压力序列验证稳定端点相对高度。",
            result_code="elevator_public_evidence",
        ),
        _evidence_candidate(
            candidate_id=_STAIR_ID,
            recording_id=_STAIR_RECORDING_ID,
            title="分析公开楼梯上升记录",
            reason="使用 NIST PerfLoc AS4 楼梯上升的压力序列验证稳定端点相对高度。",
            result_code="stairwell_public_evidence",
        ),
        _terminal_candidate(
            _LIVE_ID,
            "请求当前手机 Pressure 测量",
            "公开记录不能回答用户此刻手机所在楼层或当前环境压力。",
            "live_pressure_required",
        ),
        _terminal_candidate(
            _UNSUPPORTED_ID,
            "拒绝超出相对压力边界的结论",
            "单一公开压力序列不能提供绝对海拔、单因果证明、设备校准或市场验证。",
            "unsupported_pressure_claim",
        ),
    )


def _initial_fallback(family: str) -> str:
    return {
        "stairwell": _STAIR_ID,
        "live": _LIVE_ID,
        "unsupported": _UNSUPPORTED_ID,
    }.get(family, _ELEVATOR_ID)


def _follow_up_candidates() -> tuple[PublicPressurePlanCandidate, ...]:
    return (
        _terminal_candidate(
            _FINISH_ID,
            "形成有边界的相对高度报告",
            "稳定端点和公开来源允许报告压力变化及标准大气近似相对高度。",
            "relative_height_report",
        ),
        _terminal_candidate(
            _LIVE_ID,
            "请求当前手机 Pressure 复核",
            "证据质量不足或问题要求当前设备数据时，必须转入真机采集。",
            "live_pressure_required",
        ),
    )


def _evidence_view(
    inspection: PublicPressureTraceResult | None,
) -> PublicPressureEvidenceView:
    if inspection is None:
        return PublicPressureEvidenceView()
    warning_codes = [
        "PLATFORMS_GOOD" if inspection.platforms_passed else "PLATFORM_UNSTABLE",
        f"CONFIDENCE_{inspection.confidence.upper()}",
    ]
    if inspection.warnings:
        warning_codes.append("WARNINGS_PRESENT")
    return PublicPressureEvidenceView(
        evidence_id=f"pressure-{inspection.candidate_id}",
        candidate_id=inspection.candidate_id,
        confidence=inspection.confidence,
        platforms_passed=inspection.platforms_passed,
        pressure_direction=inspection.pressure_direction,
        approximate_height_change_m=round(
            inspection.standard_atmosphere_height_change_m, 6
        ),
        warning_codes=tuple(warning_codes),
    )


def _planner_request(
    *,
    run_id: str,
    step: int,
    operation: str,
    request: PublicPressureExploreRequest,
    inspection: PublicPressureTraceResult | None,
    candidates: tuple[PublicPressurePlanCandidate, ...],
    fallback_candidate_id: str,
) -> PublicPressurePlannerRequest:
    evidence_view = _evidence_view(inspection)
    unsigned_json = {
        "schema_version": "1.0",
        "operation": operation,
        "run_id": run_id,
        "step": step,
        "research_question_untrusted": request.research_question,
        "privacy_acknowledged": request.privacy_acknowledged,
        "evidence_view": evidence_view.model_dump(mode="json"),
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "fallback_candidate_id": fallback_candidate_id,
    }
    digest = hashlib.sha256(
        json.dumps(
            unsigned_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PublicPressurePlannerRequest(
        operation=operation,
        run_id=run_id,
        step=step,
        request_sha256=digest,
        research_question_untrusted=request.research_question,
        privacy_acknowledged=request.privacy_acknowledged,
        evidence_view=evidence_view,
        candidates=candidates,
        fallback_candidate_id=fallback_candidate_id,
    )


def _allowed_rationales(candidate_id: str) -> frozenset[str]:
    return {
        _ELEVATOR_ID: frozenset({"match_elevator_goal"}),
        _STAIR_ID: frozenset({"match_stairwell_goal"}),
        _LIVE_ID: frozenset(
            {"request_live_device_evidence", "evidence_quality_insufficient"}
        ),
        _UNSUPPORTED_ID: frozenset({"unsupported_claim_boundary"}),
        _FINISH_ID: frozenset({"evidence_quality_sufficient"}),
        _PRIVACY_ID: frozenset({"privacy_not_acknowledged"}),
    }[candidate_id]


def _safe_runtime_trace(value: object) -> PublicPressureRuntimeTrace | None:
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
    return PublicPressureRuntimeTrace.model_validate(safe) if safe else None


def _fallback_reason(error: Exception) -> str:
    value = getattr(error, "reason", None)
    if not isinstance(value, str) or not value:
        value = "planner-unavailable"
    value = re.sub(r"[^a-z0-9-]+", "-", value.casefold()).strip("-")
    return (value or "planner-unavailable")[:100]


async def _select_candidate(
    request: PublicPressurePlannerRequest,
    *,
    planner: PlannerCallable | None,
) -> tuple[str, PublicPressurePlannerTrace]:
    candidate_ids = tuple(item.candidate_id for item in request.candidates)
    fallback_id = request.fallback_candidate_id
    active_planner = planner
    try:
        if active_planner is None:
            from pocketlab.public_pressure_planner import run_public_pressure_planner

            active_planner = run_public_pressure_planner
        outcome = await active_planner(request)
        decision = getattr(outcome, "decision", None)
        runtime_trace = getattr(outcome, "runtime_trace", None)
        if not isinstance(decision, PublicPressurePlannerDecision):
            raise _PlannerDecisionRejected
        if (
            decision.run_id != request.run_id
            or decision.step != request.step
            or decision.request_sha256 != request.request_sha256
            or decision.selected_candidate_id not in candidate_ids
            or decision.rationale_code
            not in _allowed_rationales(decision.selected_candidate_id)
        ):
            raise _PlannerDecisionRejected
        return decision.selected_candidate_id, PublicPressurePlannerTrace(
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
        return fallback_id, PublicPressurePlannerTrace(
            step=request.step,
            operation=request.operation,
            request_sha256=request.request_sha256,
            candidate_ids=candidate_ids,
            selected_candidate_id=fallback_id,
            fallback_candidate_id=fallback_id,
            rationale_code="strong_workflow_fallback",
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason=_fallback_reason(exc),
            runtime_trace=_safe_runtime_trace(getattr(exc, "runtime_trace", None)),
        )


def _privacy_result(
    request: PublicPressureExploreRequest,
    *,
    run_id: str,
) -> PublicPressureExploreResult:
    candidate_ids = (_PRIVACY_ID, _UNSUPPORTED_ID)
    digest = hashlib.sha256(
        json.dumps(
            {
                "run_id": run_id,
                "question": request.research_question,
                "privacy_acknowledged": False,
                "candidate_ids": candidate_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    trace = PublicPressurePlannerTrace(
        step=1,
        operation="select_evidence_route",
        request_sha256=digest,
        candidate_ids=candidate_ids,
        selected_candidate_id=_PRIVACY_ID,
        fallback_candidate_id=_PRIVACY_ID,
        rationale_code="privacy_not_acknowledged",
        source="strong_workflow_fallback",
        outcome="fallback",
        fallback_reason="privacy-not-acknowledged",
    )
    report = PublicPressureReport(
        conclusion_kind="privacy_acknowledgement_required",
        title="需要确认本地公开 Pressure 回放边界",
        summary=(
            "该数据包只能在本机进行公开数据回放；确认前不会读取序列、调用 Planner 或写入账号。"
        ),
        uncertainties=("尚未读取任何公开 Pressure 证据。",),
        forbidden_claims=("不能把公开 NIST 数据冒充为你手机刚刚采集的 phyphox 数据。",),
        next_live_measurement="确认边界后可运行公开回放；真机测量仍需单独打开 phyphox Pressure。",
    )
    return PublicPressureExploreResult(
        run_id=run_id,
        research_question=request.research_question,
        execution_status="limited",
        selected_route_id=_PRIVACY_ID,
        planner_status="fallback",
        planner_trace=(trace,),
        report=report,
    )


def _empty_report(family: str) -> PublicPressureReport:
    if family == "live":
        return PublicPressureReport(
            conclusion_kind="live_measurement_required",
            title="这个问题需要你的手机 Pressure 数据",
            summary=(
                "公开回放不能回答你此刻所在楼层或当前设备状态；Agent 已停止公开证据替代。"
            ),
            uncertainties=("尚无当前手机的起点、终点和回程压力平台。",),
            forbidden_claims=("不得用公开 Pixel XL 记录推断你当前手机所在楼层。",),
            next_live_measurement=(
                "在 phyphox 打开 Pressure：起点静置至少 3 秒，移动到目标楼层后再次静置至少 "
                "3 秒；若安全可行，返回起点复测漂移。"
            ),
        )
    return PublicPressureReport(
        conclusion_kind="unsupported",
        title="问题超出单一压力序列的证据边界",
        summary=(
            "Pressure Beta 只能报告稳定端点的相对压力变化和标准大气近似相对高度，不能给出该请求。"
        ),
        uncertainties=("缺少可追溯的绝对海拔、独立因果对照或设备校准证据。",),
        forbidden_claims=(
            "不得报告绝对海拔。",
            "不得仅凭压力证明移动方式、天气原因、设备校准或市场有效性。",
        ),
        next_live_measurement=(
            "若目标是相对楼层变化，可按起点—目标—返回起点的 Pressure 协议重新提问。"
        ),
    )


def _report_source(manifest: Any, device_scope: str) -> PublicPressureReportSource:
    return PublicPressureReportSource(
        dataset_id=manifest.dataset_id,
        data_class=manifest.data_class,
        device_scope=device_scope,
        source_title=manifest.source.title,
        source_url=manifest.source.record_url,
        doi=manifest.source.doi,
        license_spdx=manifest.source.license_spdx,
    )


def _evidence_snapshot(
    *,
    pack_dir: Path,
    recording_id: str,
) -> tuple[PublicPressureEvidenceSnapshot, tuple[PublicPressureToolExecution, ...]]:
    try:
        trace, ground_truth = load_verified_public_pressure_evidence(
            recording_id,
            pack_dir=pack_dir,
        )
        manifest = load_public_replay_dataset(pack_dir)
        recording = next(
            item for item in manifest.recordings if item.recording_id == recording_id
        )
        inspection = inspect_pressure_trace(trace)
        comparison = compare_pressure_height_to_ground_truth(trace, ground_truth)
        claim_audit = audit_pressure_claim_support(
            "height_change_against_ground_truth", trace, ground_truth
        )
    except (OSError, StopIteration, ValueError, PublicPressureReplayError) as exc:
        raise PublicPressureExplorationUnavailable("source-or-tool-validation-failed") from exc
    evidence_id = f"pressure-{recording_id}"
    device_scope = (
        f"{recording.device_alias}; {recording.experiment_title}; public Android capture, "
        "not phyphox"
    )
    snapshot = PublicPressureEvidenceSnapshot(
        evidence_id=evidence_id,
        dataset_id=manifest.dataset_id,
        recording_id=recording_id,
        data_class=manifest.data_class,
        device_scope=device_scope,
        source_title=manifest.source.title,
        source_url=manifest.source.record_url,
        doi=manifest.source.doi,
        license_spdx=manifest.source.license_spdx,
        inspection=inspection,
        comparison=comparison,
        claim_audit=claim_audit,
        processing_disclosures=tuple(recording.processing_disclosures),
        claim_boundary=tuple(manifest.claim_boundary),
    )
    tools = (
        PublicPressureToolExecution(
            sequence=1,
            tool_id=_INSPECT_TOOL,
            evidence_ids=(evidence_id,),
            result_codes=(
                "platforms_passed" if inspection.platforms_passed else "platforms_failed",
                f"confidence_{inspection.confidence}",
            ),
        ),
        PublicPressureToolExecution(
            sequence=2,
            tool_id=_COMPARE_TOOL,
            evidence_ids=(evidence_id,),
            result_codes=(comparison.status,),
        ),
        PublicPressureToolExecution(
            sequence=3,
            tool_id=_AUDIT_TOOL,
            evidence_ids=(evidence_id,),
            result_codes=(claim_audit.status, claim_audit.evaluation_outcome),
        ),
    )
    return snapshot, tools


def _evidence_report(
    evidence: PublicPressureEvidenceSnapshot,
    *,
    selected_report_action: str,
) -> PublicPressureReport:
    inspection = evidence.inspection
    comparison = evidence.comparison
    claim = evidence.claim_audit
    source = PublicPressureReportSource(
        dataset_id=evidence.dataset_id,
        data_class=evidence.data_class,
        device_scope=evidence.device_scope,
        source_title=evidence.source_title,
        source_url=evidence.source_url,
        doi=evidence.doi,
        license_spdx=evidence.license_spdx,
    )
    findings: list[PublicPressureFinding] = [
        PublicPressureFinding(
            finding_id="relative-pressure-height",
            text=(
                f"稳定端点压力对应的标准大气近似相对高度变化为 "
                f"{inspection.standard_atmosphere_height_change_m:.3f} m。"
            ),
            evidence_ids=(evidence.evidence_id,),
        )
    ]
    if comparison.evaluable and comparison.absolute_error_m is not None:
        findings.append(
            PublicPressureFinding(
                finding_id="source-ground-truth-check",
                text=(
                    f"与 NIST 稀疏相对高程锚点相比，绝对误差为 "
                    f"{comparison.absolute_error_m:.3f} m，判定为 {comparison.status}。"
                ),
                evidence_ids=(evidence.evidence_id,),
            )
        )
    supported = (
        selected_report_action == _FINISH_ID
        and inspection.evaluation_ready
        and claim.status == "supported_with_limitations"
    )
    if supported:
        conclusion_kind = "supported_relative_height"
        title = "公开 Pressure 证据支持有边界的相对高度结论"
        summary = (
            "Agent 选择了与问题匹配的公开手机压力记录；服务端质量门、隐藏真值比较和 claim "
            "audit 均完成。该结果可说明相对压力—高度关系，但不是你的真机结论。"
        )
        next_live = (
            "最后一步仍是你的真机 Gate C：在 phyphox Pressure 中完成起点、目标楼层和返回起点的 "
            "3 秒稳定平台，并至少重复 3 次。"
        )
    else:
        conclusion_kind = "live_measurement_required"
        title = "公开证据已分析，但需要当前手机 Pressure 复核"
        summary = (
            "公开记录提供了来源明确的物理参考，但当前问题或质量边界要求真机数据，服务端没有把 "
            "公开结果替代为用户证据。"
        )
        next_live = (
            "在 phyphox 打开 Pressure，起点和目标楼层分别静置至少 3 秒；返回起点可检查 HVAC、"
            "天气和传感器漂移。"
        )
    return PublicPressureReport(
        conclusion_kind=conclusion_kind,
        title=title,
        summary=summary,
        supported_findings=tuple(findings),
        uncertainties=(
            "标准大气换算不是绝对海拔，天气、HVAC 和设备偏置仍可能影响结果。",
            "公开数据来自 NIST Pixel XL 自定义 Android 采集，不是用户的 phyphox 真机记录。",
            "当前只有两个公开 acquisition，不能满足 Gate C 的场景、条件和重复要求。",
        ),
        forbidden_claims=tuple(claim.forbidden_phrasing),
        next_live_measurement=next_live,
        evidence_ids=(evidence.evidence_id,),
        source_ids=(evidence.dataset_id,),
        sources=(source,),
    )


def _planner_status(traces: Sequence[PublicPressurePlannerTrace]) -> str:
    sources = {item.source for item in traces}
    if sources == {"agent"}:
        return "accepted"
    if sources == {"strong_workflow_fallback"}:
        return "fallback"
    return "mixed"


async def run_public_pressure_exploration(
    request: PublicPressureExploreRequest,
    *,
    root: Path,
    planner: PlannerCallable | None = None,
) -> PublicPressureExploreResult:
    """Run a stateless, bounded public Pressure Agent loop.

    The model selects only frozen candidate IDs. The server owns source validation,
    hidden labels, deterministic physics, evidence quality, termination and reporting.
    """

    run_id = f"public-pressure-{uuid4().hex}"
    if not request.privacy_acknowledged:
        return _privacy_result(request, run_id=run_id)

    family = _question_family(request.research_question)
    initial_candidates = _initial_candidates()
    initial_request = _planner_request(
        run_id=run_id,
        step=1,
        operation="select_evidence_route",
        request=request,
        inspection=None,
        candidates=initial_candidates,
        fallback_candidate_id=_initial_fallback(family),
    )
    if family in {"live", "unsupported"}:
        selected_id = initial_request.fallback_candidate_id
        first_trace = PublicPressurePlannerTrace(
            step=1,
            operation="select_evidence_route",
            request_sha256=initial_request.request_sha256,
            candidate_ids=tuple(item.candidate_id for item in initial_candidates),
            selected_candidate_id=selected_id,
            fallback_candidate_id=selected_id,
            rationale_code=(
                "request_live_device_evidence"
                if family == "live"
                else "unsupported_claim_boundary"
            ),
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason="policy-boundary",
        )
        report = _empty_report(family)
        return PublicPressureExploreResult(
            run_id=run_id,
            research_question=request.research_question,
            execution_status="unsupported" if family == "unsupported" else "limited",
            selected_route_id=selected_id,
            planner_status="fallback",
            planner_trace=(first_trace,),
            report=report,
        )

    selected_id, first_trace = await _select_candidate(
        initial_request,
        planner=planner,
    )
    selected_candidate = next(
        item for item in initial_candidates if item.candidate_id == selected_id
    )
    if selected_candidate.terminal or selected_candidate.recording_id is None:
        terminal_family = "live" if selected_id == _LIVE_ID else "unsupported"
        report = _empty_report(terminal_family)
        return PublicPressureExploreResult(
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

    pack_dir = root.resolve() / _DATASET_ID
    evidence, tool_trace = _evidence_snapshot(
        pack_dir=pack_dir,
        recording_id=selected_candidate.recording_id,
    )
    follow_candidates = _follow_up_candidates()
    follow_fallback = (
        _FINISH_ID
        if evidence.inspection.evaluation_ready
        and evidence.inspection.confidence in {"medium", "high"}
        else _LIVE_ID
    )
    follow_request = _planner_request(
        run_id=run_id,
        step=2,
        operation="select_report_action",
        request=request,
        inspection=evidence.inspection,
        candidates=follow_candidates,
        fallback_candidate_id=follow_fallback,
    )
    follow_id, follow_trace = await _select_candidate(
        follow_request,
        planner=planner,
    )
    traces = (first_trace, follow_trace)
    report = _evidence_report(evidence, selected_report_action=follow_id)
    execution_status = (
        "completed"
        if report.conclusion_kind == "supported_relative_height"
        else "limited"
    )
    return PublicPressureExploreResult(
        run_id=run_id,
        research_question=request.research_question,
        execution_status=execution_status,
        selected_route_id=follow_id,
        planner_status=_planner_status(traces),
        planner_trace=traces,
        tool_trace=tool_trace,
        evidence=(evidence,),
        report=report,
    )


def strong_pressure_workflow_route(question: str) -> tuple[str, str]:
    """Expose the production strong baseline route for paired Harness evaluation."""

    family = _question_family(question)
    return family, _initial_fallback(family)
