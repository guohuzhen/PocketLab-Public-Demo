from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pocketlab.public_sensor_exploration as shared
from pocketlab.public_proximity_tools import (
    ANALYZE_TOOL_ID,
    COMPARE_TOOL_ID,
    DATASET_ID,
    EARLY_RECORDING_ID,
    LATE_RECORDING_ID,
    load_public_proximity_evidence,
)
from pocketlab.public_sensor_agent_models import (
    PublicSensorComparison,
    PublicSensorEvidenceSnapshot,
    PublicSensorEvidenceView,
    PublicSensorExploreRequest,
    PublicSensorExploreResult,
    PublicSensorFinding,
    PublicSensorPlanCandidate,
    PublicSensorPlannerTrace,
    PublicSensorReport,
    PublicSensorReportSource,
)

PROTOCOL_ID = "proximity-public-exploration.v1"
PROTOCOL_VERSION = "1.0.0"

_EARLY_ID = "inspect_early_proximity_events"
_LATE_ID = "inspect_late_proximity_events"
_PAIR_ID = "compare_proximity_event_slices"
_LIVE_ID = "request_live_proximity"
_UNSUPPORTED_ID = "stop_unsupported_proximity"
_FINISH_ID = "finish_proximity_report"
_PRIVACY_ID = "privacy_acknowledgement_required"

_SELECTION_POLICY = (
    "问题明确只看第一段、较早切片或 early slice 时选择 inspect_early_proximity_events。",
    "问题明确只看第二段、较晚切片、later slice 或重复片段时选择 inspect_late_proximity_events。",
    "询问二态还是连续、状态编码是否重复、比较前后切片，或未限定单片段时选择 compare_proximity_event_slices。",
    "询问用户当前手机、现场材质/角度、实际触发距离或新采集读数时选择 request_live_proximity。",
    "要求人物识别、存在检测、医疗判断、远程监控、公开数据证明材质因果或候选外请求选择 stop_unsupported_proximity。",
)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def proximity_question_family(question: str) -> str:
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
            "识别人",
            "判断是谁",
            "存在检测",
            "远程监控",
            "医疗诊断",
            "identify a person",
            "presence detection",
            "medical",
            "证明某种材质",
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
            "真机数据",
            "实际触发距离",
            "触发阈值",
            "不同材质",
            "不同角度",
            "live",
            "real time",
            "realtime",
        ),
    ):
        return "live"
    if _contains_any(
        lower,
        (
            "第二段",
            "后半",
            "较晚",
            "后续",
            "重复片段",
            "late slice",
            "later slice",
            "second slice",
        ),
    ):
        return "late"
    if _contains_any(
        lower,
        (
            "第一段",
            "前半",
            "较早",
            "最初片段",
            "early slice",
            "first slice",
        ),
    ):
        return "early"
    return "pair"


def _initial_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id=_EARLY_ID,
            title="检查公开早期 Proximity 事件切片",
            server_reason="只分析已注册的前四条稀疏事件，检查 0/5 cm 二态编码和切换合同。",
            rationale_code="match_early_event_slice",
            recording_ids=(EARLY_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="early_binary_event_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LATE_ID,
            title="检查公开后期 Proximity 事件切片",
            server_reason="只分析已注册的后四条稀疏事件，检查重复的 0/5 cm 二态编码。",
            rationale_code="match_late_event_slice",
            recording_ids=(LATE_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="late_binary_event_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_PAIR_ID,
            title="比较公开前后 Proximity 事件切片",
            server_reason="比较同一 acquisition 的两个切片，检查二态编码与切换是否前后一致。",
            rationale_code="compare_binary_event_slices",
            recording_ids=(EARLY_RECORDING_ID, LATE_RECORDING_ID),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="paired_binary_event_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox Proximity 真机实验",
            server_reason="公开稀疏事件不能回答当前手机、材质、角度或真实触发距离，转入受控真机方案。",
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止超出 Proximity 证据边界的请求",
            server_reason="人物识别、存在监控、医疗判断或公开数据因果归因不在当前协议权限内。",
            rationale_code="unsupported_claim_boundary",
            terminal=True,
            result_code="unsupported",
        ),
    )


def _initial_fallback(family: str) -> str:
    return {
        "early": _EARLY_ID,
        "late": _LATE_ID,
        "pair": _PAIR_ID,
        "live": _LIVE_ID,
        "unsupported": _UNSUPPORTED_ID,
    }[family]


def _privacy_result(
    request: PublicSensorExploreRequest, *, run_id: str
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
        title="需要确认本地公开 Proximity 回放边界",
        summary="确认前不会读取公开序列、调用 Planner、执行分析或写入账号。",
        uncertainties=("尚未读取任何公开 Proximity 证据。",),
        forbidden_claims=("不能把 NIST Pixel XL 数据冒充为你的 phyphox 真机数据。",),
        next_live_measurement="确认后可运行本地公开回放；用户真机 Gate C 仍需单独采集。",
    )
    return PublicSensorExploreResult(
        sensor=request.sensor,
        protocol_id=PROTOCOL_ID,
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
            title="这个问题需要你的手机 Proximity 数据",
            summary=(
                "公开 Pixel XL 事件不能回答当前手机的材质、角度或实际触发距离，"
                "Agent 已停止证据替代并给出受控真机方案。"
            ),
            uncertainties=(
                "尚无当前手机的 near/far 编码、外部尺标、固定角度与重复触发记录。",
            ),
            forbidden_claims=(
                "不得把手机回报的 0/5 cm 状态码直接当作连续真实距离。",
            ),
            next_live_measurement=(
                "在 phyphox 打开 Proximity，先保持无遮挡记录 far 基线；用软质哑光卡片沿外部尺标缓慢接近，"
                "另用尺读取状态翻转时的位置。每种材质或角度一次只改一个变量，做接近—离开各 3 次；"
                "最终在 3 个场景完成 2 条件 × 3 重复。"
            ),
        )
    return PublicSensorReport(
        conclusion_kind="unsupported",
        title="问题超出当前 Proximity 证据边界",
        summary=(
            "当前 Beta 只分析接近传感器的状态编码和重复切换，不能识别人、进行存在监控或给出医疗判断。"
        ),
        uncertainties=("缺少受控标签、伦理授权、当前设备真机证据和独立距离真值。",),
        forbidden_claims=(
            "不得从 Proximity 状态推断人物身份、健康状态或持续存在。",
            "不得用公开事件证明某种材质或角度造成了触发变化。",
        ),
        next_live_measurement=(
            "若目标是普通传感器行为，可按单变量、外部尺标和重复触发方案采集非敏感真机数据。"
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
) -> PublicSensorReport:
    metrics = {item.key: item.value for item in comparison.metrics}
    findings: list[PublicSensorFinding] = []
    for item in evidence:
        prefix = "early" if item.recording_id == EARLY_RECORDING_ID else "late"
        findings.append(
            PublicSensorFinding(
                finding_id=f"{prefix}-binary-events",
                text=(
                    f"公开{('前' if prefix == 'early' else '后')}切片包含 "
                    f"{metrics[f'{prefix}_event_count']:.0f} 条事件、"
                    f"{metrics[f'{prefix}_transition_count']:.0f} 次状态切换，"
                    f"near/far 编码为 {metrics[f'{prefix}_near_state_cm']:.0f}/"
                    f"{metrics[f'{prefix}_far_state_cm']:.0f} cm。"
                ),
                evidence_ids=(item.evidence_id,),
            )
        )
    if "total_transition_count" in metrics:
        findings.append(
            PublicSensorFinding(
                finding_id="binary-code-consistency",
                text=(
                    f"两个切片合计 {metrics['total_transition_count']:.0f} 次切换，"
                    "且 near/far 状态编码前后一致。"
                ),
                evidence_ids=tuple(item.evidence_id for item in evidence),
            )
        )

    passed = comparison.quality_passed
    return PublicSensorReport(
        conclusion_kind="supported_with_limits" if passed else "live_measurement_required",
        title=(
            "公开 Proximity 证据支持有边界的二态响应结论"
            if passed
            else "公开 Proximity 事件门未通过，需要真机复核"
        ),
        summary=(
            "Agent 在冻结候选中选择了 NIST 公开稀疏事件；服务端专用分析器和预注册状态门已完成。"
            "结果只支持该来源呈 0/5 cm 二态状态事件，不支持连续距离、响应时间或材质因果。"
            if passed
            else "服务端状态质量门未通过，没有扩大公开数据结论。"
        ),
        supported_findings=tuple(findings),
        uncertainties=(
            "全部 8 条事件来自同一台 Pixel XL、同一次 AS7 acquisition，不是独立手机重复。",
            "公开数据由 NIST 自定义 Android 应用采集，不是 phyphox 导出。",
            "事件流稀疏且只在状态变化时出现，不能从间隔推断采样率、占空比或响应时间。",
            "来源没有物体、材质、角度或触发距离真值。",
        ),
        forbidden_claims=(
            "不得把 0/5 cm 状态编码解释成连续真实距离。",
            "不得声称某种材质、角度或物体导致了这些公开事件。",
            "不得把公开回放计入用户真机 Gate C、市场验证或 production agent_ready。",
        ),
        next_live_measurement=(
            "最后一步是真机 Gate C：phyphox Proximity 中记录 far 基线；用外部尺标和软质哑光卡片"
            "缓慢接近/离开，每次只改变材质或角度一个变量，每条件 3 次；在 3 个场景完成 2 条件 × 3 重复。"
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        source_ids=(evidence[0].dataset_id,),
        sources=(_report_source(evidence[0]),),
    )


async def run_public_proximity_exploration(
    request: PublicSensorExploreRequest,
    *,
    root: Path,
    planner: shared.PlannerCallable | None = None,
) -> PublicSensorExploreResult:
    if request.sensor != "proximity":
        raise shared.PublicSensorExplorationUnavailable("sensor-protocol-mismatch")
    run_id = f"public-{request.sensor}-{uuid4().hex}"
    if not request.privacy_acknowledged:
        return _privacy_result(request, run_id=run_id)

    family = proximity_question_family(request.research_question)
    candidates = _initial_candidates()
    first_request = shared._planner_request(
        run_id=run_id,
        step=1,
        operation="select_evidence_route",
        request=request,
        evidence_view=PublicSensorEvidenceView(),
        candidates=candidates,
        fallback_candidate_id=_initial_fallback(family),
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        selection_policy=_SELECTION_POLICY,
    )
    if family in {"live", "unsupported"}:
        selected_id = first_request.fallback_candidate_id
        candidate = next(item for item in candidates if item.candidate_id == selected_id)
        trace = PublicSensorPlannerTrace(
            step=1,
            operation="select_evidence_route",
            request_sha256=first_request.request_sha256,
            candidate_ids=tuple(item.candidate_id for item in candidates),
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
            protocol_id=PROTOCOL_ID,
            run_id=run_id,
            research_question=request.research_question,
            execution_status="unsupported" if family == "unsupported" else "limited",
            selected_route_id=selected_id,
            planner_status="fallback",
            planner_trace=(trace,),
            report=report,
        )

    selected_id, first_trace = await shared._select_candidate(
        first_request, planner=planner
    )
    selected = next(item for item in candidates if item.candidate_id == selected_id)
    if selected.terminal:
        terminal_family = "live" if selected_id == _LIVE_ID else "unsupported"
        report = _terminal_report(terminal_family)
        return PublicSensorExploreResult(
            sensor=request.sensor,
            protocol_id=PROTOCOL_ID,
            run_id=run_id,
            research_question=request.research_question,
            execution_status=(
                "unsupported" if terminal_family == "unsupported" else "limited"
            ),
            selected_route_id=selected_id,
            planner_status=shared._planner_status((first_trace,)),
            planner_trace=(first_trace,),
            report=report,
        )

    try:
        evidence, comparison, tool_trace = load_public_proximity_evidence(
            root.resolve() / DATASET_ID, selected.recording_ids
        )
    except (OSError, StopIteration, ValueError) as exc:
        raise shared.PublicSensorExplorationUnavailable(
            "source-or-tool-validation-failed"
        ) from exc

    finish_id = _FINISH_ID if comparison.quality_passed else _LIVE_ID
    digest = hashlib.sha256(
        json.dumps(
            {
                "run_id": run_id,
                "sensor": request.sensor,
                "quality_passed": comparison.quality_passed,
                "result_codes": comparison.result_codes,
                "candidate_id": finish_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    finish_trace = PublicSensorPlannerTrace(
        step=2,
        operation="select_report_action",
        request_sha256=digest,
        candidate_ids=(_FINISH_ID, _LIVE_ID),
        selected_candidate_id=finish_id,
        fallback_candidate_id=finish_id,
        rationale_code=(
            "evidence_quality_sufficient"
            if comparison.quality_passed
            else "evidence_quality_insufficient"
        ),
        source="strong_workflow_fallback",
        outcome="fallback",
        fallback_reason="server-owned-termination",
    )
    traces = (first_trace, finish_trace)
    report = _evidence_report(evidence, comparison)
    return PublicSensorExploreResult(
        sensor=request.sensor,
        protocol_id=PROTOCOL_ID,
        run_id=run_id,
        research_question=request.research_question,
        execution_status="completed" if comparison.quality_passed else "limited",
        selected_route_id=finish_id,
        planner_status=shared._planner_status(traces),
        planner_trace=traces,
        tool_trace=tool_trace,
        evidence=evidence,
        comparison=comparison,
        report=report,
    )


def strong_proximity_workflow_route(question: str) -> tuple[str, str]:
    family = proximity_question_family(question)
    return family, _initial_fallback(family)
