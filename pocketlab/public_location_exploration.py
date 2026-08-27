from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pocketlab.public_sensor_exploration as shared
from pocketlab.public_location_tools import (
    ANALYZE_TOOL_ID,
    COMPARE_TOOL_ID,
    DATASET_ID,
    ROUTE_A_RECORDING_ID,
    ROUTE_B_RECORDING_ID,
    load_public_location_evidence,
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

PROTOCOL_ID = "location-public-exploration.v1"
PROTOCOL_VERSION = "1.0.0"

_ROUTE_A_ID = "inspect_location_route_a"
_ROUTE_B_ID = "inspect_location_route_b"
_PAIR_ID = "compare_location_repeated_routes"
_LIVE_ID = "request_live_location"
_UNSUPPORTED_ID = "stop_unsupported_location"
_FINISH_ID = "finish_location_report"
_PRIVACY_ID = "privacy_acknowledgement_required"

_SELECTION_POLICY = (
    "问题明确只看两条公开记录中时间更早、先完成、older/prior/earlier 的单条 acquisition，或 route A/first route 时选择 inspect_location_route_a。",
    "问题明确只看两条公开记录中时间更晚、后完成、newer/recent/subsequent 的单条 acquisition，或 route B/second route 时选择 inspect_location_route_b。",
    "询问两次相似路线的长度、平均路径速率、形状一致性或未限定单条时选择 compare_location_repeated_routes。",
    "询问用户当前手机/真实路线、要求设计或执行一套新的现场采集而不是分析档案，或要求比较开阔与遮挡、accuracy/status、现场误差时选择 request_live_location。",
    "要求真实地址、地图定位、住址/工作地推断、人员跟踪、交通方式/路况因果或越权请求选择 stop_unsupported_location。",
)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def location_question_family(question: str) -> str:
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
            "真实地址",
            "绝对坐标",
            "经纬度是多少",
            "地图上哪里",
            "找出住址",
            "真实住址",
            "住址推断",
            "工作地点",
            "跟踪人员",
            "监控路线",
            "交通方式",
            "路况原因",
            "天气导致",
            "real address",
            "absolute coordinate",
            "home address",
            "workplace",
            "track a person",
            "surveillance",
            "identify the route",
            "transport mode",
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
            "我的路线",
            "我走的",
            "开阔",
            "遮挡",
            "城市峡谷",
            "树荫",
            "定位精度",
            "accuracy",
            "status",
            "live measurement",
            "live data",
            "live gps",
            "live route",
            "live test",
            "real time",
            "realtime",
            "open sky",
            "urban canyon",
            "my route",
        ),
    ):
        return "live"
    if _contains_any(
        lower,
        (
            "第二次",
            "第二条",
            "route b",
            "acquisition b",
            "second route",
            "later acquisition",
        ),
    ):
        return "route_b"
    if _contains_any(
        lower,
        (
            "第一次",
            "第一条",
            "route a",
            "acquisition a",
            "first route",
            "earlier acquisition",
        ),
    ):
        return "route_a"
    return "pair"


def _initial_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id=_ROUTE_A_ID,
            title="检查公开重复路线 acquisition A",
            server_reason="只分析已注册的第一条隐私变换路线，运行来源、时间轴、长度与路径效率门。",
            rationale_code="match_location_route_a",
            recording_ids=(ROUTE_A_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="location_route_a",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_ROUTE_B_ID,
            title="检查公开重复路线 acquisition B",
            server_reason="只分析已注册的第二条隐私变换路线，运行来源、时间轴、长度与路径效率门。",
            rationale_code="match_location_route_b",
            recording_ids=(ROUTE_B_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="location_route_b",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_PAIR_ID,
            title="比较两次公开相似路线",
            server_reason="比较两次独立 acquisition 的相对长度、速率、形状最近距离和相对终点。",
            rationale_code="compare_repeated_route_geometry",
            recording_ids=(ROUTE_A_RECORDING_ID, ROUTE_B_RECORDING_ID),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="location_repeated_route_pair",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox GPS 真机实验",
            server_reason="公开轨迹没有当前设备的 accuracy/status 或开阔/遮挡标签，转入隐私受控真机路线。",
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止超出 Location 隐私与证据边界的请求",
            server_reason="地址识别、人员跟踪、住址推断、交通方式和路况因果不在本协议权限内。",
            rationale_code="unsupported_claim_boundary",
            terminal=True,
            result_code="unsupported",
        ),
    )


def _initial_fallback(family: str) -> str:
    return {
        "route_a": _ROUTE_A_ID,
        "route_b": _ROUTE_B_ID,
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
        title="需要确认本地公开 Location 路线形状边界",
        summary=(
            "确认前不会读取相对路线、调用 Planner、执行分析或写入账号；公开包不包含真实经纬度，"
            "但相对路线形状仍可能被地图匹配。"
        ),
        uncertainties=("尚未读取任何公开 Location 证据。",),
        forbidden_claims=(
            "不能把 UCI 相对路线冒充为你的 phyphox 真机路线或无再识别风险的数据。",
        ),
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
            title="这个问题需要你的 phyphox GPS 真机路线",
            summary=(
                "公开 Go!Track 数据没有当前手机的 accuracy/status 或开阔/遮挡标签，"
                "Agent 已停止证据替代并给出最小化位置数据的重复路线方案。"
            ),
            uncertainties=(
                "尚无当前手机在开阔、半遮挡和城市峡谷场景下的同路线重复、accuracy、status 与漂移记录。",
            ),
            forbidden_claims=(
                "不得把相对路线图恢复或发布为真实住址与绝对坐标。",
                "不得在没有 accuracy/status 和参考轨迹时声称测绘级误差。",
            ),
            next_live_measurement=(
                "取得位置采集同意，在 phyphox 打开 GPS/Location；选择一条安全、不经过住宅入口的短路线，"
                "只保存相对起点后的局部坐标、相对时间、accuracy、speed 与 status。固定手机位置和步行节奏，"
                "在开阔、树木/檐下半遮挡、建筑夹道 3 个场景各做同方向与反方向 2 个条件，每条件重复 3 次。"
                "每次结束回到起点检查漂移；若 status 非活跃、accuracy 恶化或跳点超门，原条件纠偏重测。"
            ),
        )
    return PublicSensorReport(
        conclusion_kind="unsupported",
        title="问题超出当前 Location 隐私与证据边界",
        summary=(
            "当前 Beta 只比较隐私变换后的相对路线指标，不执行地址恢复、人员跟踪、居住地推断、"
            "交通方式识别或路况/天气因果判断。"
        ),
        uncertainties=("缺少合法目的、位置授权、绝对位置必要性证明与相应安全控制。",),
        forbidden_claims=(
            "不得恢复或发布真实经纬度、住址、工作地或人员轨迹。",
            "不得从该对照声称交通方式、路况或天气原因。",
        ),
        next_live_measurement=(
            "若目标是非敏感的 GPS 稳定性实验，只采集相对局部坐标和质量字段，避开住宅入口并按"
            "固定路线、单变量和重复测量方案执行。"
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
        prefix = "route_a" if item.recording_id == ROUTE_A_RECORDING_ID else "route_b"
        route_label = "A" if prefix == "route_a" else "B"
        findings.append(
            PublicSensorFinding(
                finding_id=f"relative-route-{route_label.casefold()}",
                text=(
                    f"公开相对路线 {route_label} 长 {metrics[f'{prefix}_distance_m']:.1f} m，"
                    f"平均路径速率 {metrics[f'{prefix}_average_speed_m_s']:.3f} m/s，"
                    f"路径效率 {metrics[f'{prefix}_efficiency_ratio']:.3f}。"
                ),
                evidence_ids=(item.evidence_id,),
            )
        )
    if "route_length_difference_percent" in metrics:
        findings.append(
            PublicSensorFinding(
                finding_id="relative-repeated-route-geometry",
                text=(
                    f"两次路线长度差 {metrics['route_length_difference_percent']:.3f}%，"
                    f"B/A 平均路径速率比 {metrics['route_b_over_a_speed_ratio']:.3f}；"
                    f"相对轨迹最近点中位距离 {metrics['symmetric_median_nearest_distance_m']:.1f} m，"
                    f"P95 {metrics['symmetric_p95_nearest_distance_m']:.1f} m，相对终点分离 "
                    f"{metrics['relative_endpoint_separation_m']:.1f} m。"
                ),
                evidence_ids=tuple(item.evidence_id for item in evidence),
            )
        )

    passed = comparison.quality_passed
    return PublicSensorReport(
        conclusion_kind="supported_with_limits" if passed else "live_measurement_required",
        title=(
            "公开 Location 证据支持有边界的重复路线结论"
            if passed
            else "公开相对路线质量门未通过，需要真机复核"
        ),
        summary=(
            "Agent 在冻结候选中选择了真实 Android Go!Track 公开记录；服务端完成来源、"
            "隐私投影、轨迹分析和相对几何门。结果只描述两次相似路线，不能解释为绝对 GPS 误差"
            "或开阔/遮挡因果。"
            if passed
            else "服务端来源、时间轴或相对几何门未通过，没有扩大公开位置数据结论。"
        ),
        supported_findings=tuple(findings),
        uncertainties=(
            "来源没有 per-point accuracy、status、altitude 或设备 speed 通道。",
            "路线已平移、反射、旋转并量化；只能比较相对几何，不能映射到真实地点。",
            "两次 acquisition 来自一个 Android 设备和同一天，不覆盖设备间差异。",
            "相对路线形状仍可能被地图匹配，因此只允许确认后的本地回放。",
        ),
        forbidden_claims=(
            "不得恢复、推断或发布绝对坐标、住址、工作地或人员身份。",
            "不得声称 accuracy、测绘级绝对误差或开阔/遮挡环境因果。",
            "不得声称交通方式、路况、天气或路线名称。",
            "不得把公开回放计入用户真机 Gate C、市场验证或 production agent_ready。",
        ),
        next_live_measurement=(
            "最后一步是真机 Gate C：在取得位置采集同意后，用 phyphox GPS 只保留相对局部坐标、"
            "相对时间、accuracy、speed 与 status；开阔、半遮挡、建筑夹道 3 场景 × 正反方向 2 条件"
            "× 3 重复，并回到起点检查漂移。"
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        source_ids=(evidence[0].dataset_id,),
        sources=(_report_source(evidence[0]),),
    )


async def run_public_location_exploration(
    request: PublicSensorExploreRequest,
    *,
    root: Path,
    planner: shared.PlannerCallable | None = None,
) -> PublicSensorExploreResult:
    if request.sensor != "location":
        raise shared.PublicSensorExplorationUnavailable("sensor-protocol-mismatch")
    run_id = f"public-{request.sensor}-{uuid4().hex}"
    if not request.privacy_acknowledged:
        return _privacy_result(request, run_id=run_id)

    family = location_question_family(request.research_question)
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

    selected_id, first_trace = await shared._select_candidate(first_request, planner=planner)
    selected = next(item for item in candidates if item.candidate_id == selected_id)
    if selected.terminal:
        terminal_family = "live" if selected_id == _LIVE_ID else "unsupported"
        report = _terminal_report(terminal_family)
        return PublicSensorExploreResult(
            sensor=request.sensor,
            protocol_id=PROTOCOL_ID,
            run_id=run_id,
            research_question=request.research_question,
            execution_status=("unsupported" if terminal_family == "unsupported" else "limited"),
            selected_route_id=selected_id,
            planner_status=shared._planner_status((first_trace,)),
            planner_trace=(first_trace,),
            report=report,
        )

    try:
        evidence, comparison, tool_trace = load_public_location_evidence(
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


def strong_location_workflow_route(question: str) -> tuple[str, str]:
    family = location_question_family(question)
    return family, _initial_fallback(family)
