from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pocketlab.public_sensor_exploration as shared
from pocketlab.public_magnetometer_tools import (
    ANALYZE_TOOL_ID,
    ANCHOR_RECORDING_ID,
    CHANGE_RECORDING_ID,
    COMPARE_TOOL_ID,
    DATASET_ID,
    load_public_magnetometer_evidence,
)
from pocketlab.public_sensor_agent_models import (
    PublicSensorComparison,
    PublicSensorEvidenceSnapshot,
    PublicSensorEvidenceView,
    PublicSensorExploreRequest,
    PublicSensorExploreResult,
    PublicSensorFinding,
    PublicSensorPlanCandidate,
    PublicSensorPlannerRequest,
    PublicSensorPlannerTrace,
    PublicSensorReport,
    PublicSensorReportSource,
)

PROTOCOL_ID = "magnetometer-public-exploration.v1"
PROTOCOL_VERSION = "1.0.0"

_STABLE_ID = "inspect_stable_magnetic_field"
_CHANGE_ID = "inspect_changing_magnetic_field"
_PAIR_ID = "compare_stable_changing_magnetic_field"
_LIVE_ID = "request_live_magnetometer"
_UNSUPPORTED_ID = "stop_unsupported_magnetometer"
_FINISH_ID = "finish_magnetometer_report"
_PRIVACY_ID = "privacy_acknowledgement_required"

_SELECTION_POLICY = (
    "背景、稳定性、静置噪声或单一参考问题选择 inspect_stable_magnetic_field。",
    "问题明确要求只看、单独检查、on its own、alone 或 non-reference 的变化片段时，必须选择 inspect_changing_magnetic_field，不得擅自加入背景对照。",
    "只有问题要求比较、区分、分离两个状态，或泛问传感器能否检测局部变化而未限定单片段时，才选择 compare_stable_changing_magnetic_field。",
    "询问用户当前手机、此刻现场、具体房间或新采集读数时选择 request_live_magnetometer。",
    "要求识别具体物体、证明因果、绝对航向、精确空间定位、设备校准或候选外请求选择 stop_unsupported_magnetometer。",
)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def magnetometer_question_family(question: str) -> str:
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
            "证明是",
            "确定是哪个物体",
            "识别具体物体",
            "identify the object",
            "prove causation",
            "绝对航向",
            "正北",
            "absolute heading",
            "精确定位",
            "校准系数",
            "设备校准",
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
            "这个房间",
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
            "局部",
            "空间",
            "扫描",
            "附近",
            "异常",
            "变化",
            "compare",
            "contrast",
            "scan",
            "anomaly",
            "change",
        ),
    ):
        return "pair"
    if _contains_any(
        lower,
        (
            "背景",
            "静止",
            "静置",
            "稳定",
            "噪声",
            "baseline",
            "stationary",
            "stable",
            "background",
        ),
    ):
        return "stable"
    return "pair"


def _initial_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id=_STABLE_ID,
            title="检查公开稳定磁场锚点",
            server_reason="只分析已注册的 NIST AS7 稳定窗口，判断场强波动是否处在预注册低值门内。",
            rationale_code="match_stable_field_goal",
            recording_ids=(ANCHOR_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="stable_field_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_CHANGE_ID,
            title="检查公开磁场变化窗口",
            server_reason="只分析已注册的 NIST AS7 变化窗口，判断是否存在明显场强变化候选。",
            rationale_code="match_field_change_goal",
            recording_ids=(CHANGE_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="field_change_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_PAIR_ID,
            title="比较稳定与变化磁场窗口",
            server_reason="比较同一公开 acquisition 的稳定和变化窗口，用预注册门检查磁力计响应分离。",
            rationale_code="compare_field_states",
            recording_ids=(ANCHOR_RECORDING_ID, CHANGE_RECORDING_ID),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="paired_field_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox Magnetometer 真机扫描",
            server_reason="公开数据不能回答当前用户空间或具体物体，转入受控真机测量建议。",
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止超出证据边界的请求",
            server_reason="物体身份、因果、绝对航向、精确定位或校准请求不在当前证据范围内。",
            rationale_code="unsupported_claim_boundary",
            terminal=True,
            result_code="unsupported",
        ),
    )


def _initial_fallback(family: str) -> str:
    return {
        "stable": _STABLE_ID,
        "pair": _PAIR_ID,
        "live": _LIVE_ID,
        "unsupported": _UNSUPPORTED_ID,
    }[family]


def _planner_request(
    *,
    run_id: str,
    request: PublicSensorExploreRequest,
    candidates: tuple[PublicSensorPlanCandidate, ...],
    fallback_candidate_id: str,
) -> PublicSensorPlannerRequest:
    return shared._planner_request(
        run_id=run_id,
        step=1,
        operation="select_evidence_route",
        request=request,
        evidence_view=PublicSensorEvidenceView(),
        candidates=candidates,
        fallback_candidate_id=fallback_candidate_id,
        protocol_id=PROTOCOL_ID,
        protocol_version=PROTOCOL_VERSION,
        selection_policy=_SELECTION_POLICY,
    )


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
        title="需要确认本地公开 Magnetometer 回放边界",
        summary="确认前不会读取公开序列、调用 Planner、执行分析或写入账号。",
        uncertainties=("尚未读取任何公开 Magnetometer 证据。",),
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
            title="这个问题需要你的手机 Magnetometer 数据",
            summary="公开 Pixel XL 回放不能回答当前空间或具体物体附近的磁场状态，Agent 已停止证据替代。",
            uncertainties=("尚无当前手机的校准状态、固定朝向背景和重复空间扫描。",),
            forbidden_claims=("不得用公开记录定位你现场的异常或识别具体物体。",),
            next_live_measurement=(
                "先按手机指引完成磁力计校准；在 phyphox 打开 Magnetometer，保持手机方向不变，"
                "先远离金属记录背景 5 秒，再按固定间距扫描目标区域，每个位置记录 5 秒并至少重复 3 次。"
            ),
        )
    return PublicSensorReport(
        conclusion_kind="unsupported",
        title="问题超出当前 Magnetometer 证据边界",
        summary="当前 Beta 只能比较磁场模长的稳定与变化状态，不能识别物体、证明因果或给出绝对航向。",
        uncertainties=("缺少受控物体标签、空间真值、当前设备校准与重复真机证据。",),
        forbidden_claims=(
            "不得从公开切片声称某个具体物体导致了变化。",
            "不得声称绝对航向、精确空间位置、设备校准、Gate C 或市场有效性。",
        ),
        next_live_measurement="若目标是局部场扫描，可按固定朝向、固定间距和单变量原则采集背景—目标对照。",
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
    quality_passed: bool,
) -> PublicSensorReport:
    metric_by_key = {item.key: item.value for item in comparison.metrics}
    findings: list[PublicSensorFinding] = []
    anchor = next(
        (item for item in evidence if item.recording_id == ANCHOR_RECORDING_ID), None
    )
    change = next(
        (item for item in evidence if item.recording_id == CHANGE_RECORDING_ID), None
    )
    if anchor is not None and "stable_field_std_ut" in metric_by_key:
        findings.append(
            PublicSensorFinding(
                finding_id="stable-field-variability",
                text=f"公开稳定窗口的场强模长标准差为 {metric_by_key['stable_field_std_ut']:.3f} uT。",
                evidence_ids=(anchor.evidence_id,),
            )
        )
    if change is not None and "changing_max_deviation_ut" in metric_by_key:
        findings.append(
            PublicSensorFinding(
                finding_id="field-change-candidate",
                text=f"公开变化窗口相对中位数的最大偏差为 {metric_by_key['changing_max_deviation_ut']:.3f} uT。",
                evidence_ids=(change.evidence_id,),
            )
        )
    if "field_variability_ratio" in metric_by_key:
        findings.append(
            PublicSensorFinding(
                finding_id="field-state-separation",
                text=f"变化与稳定窗口的峰峰值比为 {metric_by_key['field_variability_ratio']:.1f}。",
                evidence_ids=tuple(item.evidence_id for item in evidence),
            )
        )

    return PublicSensorReport(
        conclusion_kind=("supported_with_limits" if quality_passed else "live_measurement_required"),
        title=(
            "公开 Magnetometer 证据支持有边界的场变化响应结论"
            if quality_passed
            else "公开 Magnetometer 质量门不足，需要真机复核"
        ),
        summary=(
            "Agent 在冻结候选中选择了与问题匹配的 NIST 公开手机记录；服务端分析器与预注册质量门完成。"
            "结果只支持该记录中磁场稳定与变化状态可分，不识别物体、不证明因果。"
            if quality_passed
            else "服务端质量门未通过，没有扩大公开数据结论。"
        ),
        supported_findings=tuple(findings),
        uncertainties=(
            "两段切片来自同一台 Pixel XL、同一次 AS7 acquisition，不能视为独立手机重复。",
            "公开数据由 NIST 自定义 Android 应用采集，不是 phyphox 导出。",
            "没有受控磁性物体、手机朝向或空间位置真值，不能进行因果归因或定位。",
        ),
        forbidden_claims=(
            "不得声称具体物体导致变化、给出绝对航向或精确异常位置。",
            "不得把公开回放计入用户真机 Gate C、市场验证或 production agent_ready。",
        ),
        next_live_measurement=(
            "最后一步是真机 Gate C：校准磁力计后固定手机方向，采集远离金属的背景和固定间距扫描；"
            "每个位置记录 5 秒，并在 3 个场景中完成 2 条件 × 3 重复。"
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        source_ids=(evidence[0].dataset_id,),
        sources=(_report_source(evidence[0]),),
    )


async def run_public_magnetometer_exploration(
    request: PublicSensorExploreRequest,
    *,
    root: Path,
    planner: shared.PlannerCallable | None = None,
) -> PublicSensorExploreResult:
    if request.sensor != "magnetometer":
        raise shared.PublicSensorExplorationUnavailable("sensor-protocol-mismatch")
    run_id = f"public-{request.sensor}-{uuid4().hex}"
    if not request.privacy_acknowledged:
        return _privacy_result(request, run_id=run_id)

    family = magnetometer_question_family(request.research_question)
    candidates = _initial_candidates()
    planner_request = _planner_request(
        run_id=run_id,
        request=request,
        candidates=candidates,
        fallback_candidate_id=_initial_fallback(family),
    )
    if family in {"live", "unsupported"}:
        selected_id = planner_request.fallback_candidate_id
        candidate = next(item for item in candidates if item.candidate_id == selected_id)
        trace = PublicSensorPlannerTrace(
            step=1,
            operation="select_evidence_route",
            request_sha256=planner_request.request_sha256,
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
        planner_request, planner=planner
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
            execution_status="unsupported" if terminal_family == "unsupported" else "limited",
            selected_route_id=selected_id,
            planner_status=shared._planner_status((first_trace,)),
            planner_trace=(first_trace,),
            report=report,
        )

    try:
        evidence, comparison, tool_trace = load_public_magnetometer_evidence(
            root.resolve() / DATASET_ID, selected.recording_ids
        )
    except (OSError, StopIteration, ValueError) as exc:
        raise shared.PublicSensorExplorationUnavailable(
            "source-or-tool-validation-failed"
        ) from exc

    finish_id = _FINISH_ID if comparison.quality_passed else _LIVE_ID
    finish_candidate = PublicSensorPlanCandidate(
        candidate_id=finish_id,
        title=("形成有边界的 Magnetometer 报告" if comparison.quality_passed else "请求真机复核"),
        server_reason=(
            "服务端确定性质量门已通过，形成有边界的描述性报告。"
            if comparison.quality_passed
            else "服务端质量门未通过，停止并请求用户真机复核。"
        ),
        rationale_code=(
            "evidence_quality_sufficient"
            if comparison.quality_passed
            else "evidence_quality_insufficient"
        ),
        terminal=True,
        result_code=("finish_report" if comparison.quality_passed else "live_measurement_required"),
    )
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
        rationale_code=finish_candidate.rationale_code,
        source="strong_workflow_fallback",
        outcome="fallback",
        fallback_reason="server-owned-termination",
    )
    traces = (first_trace, finish_trace)
    report = _evidence_report(
        evidence, comparison, quality_passed=comparison.quality_passed
    )
    return PublicSensorExploreResult(
        sensor=request.sensor,
        protocol_id=PROTOCOL_ID,
        run_id=run_id,
        research_question=request.research_question,
        execution_status=("completed" if comparison.quality_passed else "limited"),
        selected_route_id=finish_id,
        planner_status=shared._planner_status(traces),
        planner_trace=traces,
        tool_trace=tool_trace,
        evidence=evidence,
        comparison=comparison,
        report=report,
    )


def strong_magnetometer_workflow_route(question: str) -> tuple[str, str]:
    family = magnetometer_question_family(question)
    return family, _initial_fallback(family)
