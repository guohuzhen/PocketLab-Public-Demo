from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import pocketlab.public_sensor_exploration as shared
from pocketlab.public_microphone_tools import (
    ANALYZE_TOOL_ID,
    COMPARE_TOOL_ID,
    DATASET_ID,
    EARLY_RECORDING_ID,
    LATE_RECORDING_ID,
    load_public_microphone_evidence,
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

PROTOCOL_ID = "microphone-public-exploration.v1"
PROTOCOL_VERSION = "1.0.0"

_EARLY_ID = "inspect_early_microphone_window"
_LATE_ID = "inspect_late_microphone_window"
_PAIR_ID = "compare_microphone_time_windows"
_LIVE_ID = "request_live_microphone"
_UNSUPPORTED_ID = "stop_unsupported_microphone"
_FINISH_ID = "finish_microphone_report"
_PRIVACY_ID = "privacy_acknowledgement_required"

_SELECTION_POLICY = (
    "问题明确只看公开轨迹的开头、前 20 秒或 first window 时选择 inspect_early_microphone_window。",
    "问题明确只看公开轨迹的结尾、后 20 秒或 last window 时选择 inspect_late_microphone_window。",
    "询问派生相对级别前后是否变化、比较两个窗口或未限定单窗口时选择 compare_microphone_time_windows。",
    "询问用户当前手机、房间位置、声源布置、现场数据或新实验读数时选择 request_live_microphone。",
    "要求原始音频、转写/识别说话内容、监听人员、校准 SPL、合规/健康判断或越权请求选择 stop_unsupported_microphone。",
)


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def microphone_question_family(question: str) -> str:
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
            "原始音频",
            "保存录音",
            "恢复录音",
            "转写",
            "识别说话",
            "谁在说话",
            "监听",
            "窃听",
            "raw audio",
            "transcribe",
            "speech recognition",
            "identify the speaker",
            "surveillance",
            "校准声压",
            "绝对分贝",
            "合规分贝",
            "听力损伤",
            "医疗判断",
            "calibrated spl",
            "compliance",
            "hearing damage",
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
            "房间",
            "位置",
            "驻波",
            "固定声源",
            "声源布置",
            "live",
            "real time",
            "realtime",
            "this room",
            "room position",
        ),
    ):
        return "live"
    if _contains_any(
        lower,
        (
            "结尾",
            "后 20",
            "后20",
            "最后一段",
            "后段",
            "late window",
            "last window",
            "closing window",
        ),
    ):
        return "late"
    if _contains_any(
        lower,
        (
            "开头",
            "前 20",
            "前20",
            "最初一段",
            "前段",
            "early window",
            "first window",
            "opening window",
        ),
    ):
        return "early"
    return "pair"


def _initial_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id=_EARLY_ID,
            title="检查公开轨迹前 20 秒派生相对级别",
            server_reason="只分析已注册的前 20 个一秒 LAeq 派生点，运行来源、节奏和相对级别质量门。",
            rationale_code="match_early_relative_window",
            recording_ids=(EARLY_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="early_relative_level_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LATE_ID,
            title="检查公开轨迹后 20 秒派生相对级别",
            server_reason="只分析已注册的后 20 个一秒 LAeq 派生点，运行来源、节奏和相对级别质量门。",
            rationale_code="match_late_relative_window",
            recording_ids=(LATE_RECORDING_ID,),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="late_relative_level_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_PAIR_ID,
            title="比较公开轨迹前后相对级别窗口",
            server_reason="比较同一公开轨迹的前后窗口，检查平均、峰值和变化范围的时间对比。",
            rationale_code="compare_chronological_relative_levels",
            recording_ids=(EARLY_RECORDING_ID, LATE_RECORDING_ID),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_TOOL_ID),
            result_code="paired_relative_level_route",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox Audio amplitude 真机实验",
            server_reason="公开时间窗口没有房间位置或当前设备标签，转入隐私受控、单变量真机方案。",
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止超出 Microphone 证据边界的请求",
            server_reason="原始音频、语音内容、人员监听、绝对声压和健康/合规判断不在本协议权限内。",
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
        title="需要确认本地公开 Microphone 派生回放边界",
        summary=(
            "确认前不会读取公开相对级别序列、调用 Planner、执行分析或写入账号；系统始终不处理原始音频。"
        ),
        uncertainties=("尚未读取任何公开 Microphone 派生证据。",),
        forbidden_claims=(
            "不能把 NoiseCapture 公开数据冒充为你的 phyphox 真机记录或录音内容。",
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
            title="这个问题需要你的 phyphox Microphone 派生数据",
            summary=(
                "公开 NoiseCapture 时间窗口没有房间位置、固定声源或当前手机标签，"
                "Agent 已停止证据替代并给出不保存原始音频的受控实验方案。"
            ),
            uncertainties=(
                "尚无当前手机的参考位置、比较位置、固定音量/片段、方向控制、漂移复测和重复记录。",
            ),
            forbidden_claims=(
                "不得把 Audio amplitude 或 dB_relative 当作校准 SPL。",
                "不得记录、恢复、转写或识别语音内容。",
            ),
            next_live_measurement=(
                "先确认环境中没有对话并取得在场人员同意；在 phyphox 打开 Audio amplitude，"
                "固定舒适音量的测试声源、播放片段、手机方向与测量时长。标记参考位置和一个比较位置，"
                "每处采 5 秒派生幅值并各重复 3 次，再回到参考位置复测漂移。一次只改变位置；"
                "若出现上限平台或前后漂移，降低音量或缩短距离后重测。最终在 3 个场景完成"
                "2 个位置 × 3 重复，且只保存派生数值。"
            ),
        )
    return PublicSensorReport(
        conclusion_kind="unsupported",
        title="问题超出当前 Microphone 隐私与证据边界",
        summary=(
            "当前 Beta 只接收隐私确认后的派生相对级别或振幅，不保存原始音频，也不执行语音、人员或绝对声压判断。"
        ),
        uncertainties=("缺少原始波形授权、校准声级计、伦理授权和当前设备真机证据。",),
        forbidden_claims=(
            "不得恢复、转写、识别或监听语音内容。",
            "不得声称校准 SPL、法规合规、听力损伤或人员身份。",
        ),
        next_live_measurement=(
            "若目标是非敏感的相对声学比较，只采集 phyphox Audio amplitude 派生数值，"
            "避开对话并按固定声源、单变量和重复测量方案执行。"
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
                finding_id=f"{prefix}-relative-level-window",
                text=(
                    f"公开轨迹{('前' if prefix == 'early' else '后')} 20 秒窗口的"
                    f"平均相对级别为 {metrics[f'{prefix}_mean_relative_level_db']:.3f} dB_relative，"
                    f"峰值 {metrics[f'{prefix}_peak_relative_level_db']:.3f}，"
                    f"范围 {metrics[f'{prefix}_relative_level_span_db']:.3f}。"
                ),
                evidence_ids=(item.evidence_id,),
            )
        )
    if "late_minus_early_mean_db" in metrics:
        findings.append(
            PublicSensorFinding(
                finding_id="chronological-relative-level-contrast",
                text=(
                    f"后窗口平均相对级别比前窗口高 {metrics['late_minus_early_mean_db']:.3f}，"
                    f"峰值高 {metrics['late_minus_early_peak_db']:.3f}，"
                    f"变化范围宽 {metrics['late_minus_early_span_db']:.3f} dB_relative。"
                ),
                evidence_ids=tuple(item.evidence_id for item in evidence),
            )
        )

    passed = comparison.quality_passed
    return PublicSensorReport(
        conclusion_kind="supported_with_limits" if passed else "live_measurement_required",
        title=(
            "公开 Microphone 派生证据支持有边界的时间窗口结论"
            if passed
            else "公开派生级别质量门未通过，需要真机复核"
        ),
        summary=(
            "Agent 在冻结候选中选择了真实 NoiseCapture 公开数值；服务端专用分析器、来源回归和"
            "预注册相对级别门已完成。结果只描述该轨迹窗口，不能解释为房间位置、声源因果或绝对声压。"
            if passed
            else "服务端来源、节奏或相对级别门未通过，没有扩大公开数据结论。"
        ),
        supported_findings=tuple(findings),
        uncertainties=(
            "两个窗口来自同一条 Android NoiseCapture 轨迹，不是独立手机或独立场景重复。",
            "来源未公开当前回放所需的设备型号和可验证校准状态，也不是 phyphox 导出。",
            "窗口没有房间位置、声源类型或实验控制标签，前后差异不能作因果归因。",
            "手机自动增益和频率响应可能影响用户真机相对幅值。",
        ),
        forbidden_claims=(
            "不得把 dB_relative 解释为校准 dB(A) SPL、法规合规或听力风险。",
            "不得从派生级别恢复、转写或识别原始音频与说话人。",
            "不得声称某个房间位置或声源导致了公开窗口差异。",
            "不得把公开回放计入用户真机 Gate C、市场验证或 production agent_ready。",
        ),
        next_live_measurement=(
            "最后一步是真机 Gate C：避开对话并取得同意，在 phyphox Audio amplitude 中固定舒适"
            "声源、片段、方向和时长；参考位置与比较位置各做 3 次并回到参考位置查漂移，"
            "最终覆盖 3 个场景 × 2 个位置 × 3 重复，只保存派生数值。"
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence),
        source_ids=(evidence[0].dataset_id,),
        sources=(_report_source(evidence[0]),),
    )


async def run_public_microphone_exploration(
    request: PublicSensorExploreRequest,
    *,
    root: Path,
    planner: shared.PlannerCallable | None = None,
) -> PublicSensorExploreResult:
    if request.sensor != "microphone":
        raise shared.PublicSensorExplorationUnavailable("sensor-protocol-mismatch")
    run_id = f"public-{request.sensor}-{uuid4().hex}"
    if not request.privacy_acknowledged:
        return _privacy_result(request, run_id=run_id)

    family = microphone_question_family(request.research_question)
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
        evidence, comparison, tool_trace = load_public_microphone_evidence(
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


def strong_microphone_workflow_route(question: str) -> tuple[str, str]:
    family = microphone_question_family(question)
    return family, _initial_fallback(family)
