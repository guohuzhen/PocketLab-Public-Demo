from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pocketlab.public_sensor_exploration as shared
from pocketlab.public_accelerometer_tools import (
    ANALYZE_TOOL_ID,
    CADENCE_PROTOCOL_ID,
    CADENCE_RECORDING_IDS,
    COMPARE_CADENCE_TOOL_ID,
    COMPARE_ELEVATOR_TOOL_ID,
    COMPARE_VIBRATION_TOOL_ID,
    DATASET_ID,
    ELEVATOR_PROTOCOL_ID,
    ELEVATOR_RECORDING_IDS,
    SEGMENT_ELEVATOR_TOOL_ID,
    VIBRATION_PROTOCOL_ID,
    VIBRATION_RECORDING_IDS,
    ProtocolId,
    load_public_accelerometer_evidence,
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

PROTOCOL_VERSION = "1.0.0"
_LIVE_ID = "request_live_accelerometer"
_UNSUPPORTED_ID = "stop_unsupported_accelerometer"
_PRIVACY_ID = "privacy_acknowledgement_required"


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: ProtocolId
    name: str
    finish_id: str
    candidates: tuple[PublicSensorPlanCandidate, ...]
    selection_policy: tuple[str, ...]


def _cadence_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id="inspect_lower_stair_cadence",
            title="检查第一段公开楼梯步频",
            server_reason="只分析 AS4 lower 窗口的主频、动态轴 RMS 与频谱信噪比。",
            rationale_code="match_lower_stair_cadence",
            recording_ids=(CADENCE_RECORDING_IDS[0],),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_CADENCE_TOOL_ID),
            result_code="lower_stair_cadence",
        ),
        PublicSensorPlanCandidate(
            candidate_id="inspect_middle_stair_cadence",
            title="检查第二段公开楼梯步频",
            server_reason="只分析 AS4 middle 窗口的主频、动态轴 RMS 与频谱信噪比。",
            rationale_code="match_middle_stair_cadence",
            recording_ids=(CADENCE_RECORDING_IDS[1],),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_CADENCE_TOOL_ID),
            result_code="middle_stair_cadence",
        ),
        PublicSensorPlanCandidate(
            candidate_id="compare_three_stair_cadence_repeats",
            title="比较三段公开楼梯步频",
            server_reason="比较三个冻结 AS4 上行窗口，并用频带、SNR 与 CV 门检查重复性。",
            rationale_code="compare_stair_cadence_repeats",
            recording_ids=CADENCE_RECORDING_IDS,
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_CADENCE_TOOL_ID),
            result_code="stair_cadence_repeats",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox 加速度步频实验",
            server_reason="公开 AS4 没有草地、瓷砖或当前用户条件，转入成组真机测量设计。",
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止医疗、身份或越权步态请求",
            server_reason="医疗诊断、人物识别、跌倒风险和身份推断超出当前实验与隐私边界。",
            rationale_code="unsupported_claim_boundary",
            terminal=True,
            result_code="unsupported",
        ),
    )


def _elevator_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id="inspect_full_elevator_ascent",
            title="分割一段完整公开电梯上行",
            server_reason="只分析 AS5 full ascent，检测正加速、稳定中段与反向减速的顺序。",
            rationale_code="match_full_elevator_ascent",
            recording_ids=(ELEVATOR_RECORDING_IDS[0],),
            tool_ids=(ANALYZE_TOOL_ID, SEGMENT_ELEVATOR_TOOL_ID, COMPARE_ELEVATOR_TOOL_ID),
            result_code="full_elevator_ascent",
        ),
        PublicSensorPlanCandidate(
            candidate_id="compare_half_elevator_ascents",
            title="比较两段半程公开电梯上行",
            server_reason="比较 AS5 lower/upper half 两段阶段序列，检查算法对不同短行程的响应。",
            rationale_code="compare_half_elevator_ascents",
            recording_ids=ELEVATOR_RECORDING_IDS[1:],
            tool_ids=(ANALYZE_TOOL_ID, SEGMENT_ELEVATOR_TOOL_ID, COMPARE_ELEVATOR_TOOL_ID),
            result_code="half_elevator_ascents",
        ),
        PublicSensorPlanCandidate(
            candidate_id="compare_three_elevator_ascents",
            title="比较三段公开电梯阶段序列",
            server_reason="对三个冻结 AS5 行程运行相同阶段门，验证正加速—稳定—减速顺序。",
            rationale_code="compare_elevator_phase_repeats",
            recording_ids=ELEVATOR_RECORDING_IDS,
            tool_ids=(ANALYZE_TOOL_ID, SEGMENT_ELEVATOR_TOOL_ID, COMPARE_ELEVATOR_TOOL_ID),
            result_code="elevator_phase_repeats",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox 电梯加速度实验",
            server_reason="公开记录不能回答当前电梯、手机朝向或具体楼层，转入真机重复。",
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止位移积分或人员监控请求",
            server_reason="无姿态校正的位移/楼层积分、人员监控和安全认证超出协议权限。",
            rationale_code="unsupported_claim_boundary",
            terminal=True,
            result_code="unsupported",
        ),
    )


def _vibration_candidates() -> tuple[PublicSensorPlanCandidate, ...]:
    return (
        PublicSensorPlanCandidate(
            candidate_id="inspect_stationary_acceleration_anchor",
            title="检查公开静止加速度锚点",
            server_reason="只分析 AS7 floor anchor 的动态轴 RMS，验证静止噪声是否足够低。",
            rationale_code="match_stationary_acceleration",
            recording_ids=(VIBRATION_RECORDING_IDS[0],),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_VIBRATION_TOOL_ID),
            result_code="stationary_acceleration_anchor",
        ),
        PublicSensorPlanCandidate(
            candidate_id="inspect_handheld_acceleration_response",
            title="检查公开单条高运动响应记录",
            server_reason=(
                "只分析 AS7 handheld interval 的 RMS 与峰峰值；当问题要求排除静止锚点、"
                "只保留更高能量的单条记录时使用。"
            ),
            rationale_code="match_handheld_acceleration",
            recording_ids=(VIBRATION_RECORDING_IDS[1],),
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_VIBRATION_TOOL_ID),
            result_code="handheld_acceleration_response",
        ),
        PublicSensorPlanCandidate(
            candidate_id="compare_acceleration_motion_states",
            title="比较公开静止与手持加速度",
            server_reason="用同一 acquisition 的静止—手持 RMS 比验证测量链响应，但不诊断设备故障。",
            rationale_code="compare_acceleration_motion_states",
            recording_ids=VIBRATION_RECORDING_IDS,
            tool_ids=(ANALYZE_TOOL_ID, COMPARE_VIBRATION_TOOL_ID),
            result_code="acceleration_motion_state_pair",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求新的或当前用户 phyphox 设备振动对照",
            server_reason=(
                "偏载、松动、传振和结构放大，或任何要求现在/未来重新采集的数据，"
                "必须由当前设备的单变量真机对照判定。"
            ),
            rationale_code="request_live_device_evidence",
            terminal=True,
            result_code="live_measurement_required",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_UNSUPPORTED_ID,
            title="停止安全事故或无证据故障断言",
            server_reason="漏电、漏水、焦糊味、结构危险或无真机证据的故障定性不能由公开回放处理。",
            rationale_code="unsupported_claim_boundary",
            terminal=True,
            result_code="unsupported",
        ),
    )


_SPECS = {
    CADENCE_PROTOCOL_ID: ProtocolSpec(
        protocol_id=CADENCE_PROTOCOL_ID,
        name="步频",
        finish_id="finish_cadence_report",
        candidates=_cadence_candidates(),
        selection_policy=(
            "明确只问第一段/lower 公开楼梯记录时选择 inspect_lower_stair_cadence。",
            "明确只问第二段/middle 公开楼梯记录时选择 inspect_middle_stair_cadence。",
            "询问公开楼梯重复性、平均步频或未限定单段时选择 compare_three_stair_cadence_repeats。",
            "询问用户当前手机、草地/瓷砖等新路面或要求新采集时选择 request_live_accelerometer。",
            "医疗诊断、人物身份、跌倒风险或越权请求选择 stop_unsupported_accelerometer。",
        ),
    ),
    ELEVATOR_PROTOCOL_ID: ProtocolSpec(
        protocol_id=ELEVATOR_PROTOCOL_ID,
        name="电梯阶段",
        finish_id="finish_elevator_report",
        candidates=_elevator_candidates(),
        selection_policy=(
            "明确只分析一段完整/full 行程时选择 inspect_full_elevator_ascent。",
            "明确比较两段半程/half 行程时选择 compare_half_elevator_ascents。",
            "询问阶段重复性、三段对照或未限定单段时选择 compare_three_elevator_ascents。",
            "询问当前电梯、具体楼层或要求新采集时选择 request_live_accelerometer。",
            "要求双积分位移/楼层、人员监控、安全认证或越权请求选择 stop_unsupported_accelerometer。",
        ),
    ),
    VIBRATION_PROTOCOL_ID: ProtocolSpec(
        protocol_id=VIBRATION_PROTOCOL_ID,
        name="振动响应",
        finish_id="finish_vibration_response_report",
        candidates=_vibration_candidates(),
        selection_policy=(
            "明确只问静止噪声或基线时选择 inspect_stationary_acceleration_anchor。",
            (
                "明确只问公开手持运动响应、更高能量的单条记录，或要求排除 "
                "quiet/calm/stationary 锚点时选择 inspect_handheld_acceleration_response。"
            ),
            (
                "只有明确要求同时比较、对照或区分公开静止与运动两种状态时，才选择 "
                "compare_acceleration_motion_states；单条选择不能因为提到被排除的另一条"
                "而升级为成对比较。"
            ),
            (
                "询问洗衣机/风扇等当前设备原因、偏载、松动、传振，或要求 "
                "new/fresh/future/tomorrow collection 而不是分析档案时选择 "
                "request_live_accelerometer。"
            ),
            "安全事故、无证据故障定性或越权请求选择 stop_unsupported_accelerometer。",
        ),
    ),
}


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _common_unsupported(lower: str) -> bool:
    return _contains_any(
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
            "市场验证",
        ),
    )


def _live_terms(lower: str) -> bool:
    return _contains_any(
        lower,
        (
            "我的手机",
            "当前手机",
            "现在手机",
            "此刻",
            "现场",
            "刚才采集",
            "真机数据",
            "live measurement",
            "live data",
            "real time",
            "realtime",
        ),
    )


def accelerometer_question_family(protocol_id: ProtocolId, question: str) -> str:
    lower = question.casefold()
    if _common_unsupported(lower):
        return "unsupported"
    if protocol_id == CADENCE_PROTOCOL_ID:
        if _contains_any(
            lower,
            (
                "医疗",
                "诊断疾病",
                "帕金森",
                "跌倒风险",
                "识别是谁",
                "人物身份",
                "medical diagnosis",
                "identify the person",
            ),
        ):
            return "unsupported"
        if _live_terms(lower) or _contains_any(
            lower, ("草地", "瓷砖", "路面", "柏油", "户外", "new surface")
        ):
            return "live"
        if _contains_any(lower, ("第一段", "lower", "first segment")):
            return "lower"
        if _contains_any(lower, ("第二段", "middle", "second segment")):
            return "middle"
        return "repeats"
    if protocol_id == ELEVATOR_PROTOCOL_ID:
        if _contains_any(
            lower,
            (
                "双积分",
                "精确位移",
                "精确楼层",
                "人员监控",
                "安全认证",
                "double integration",
                "exact floor",
                "surveillance",
            ),
        ):
            return "unsupported"
        if _live_terms(lower) or _contains_any(lower, ("当前电梯", "这部电梯", "我乘坐")):
            return "live"
        if _contains_any(lower, ("半程", "两段短", "half ascent", "two short")):
            return "half_pair"
        if _contains_any(lower, ("单段", "一段完整", "full ascent", "single ride")):
            return "full"
        return "repeats"
    if _contains_any(
        lower,
        ("漏电", "漏水", "焦糊味", "结构危险", "起火", "安全认证", "fire hazard"),
    ):
        return "unsupported"
    if _live_terms(lower) or _contains_any(
        lower,
        (
            "洗衣机",
            "风扇",
            "冰箱",
            "设备故障",
            "偏载",
            "松动",
            "传振",
            "结构放大",
            "appliance",
            "imbalance",
            "loose",
        ),
    ):
        return "live"
    if _contains_any(
        lower,
        (
            "比较",
            "对照",
            "区分",
            "静止与",
            "静止和",
            "compare",
            "contrast",
            "stationary versus",
        ),
    ):
        return "pair"
    if _contains_any(lower, ("静止", "基线", "噪声", "stationary", "idle")):
        return "stationary"
    if _contains_any(lower, ("只看手持", "handheld only", "运动段")):
        return "handheld"
    return "pair"


def _initial_fallback(protocol_id: ProtocolId, family: str) -> str:
    mappings = {
        CADENCE_PROTOCOL_ID: {
            "lower": "inspect_lower_stair_cadence",
            "middle": "inspect_middle_stair_cadence",
            "repeats": "compare_three_stair_cadence_repeats",
            "live": _LIVE_ID,
            "unsupported": _UNSUPPORTED_ID,
        },
        ELEVATOR_PROTOCOL_ID: {
            "full": "inspect_full_elevator_ascent",
            "half_pair": "compare_half_elevator_ascents",
            "repeats": "compare_three_elevator_ascents",
            "live": _LIVE_ID,
            "unsupported": _UNSUPPORTED_ID,
        },
        VIBRATION_PROTOCOL_ID: {
            "stationary": "inspect_stationary_acceleration_anchor",
            "handheld": "inspect_handheld_acceleration_response",
            "pair": "compare_acceleration_motion_states",
            "live": _LIVE_ID,
            "unsupported": _UNSUPPORTED_ID,
        },
    }
    return mappings[protocol_id][family]


def _privacy_result(
    request: PublicSensorExploreRequest,
    spec: ProtocolSpec,
    *,
    run_id: str,
) -> PublicSensorExploreResult:
    candidate_ids = (_PRIVACY_ID, _UNSUPPORTED_ID)
    digest = hashlib.sha256(
        json.dumps(
            {
                "run_id": run_id,
                "sensor": request.sensor,
                "protocol_id": spec.protocol_id,
                "question": request.research_question,
                "privacy_acknowledged": False,
                "candidate_ids": candidate_ids,
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
        candidate_ids=candidate_ids,
        selected_candidate_id=_PRIVACY_ID,
        fallback_candidate_id=_PRIVACY_ID,
        rationale_code="privacy_not_acknowledged",
        source="strong_workflow_fallback",
        outcome="fallback",
        fallback_reason="privacy-not-acknowledged",
    )
    report = PublicSensorReport(
        conclusion_kind="privacy_acknowledgement_required",
        title=f"需要确认本地公开{spec.name}回放边界",
        summary="确认前不会读取 NIST 序列、调用 Planner、执行分析或写入账号。",
        uncertainties=(f"尚未读取任何公开{spec.name}证据。",),
        forbidden_claims=("不能把 NIST Pixel XL 数据冒充为你的 phyphox 真机数据。",),
        next_live_measurement="确认后可运行本地公开回放；用户真机 Gate C 仍需单独采集。",
    )
    return PublicSensorExploreResult(
        sensor="accelerometer",
        protocol_id=spec.protocol_id,
        run_id=run_id,
        research_question=request.research_question,
        execution_status="limited",
        selected_route_id=_PRIVACY_ID,
        planner_status="fallback",
        planner_trace=(trace,),
        report=report,
    )


def _terminal_report(protocol_id: ProtocolId, family: str) -> PublicSensorReport:
    if family == "live":
        plans = {
            CADENCE_PROTOCOL_ID: (
                "在 phyphox 打开“加速度（不含重力）”或“加速度”：手机固定在同一口袋/腰包，"
                "每种安全路面稳定行走 20 秒，每条件至少 3 次；保持速度范围、路线、手机方向一致。"
            ),
            ELEVATOR_PROTOCOL_ID: (
                "在 phyphox 打开“加速度”：手机固定且全程保持方向，从开门前静止记录到到站后，"
                "同一楼层区间至少重复 3 次；同时记下方向但不要把楼层标签送入阶段分析器。"
            ),
            VIBRATION_PROTOCOL_ID: (
                "在 phyphox 打开“加速度（不含重力）”：同一测点先记录原工况，再一次只改变偏载、"
                "支脚接触或隔振中的一个条件，每条件至少 3 次；若漏电、漏水、焦糊味或剧烈位移立即停止。"
            ),
        }
        return PublicSensorReport(
            conclusion_kind="live_measurement_required",
            title="这个问题需要你的 phyphox 加速度真机数据",
            summary="公开 Pixel XL 回放与当前设备、路面或电梯不是同一实验，Agent 已停止证据替代并给出受控采集方案。",
            uncertainties=("尚无当前手机、目标条件和三次重复的绑定证据。",),
            forbidden_claims=("不得用公开 NIST 记录推断当前用户设备、路面或电梯的结论。",),
            next_live_measurement=plans[protocol_id],
        )
    forbidden = {
        CADENCE_PROTOCOL_ID: (
            "不得把步频候选解释为医疗诊断、身份识别或跌倒风险评分。",
        ),
        ELEVATOR_PROTOCOL_ID: (
            "不得在无姿态与漂移校正时双积分为精确位移或楼层。",
            "不得用于人员监控或电梯安全认证。",
        ),
        VIBRATION_PROTOCOL_ID: (
            "不得用公开手持运动记录诊断偏载、松动、传振或结构故障。",
            "出现电气、漏水、焦糊或剧烈机械风险时应立即停止实验。",
        ),
    }
    return PublicSensorReport(
        conclusion_kind="unsupported",
        title="问题超出当前加速度实验的安全与证据边界",
        summary="服务端没有可授权该结论的候选工具或证据，已在读取公开序列前停止。",
        uncertainties=("缺少与请求匹配的受控真机条件、参考标签或安全授权。",),
        forbidden_claims=forbidden[protocol_id],
        next_live_measurement="可把问题改写为公开记录的有边界分析，或按安全说明采集用户真机对照。",
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
    protocol_id: ProtocolId,
    evidence: tuple[PublicSensorEvidenceSnapshot, ...],
    comparison: PublicSensorComparison,
    *,
    selected_action: str,
) -> PublicSensorReport:
    metrics = {item.key: item.value for item in comparison.metrics}
    evidence_ids = tuple(item.evidence_id for item in evidence)
    findings: list[PublicSensorFinding] = []
    if protocol_id == CADENCE_PROTOCOL_ID:
        if "mean_cadence_hz" in metrics:
            findings.extend(
                (
                    PublicSensorFinding(
                        finding_id="public-stair-cadence-mean",
                        text=(
                            f"三段公开楼梯步频候选均值为 {metrics['mean_cadence_hz']:.3f} Hz "
                            f"（约 {metrics['mean_cadence_steps_min']:.1f} steps/min）。"
                        ),
                        evidence_ids=evidence_ids,
                    ),
                    PublicSensorFinding(
                        finding_id="public-stair-cadence-repeatability",
                        text=f"三段主频变异系数为 {metrics['cadence_cv_ratio']:.3f}。",
                        evidence_ids=evidence_ids,
                    ),
                )
            )
        else:
            key = next(item for item in metrics if item.endswith("_cadence_hz"))
            findings.append(
                PublicSensorFinding(
                    finding_id="public-stair-cadence-single",
                    text=f"该公开楼梯段的周期主频候选为 {metrics[key]:.3f} Hz。",
                    evidence_ids=evidence_ids,
                )
            )
        uncertainties = (
            "三段来自同一 Pixel XL、同一 AS4 acquisition，不是独立手机重复。",
            "公开记录只有楼梯上行，没有草地或瓷砖条件，不能回答路面效应。",
            "主频候选未与人工逐步标注逐步对齐。",
        )
        forbidden = (
            "不得声称草地、瓷砖或其他路面导致步频变化。",
            "不得把结果解释为医疗、身份或跌倒风险结论。",
        )
        next_live = (
            "真机 Gate C：同一手机固定方式下，在安全的草地、瓷砖和楼梯分别记录 20 秒，"
            "每条件至少 3 次并记录速度范围；服务端再比较步频、RMS 与重复性。"
        )
        title = "公开楼梯重复记录支持有边界的步频候选分析"
    elif protocol_id == ELEVATOR_PROTOCOL_ID:
        findings.extend(
            (
                PublicSensorFinding(
                    finding_id="public-elevator-phase-count",
                    text=f"{metrics['phase_sequences_detected']:.0f} 条公开电梯记录通过正加速—稳定—减速顺序门。",
                    evidence_ids=evidence_ids,
                ),
                PublicSensorFinding(
                    finding_id="public-elevator-phase-excursions",
                    text=(
                        f"平均正向偏移 {metrics['mean_positive_excursion_m_s2']:.3f} m/s²，"
                        f"平均反向偏移 {metrics['mean_negative_excursion_m_s2']:.3f} m/s²。"
                    ),
                    evidence_ids=evidence_ids,
                ),
            )
        )
        uncertainties = (
            "记录来自同一 Pixel XL 和 AS5 acquisition，不是独立手机重复。",
            "阶段使用加速度模长与固定阈值，手机转动可能模拟阶段偏移。",
            "隐藏高度标签未送入分析器，结果没有校准楼层或位移。",
        )
        forbidden = (
            "不得把加速度双积分为精确位移、速度或楼层。",
            "不得把公开结果作为当前电梯性能或安全认证。",
        )
        next_live = (
            "真机 Gate C：手机保持固定方向，从电梯静止前记录到到站后；同一楼层区间至少 3 次，"
            "服务端只比较阶段顺序、时刻和幅值，不从加速度积分楼层。"
        )
        title = "公开电梯记录支持有边界的运动阶段分割"
    else:
        if "stationary_rms_m_s2" in metrics:
            findings.append(
                PublicSensorFinding(
                    finding_id="public-stationary-acceleration-rms",
                    text=f"公开静止锚点动态轴 RMS 为 {metrics['stationary_rms_m_s2']:.4f} m/s²。",
                    evidence_ids=(evidence[0].evidence_id,),
                )
            )
        if "handheld_rms_m_s2" in metrics:
            handheld = next(
                item for item in evidence if item.recording_id == VIBRATION_RECORDING_IDS[1]
            )
            findings.append(
                PublicSensorFinding(
                    finding_id="public-handheld-acceleration-rms",
                    text=f"公开手持窗口动态轴 RMS 为 {metrics['handheld_rms_m_s2']:.3f} m/s²。",
                    evidence_ids=(handheld.evidence_id,),
                )
            )
        if "motion_to_stationary_rms_ratio" in metrics:
            findings.append(
                PublicSensorFinding(
                    finding_id="public-motion-state-separation",
                    text=f"手持与静止 RMS 比为 {metrics['motion_to_stationary_rms_ratio']:.1f}。",
                    evidence_ids=evidence_ids,
                )
            )
        uncertainties = (
            "静止与手持窗口来自同一 Pixel XL、同一次 AS7 acquisition。",
            "公开手持运动不是家电振动，也没有偏载、松动或隔振标签。",
            "该结果只验证测量链响应，不能诊断设备原因。",
        )
        forbidden = (
            "不得声称公开记录证明具体设备存在偏载、松动、传振或结构放大。",
            "不得把公开回放计入当前用户真机 Gate C。",
        )
        next_live = (
            "真机 Gate C：固定同一测点，先测原工况，再一次只改变一个候选原因；"
            "每条件至少 3 次，比较 RMS、主频与谐波。出现漏电、漏水、焦糊味或剧烈位移立即停止。"
        )
        title = "公开加速度记录支持有边界的运动响应验证"

    supported = selected_action == _SPECS[protocol_id].finish_id and comparison.quality_passed
    return PublicSensorReport(
        conclusion_kind="supported_with_limits" if supported else "live_measurement_required",
        title=title if supported else "公开加速度证据不足，需要真机复核",
        summary=(
            "Agent 在服务端冻结候选中选择了与协议匹配的真实公开手机记录；确定性分析、"
            "专用质量门和服务端终止已完成。结论仅覆盖该公开记录，不是用户真机结论。"
            if supported
            else "服务端质量门未通过或已转入真机复核，没有扩大公开数据结论。"
        ),
        supported_findings=tuple(findings),
        uncertainties=uncertainties,
        forbidden_claims=forbidden,
        next_live_measurement=next_live,
        evidence_ids=evidence_ids,
        source_ids=(evidence[0].dataset_id,),
        sources=(_report_source(evidence[0]),),
    )


async def run_public_accelerometer_exploration(
    request: PublicSensorExploreRequest,
    *,
    root: Path,
    planner: shared.PlannerCallable | None = None,
) -> PublicSensorExploreResult:
    if request.sensor != "accelerometer" or request.protocol_id not in _SPECS:
        raise shared.PublicSensorExplorationUnavailable(
            "accelerometer-protocol-not-registered"
        )
    protocol_id: ProtocolId = request.protocol_id  # type: ignore[assignment]
    spec = _SPECS[protocol_id]
    run_id = f"public-accelerometer-{uuid4().hex}"
    if not request.privacy_acknowledged:
        return _privacy_result(request, spec, run_id=run_id)

    family = accelerometer_question_family(protocol_id, request.research_question)
    fallback_id = _initial_fallback(protocol_id, family)
    first_request = shared._planner_request(
        run_id=run_id,
        step=1,
        operation="select_evidence_route",
        request=request,
        evidence_view=PublicSensorEvidenceView(),
        candidates=spec.candidates,
        fallback_candidate_id=fallback_id,
        protocol_id=protocol_id,
        protocol_version=PROTOCOL_VERSION,
        selection_policy=spec.selection_policy,
    )
    if family in {"live", "unsupported"}:
        candidate = next(item for item in spec.candidates if item.candidate_id == fallback_id)
        first_trace = PublicSensorPlannerTrace(
            step=1,
            operation="select_evidence_route",
            request_sha256=first_request.request_sha256,
            candidate_ids=tuple(item.candidate_id for item in spec.candidates),
            selected_candidate_id=fallback_id,
            fallback_candidate_id=fallback_id,
            rationale_code=candidate.rationale_code,
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason="policy-boundary",
        )
        report = _terminal_report(protocol_id, family)
        return PublicSensorExploreResult(
            sensor="accelerometer",
            protocol_id=protocol_id,
            run_id=run_id,
            research_question=request.research_question,
            execution_status="unsupported" if family == "unsupported" else "limited",
            selected_route_id=fallback_id,
            planner_status="fallback",
            planner_trace=(first_trace,),
            report=report,
        )

    selected_id, first_trace = await shared._select_candidate(first_request, planner=planner)
    selected = next(item for item in spec.candidates if item.candidate_id == selected_id)
    if selected.terminal:
        terminal_family = "live" if selected_id == _LIVE_ID else "unsupported"
        report = _terminal_report(protocol_id, terminal_family)
        return PublicSensorExploreResult(
            sensor="accelerometer",
            protocol_id=protocol_id,
            run_id=run_id,
            research_question=request.research_question,
            execution_status="unsupported" if terminal_family == "unsupported" else "limited",
            selected_route_id=selected_id,
            planner_status=shared._planner_status((first_trace,)),
            planner_trace=(first_trace,),
            report=report,
        )

    try:
        evidence, comparison, tool_trace = load_public_accelerometer_evidence(
            root.resolve() / DATASET_ID,
            protocol_id,
            selected.recording_ids,
        )
    except (OSError, StopIteration, ValueError) as exc:
        raise shared.PublicSensorExplorationUnavailable(
            "accelerometer-source-or-tool-validation-failed"
        ) from exc

    follow_candidates = (
        PublicSensorPlanCandidate(
            candidate_id=spec.finish_id,
            title=f"形成有边界的{spec.name}报告",
            server_reason="服务端确定性质量门通过时，允许形成引用公开证据的受限报告。",
            rationale_code="evidence_quality_sufficient",
            terminal=True,
            result_code="finish_report",
        ),
        PublicSensorPlanCandidate(
            candidate_id=_LIVE_ID,
            title="请求用户 phyphox 真机复核",
            server_reason="质量门不足时只能请求真机证据，不能扩大公开数据结论。",
            rationale_code="evidence_quality_insufficient",
            terminal=True,
            result_code="live_measurement_required",
        ),
    )
    follow_id = spec.finish_id if comparison.quality_passed else _LIVE_ID
    minimum_confidence: Literal["low", "medium", "high"] = (
        "low"
        if any(item.analysis.confidence == "low" for item in evidence)
        else "medium"
        if any(item.analysis.confidence == "medium" for item in evidence)
        else "high"
    )
    follow_request = shared._planner_request(
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
        fallback_candidate_id=follow_id,
        protocol_id=protocol_id,
        protocol_version=PROTOCOL_VERSION,
        selection_policy=(
            "服务端质量门通过时结束并形成受限报告。",
            "质量门失败时请求真机复核；模型没有终止或改写阈值权限。",
        ),
    )
    follow_candidate = next(
        item for item in follow_candidates if item.candidate_id == follow_id
    )
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
        protocol_id,
        evidence,
        comparison,
        selected_action=follow_id,
    )
    return PublicSensorExploreResult(
        sensor="accelerometer",
        protocol_id=protocol_id,
        run_id=run_id,
        research_question=request.research_question,
        execution_status=(
            "completed" if report.conclusion_kind == "supported_with_limits" else "limited"
        ),
        selected_route_id=follow_id,
        planner_status=shared._planner_status(traces),
        planner_trace=traces,
        tool_trace=tool_trace,
        evidence=evidence,
        comparison=comparison,
        report=report,
    )


def strong_accelerometer_workflow_route(
    protocol_id: ProtocolId, question: str
) -> tuple[str, str]:
    family = accelerometer_question_family(protocol_id, question)
    return family, _initial_fallback(protocol_id, family)
