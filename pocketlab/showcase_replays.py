from __future__ import annotations

import math

from pydantic import Field

from pocketlab.diagnostics import DiagnosticCaseStore
from pocketlab.general_exploration_engine import (
    commit_general_measurement,
    prepare_reasoned_report,
    select_deterministic_information_candidate,
)
from pocketlab.general_exploration_models import (
    GeneralConditionDraft,
    GeneralExplorationDraft,
    GeneralSensorIntentDraft,
    StrictFrozenModel,
)
from pocketlab.general_exploration_reasoner import (
    render_reasoned_general_report,
    run_general_showcase_reasoner,
)
from pocketlab.general_exploration_state import GeneralExperimentCase
from pocketlab.general_exploration_store import (
    GeneralExplorationCaseCreate,
    GeneralExplorationStore,
)
from pocketlab.general_simulation import (
    GeneralSimulationCaptureMetadata,
    GeneralSimulationMeasurementRequest,
    GeneralSimulationMeasurementResponse,
)
from pocketlab.schemas import (
    DiagnosticAgentResponse,
    DiagnosticCaseCreate,
    DiagnosticHypothesisDraft,
    DiagnosticReasoningReceipt,
    DiagnosticSensorPlanDraft,
    DiagnosticSensorTaskResponse,
    HypothesisAssessmentDraft,
    MeasurementTaskDraft,
)
from pocketlab.sensor_models import (
    SensorChannelDefinition,
    SensorProvenance,
    SensorRecordingCreated,
    SensorRecordingUpload,
    SensorSample,
)
from pocketlab.store import SessionStore

SHOWCASE_MODEL = "server-showcase-replay"
DIAGNOSTIC_SHOWCASE_MARKER = "showcase-replay:diagnostic-v1"
GENERAL_SHOWCASE_TITLE = "灯离远一倍，照度会怎样变化？· 零等待回放"


class GeneralShowcaseAdvanceRequest(StrictFrozenModel):
    expected_revision: int = Field(ge=1)


def is_diagnostic_showcase_case(case) -> bool:
    return DIAGNOSTIC_SHOWCASE_MARKER in case.context


def is_general_showcase_case(case: GeneralExperimentCase) -> bool:
    return (
        case.protocol.title == GENERAL_SHOWCASE_TITLE
        and set(case.protocol.selected_sources) == {"protocol_emulator"}
    )


def create_diagnostic_showcase(
    store: DiagnosticCaseStore,
) -> DiagnosticAgentResponse:
    case = store.create(
        DiagnosticCaseCreate(
            title="洗衣机脱水振动：偏载还是地面放大？",
            problem_statement=(
                "洗衣机进入高速脱水后出现明显周期性振动，需要区分衣物偏载造成的转动不平衡，"
                "还是机身放置与地面耦合放大了振动。"
            ),
            context=(
                "预置演示回放：后台保存两轮已标注加速度序列，页面逐轮走完标准诊断状态机；"
                "不要求用户实际操作洗衣机，也不把回放当作用户家庭现场证据。"
                f" [{DIAGNOSTIC_SHOWCASE_MARKER}]"
            ),
        )
    )
    case = store.commit_initial_plan(
        case.case_id,
        [
            DiagnosticHypothesisDraft(
                statement="衣物偏载造成滚筒转动不平衡",
                rationale="负载质心偏离转轴时，高速脱水会产生与转速相关的周期性惯性力。",
                critical_prediction="只重新均匀分布同一批衣物后，振动 RMS 应明显下降。",
                critical_sensor="accelerometer",
                critical_expected_effect="decrease",
            ),
            DiagnosticHypothesisDraft(
                statement="机身放置与地面耦合放大振动",
                rationale="若主要由支脚或地面传振路径主导，仅改变衣物分布不应产生稳定的大幅下降。",
                critical_prediction="地面和机位不变时，重新分布衣物后的振动仍应保持接近原水平。",
                critical_sensor="accelerometer",
                critical_expected_effect="no_change",
            ),
        ],
        MeasurementTaskDraft(
            title="回放原始偏载工况基线",
            instruction=(
                "演示回放会读取后台冻结的原始偏载工况：同一手机位于洗衣机前方地面固定测点，"
                "记录高速脱水阶段的三轴加速度、振动 RMS 与主频。点击下方按钮即可提交本轮证据。"
            ),
            variable_to_change="不改变任何条件，仅记录当前振动基线",
            controlled_variables=["同一洗衣机", "同一衣物总量", "同一转速", "同一测点", "同一手机姿态", "同一记录时长"],
            required_sensor="accelerometer",
            target_metric_key="selected_axis_rms_m_s2",
            task_kind="baseline",
            target_hypothesis_ids=["h1", "h2"],
            expected_effect="unknown",
            effect_metric="rms",
        ),
        sensor_plan=[
            DiagnosticSensorPlanDraft(
                sensor="accelerometer",
                role="primary",
                rationale="三轴加速度可直接量化脱水阶段的振动强度、周期与重复性。",
                target_metric_key="selected_axis_rms_m_s2",
            )
        ],
    )
    case = store.set_intake_runtime(
        case.case_id,
        transport="deterministic_fallback",
        model=SHOWCASE_MODEL,
        model_requests=0,
        elapsed_ms=0,
        fallback_reason="showcase-replay-zero-model",
    )
    message = (
        "零等待演示已建立：系统预注册了两个竞争解释，并冻结两轮单变量证据。"
        "每次点击“回放本步并立即推进”，都会提交一条标准 Session，并更新假设、下一任务或最终报告。"
    )
    store.set_latest_agent_message(case.case_id, message)
    return DiagnosticAgentResponse(case=case, agent_message=message, model=SHOWCASE_MODEL)


def _diagnostic_showcase_upload(task_id: str) -> SensorRecordingUpload:
    amplitudes = {"task-1": 1.90, "task-2": 0.42, "task-3": 0.44}
    amplitude = amplitudes.get(task_id)
    if amplitude is None:
        raise ValueError("该诊断演示没有更多预置测量。")
    samples = []
    for index in range(600):
        time_s = index / 100.0
        spin = math.sin(2.0 * math.pi * 8.2 * time_s)
        floor = math.sin(2.0 * math.pi * 1.3 * time_s + 0.35)
        samples.append(
            SensorSample(
                timestamp_ms=index * 10.0,
                values={
                    "x": amplitude * spin + 0.06 * floor,
                    "y": 0.34 * amplitude * math.sin(2.0 * math.pi * 8.2 * time_s + 0.8),
                    "z": 9.81 + 0.18 * amplitude * math.sin(2.0 * math.pi * 8.2 * time_s + 1.4),
                },
            )
        )
    labels = {
        "task-1": "SHOWCASE · 原始偏载基线",
        "task-2": "SHOWCASE · 均匀重排衣物",
        "task-3": "SHOWCASE · 均匀工况重复验证",
    }
    return SensorRecordingUpload(
        label=labels[task_id],
        device="PocketLab Showcase Replay",
        sensor="accelerometer",
        notes="服务器冻结的演示序列；不是当前家庭、当前洗衣机或当前手机的实测数据。",
        channels={
            "x": SensorChannelDefinition(unit="m/s^2", description="横向加速度"),
            "y": SensorChannelDefinition(unit="m/s^2", description="纵向加速度"),
            "z": SensorChannelDefinition(unit="m/s^2", description="竖向加速度（含重力）"),
        },
        samples=samples,
        provenance=SensorProvenance(
            source="test_fixture",
            experiment_title="PocketLab Showcase Replay · 三轴加速度",
        ),
    )


def _diagnostic_showcase_next_task(task_id: str) -> MeasurementTaskDraft:
    if task_id == "task-1":
        return MeasurementTaskDraft(
            title="只重新均匀分布同一批衣物",
            instruction=(
                "回放保持洗衣机、衣物总量、转速、地面、手机测点与姿态不变，"
                "只把同一批衣物重新均匀分布，再提交高速脱水阶段加速度。"
            ),
            variable_to_change="衣物由偏载改为均匀分布",
            controlled_variables=["同一洗衣机", "同一衣物总量", "同一转速", "同一地面", "同一测点", "同一手机姿态"],
            required_sensor="accelerometer",
            target_metric_key="selected_axis_rms_m_s2",
            task_kind="control",
            comparison_task_id="task-1",
            target_hypothesis_ids=["h1", "h2"],
            expected_effect="decrease",
            effect_metric="rms",
        )
    if task_id == "task-2":
        return MeasurementTaskDraft(
            title="重复均匀工况，验证下降能否复现",
            instruction=(
                "回放再次保持衣物均匀、地面和测点不变，重复同一高速脱水工况。"
                "如果振动继续保持低位，系统会检查偏载解释是否达到终止门。"
            ),
            variable_to_change="不再改变条件，仅重复均匀工况",
            controlled_variables=["同一洗衣机", "衣物均匀分布", "同一衣物总量", "同一转速", "同一测点", "同一手机姿态"],
            required_sensor="accelerometer",
            target_metric_key="selected_axis_rms_m_s2",
            task_kind="control",
            comparison_task_id="task-2",
            target_hypothesis_ids=["h1", "h2"],
            expected_effect="no_change",
            effect_metric="rms",
        )
    return MeasurementTaskDraft(
        title="保留现场边界，必要时再做机脚水平复核",
        instruction="若现实现场仍有明显振动，应停止演示推断并按厂家要求检查机脚、地面水平和安全状态。",
        variable_to_change="现实现场安全复核",
        controlled_variables=["不把演示数据当作现场证据"],
        required_sensor="accelerometer",
        target_metric_key="selected_axis_rms_m_s2",
        task_kind="exploration",
        target_hypothesis_ids=["h1", "h2"],
        expected_effect="unknown",
        effect_metric="rms",
    )


def _diagnostic_showcase_assessments(task_id: str) -> list[HypothesisAssessmentDraft]:
    if task_id == "task-1":
        return [
            HypothesisAssessmentDraft(
                hypothesis_id="h1",
                status="unverified",
                reasoning="基线确认了稳定周期振动，但单一工况尚不能区分偏载与地面耦合。",
                critical_prediction_tested=False,
            ),
            HypothesisAssessmentDraft(
                hypothesis_id="h2",
                status="unverified",
                reasoning="地面传振仍是可能解释，需要只改变衣物分布的受控对照。",
                critical_prediction_tested=False,
            ),
        ]
    critical_prediction_tested = task_id == "task-3"
    return [
        HypothesisAssessmentDraft(
            hypothesis_id="h1",
            status="supported",
            reasoning=(
                "只改变衣物分布后振动显著下降，且均匀工况重复保持低位，符合偏载造成转动不平衡的关键预测。"
                if task_id == "task-3"
                else "只改变衣物分布后振动显著下降，符合偏载造成转动不平衡的关键预测。"
            ),
            critical_prediction_tested=critical_prediction_tested,
        ),
        HypothesisAssessmentDraft(
            hypothesis_id="h2",
            status="weakened",
            reasoning=(
                "地面、机位和测点均未改变，但振动随衣物分布改变并在重复中保持，地面耦合不是本次回放的主要解释。"
            ),
            critical_prediction_tested=critical_prediction_tested,
        ),
    ]


def _diagnostic_showcase_receipt(task_id: str) -> DiagnosticReasoningReceipt | None:
    if task_id == "task-1":
        return None
    source_fact_ids = ["fact-task-1-1", "fact-task-2-2"]
    if task_id == "task-3":
        source_fact_ids.append("fact-task-3-2")
    return DiagnosticReasoningReceipt(
        model_name=SHOWCASE_MODEL,
        answer_headline=(
            "均匀分布工况的重复回放再次保持低振动，偏载解释已通过终止门"
            if task_id == "task-3"
            else "重新均匀分布衣物后，振动强度显著下降"
        ),
        mechanism_explanation=(
            "回放只改变衣物分布，洗衣机、衣物总量、转速、地面与测点保持不变。"
            "振动随负载分布显著下降，符合质心偏离转轴产生周期惯性力的机制；"
            "固定不变的地面耦合难以单独解释这种受控变化。"
        ),
        confidence="high" if task_id == "task-3" else "medium",
        ranked_hypothesis_ids=["h1", "h2"],
        source_fact_ids=source_fact_ids,
        next_measurement_reason="",
        solution_rationale="优先采用可逆、低风险的均匀装载；现实现场异常持续时再检查水平与机脚并参考厂家说明。",
        recommended_action_ids=[
            "redistribute-balanced-load",
            "repeat-controlled-measurement",
            "check-manufacturer-guidance",
        ],
        transport="deterministic_fallback",
        model_requests=0,
        elapsed_ms=0,
        fallback_reason="showcase-replay-zero-model",
    )


def advance_diagnostic_showcase(
    store: DiagnosticCaseStore,
    recordings: SessionStore,
    *,
    case_id: str,
    task_id: str,
) -> DiagnosticSensorTaskResponse:
    case = store.get(case_id)
    if not is_diagnostic_showcase_case(case):
        raise ValueError("该接口只接受 PocketLab 零等待诊断演示案例。")
    if case.current_task is None or case.current_task.task_id != task_id:
        raise ValueError("演示任务已经推进，请刷新当前案例。")
    upload = _diagnostic_showcase_upload(task_id)
    stored = recordings.create_sensor_recording(upload)
    updated = store.commit_measurement(
        case_id=case_id,
        task_id=task_id,
        session_id=stored.session_id,
        observation_notes="服务器预置演示回放；非当前家庭或当前手机的物理证据。",
        evidence_summary={
            "task-1": "已建立原始偏载工况的高质量周期振动基线。",
            "task-2": "只重新均匀分布衣物后，振动 RMS 相对基线显著下降。",
            "task-3": "均匀工况重复回放保持低振动，下降方向得到复现。",
        }[task_id],
        assessments=_diagnostic_showcase_assessments(task_id),
        next_task=_diagnostic_showcase_next_task(task_id),
        reasoning_receipt=_diagnostic_showcase_receipt(task_id),
    )
    message = {
        "task-1": "基线证据已保存。下一步只改变衣物分布，用同一分析器检验两个竞争解释。",
        "task-2": "振动强度显著下降。系统还需要一次相同条件重复，确认变化不是偶然波动。",
        "task-3": "重复证据保持同一方向，诊断终止向量已满足；报告与可执行建议已经生成。",
    }[task_id]
    store.set_latest_agent_message(case_id, message)
    return DiagnosticSensorTaskResponse(
        session=SensorRecordingCreated(
            session_id=stored.session_id,
            label=stored.upload.label,
            sensor=stored.upload.sensor,
            analysis=stored.analysis,
            created_at=stored.created_at,
        ),
        case=updated,
        agent_message=message,
        model=SHOWCASE_MODEL,
        preview_samples=stored.upload.samples,
    )


def create_general_showcase(store: GeneralExplorationStore) -> GeneralExperimentCase:
    draft = GeneralExplorationDraft(
        title=GENERAL_SHOWCASE_TITLE,
        question="把手机光照传感器与台灯的距离加倍后，照度会明显下降，还是基本不变？",
        objective="compare_conditions",
        requested_claim="relative_comparison",
        independent_variable="手机光照传感器到台灯的距离",
        conditions=(
            GeneralConditionDraft(
                condition_id="reference",
                label="近距离参考位置",
                factor_level="靠近光源",
                instruction="回放固定朝向与环境光，在近距离参考位置记录照度平台。",
            ),
            GeneralConditionDraft(
                condition_id="comparison",
                label="距离加倍位置",
                factor_level="距离加倍",
                instruction="回放只把手机沿光轴移到距离加倍位置，保持朝向和环境光不变。",
            ),
        ),
        sensor_intents=(
            GeneralSensorIntentDraft(
                sensor="light",
                role="primary",
                metric_key="median_illuminance_lx",
                metric_unit="lx",
                measurement_purpose="比较两个距离条件下稳定平台的照度中位数与重复波动。",
            ),
        ),
        alignment="sequential",
        controls=(
            "同一台灯与亮度档位",
            "手机感光面朝向不变",
            "沿同一光轴移动",
            "环境背景光保持不变",
            "每轮使用相同记录时长",
        ),
        expected_pattern="若几何扩散主导，距离加倍后照度应稳定下降，并明显超过同条件重复波动。",
        safety_notes=("本演示不控制真实灯具、不读取摄像头，也不要求用户直视强光源。",),
        privacy_notes=("只使用服务器生成的照度序列，不读取当前手机或环境行为数据。",),
        claim_boundaries=(
            "回放只证明 PocketLab 能执行条件比较、重复性审计、动态充分度判断与终止。",
            "结果不是当前台灯或手机的校准结论，也不替代真实距离测量。",
        ),
    )
    return store.create(GeneralExplorationCaseCreate(draft=draft, source="protocol_emulator"))


def advance_general_showcase(
    store: GeneralExplorationStore,
    *,
    case_id: str,
    expected_revision: int,
    task_id: str,
) -> GeneralSimulationMeasurementResponse:
    before = store.get(case_id)
    if not is_general_showcase_case(before):
        raise ValueError("该接口只接受 PocketLab 零等待光学探索演示。")
    task = before.current_task
    if task is None or task.task_id != task_id or before.revision != expected_revision:
        raise ValueError("演示任务已经推进，请刷新当前探索。")
    previous_evidence_ids = {item.evidence_id for item in before.evidence}
    request = GeneralSimulationMeasurementRequest(
        expected_revision=expected_revision,
        task_id=task_id,
        profile="inverse_square_light",
        controls_confirmed=True,
    )
    prepared = store.prepare_simulated_submission(case_id, request)
    if prepared.report is not None and prepared.report.outcome == "completed_descriptive":
        reasoning = run_general_showcase_reasoner(prepared)
        report = render_reasoned_general_report(
            prepared.report,
            reasoning.request,
            reasoning.receipt,
        )
        reasoned = prepare_reasoned_report(prepared, report)
        updated = commit_general_measurement(
            reasoned,
            reasoning_receipt=reasoning.receipt,
        )
    else:
        updated = commit_general_measurement(
            prepared,
            selected_candidate_id=select_deterministic_information_candidate(prepared),
            selection_source="deterministic_policy",
        )
    store.save_committed(updated, expected_revision=prepared.base_case.revision)
    evidence = tuple(
        item
        for item in updated.evidence
        if item.evidence_id not in previous_evidence_ids
        and item.lineage.source == "protocol_emulator"
    )
    return GeneralSimulationMeasurementResponse(
        case=updated,
        evidence=evidence,
        simulation=GeneralSimulationCaptureMetadata(
            profile=request.profile,
            sensors=task.sensors,
            recording_ids=tuple(item.lineage.recording_id for item in evidence),
        ),
    )
