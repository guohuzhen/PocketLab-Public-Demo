from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from pocketlab.general_exploration_models import StrictFrozenModel
from pocketlab.general_exploration_state import GeneralExperimentCase
from pocketlab.public_light_exploration import run_public_light_exploration
from pocketlab.public_light_models import PublicLightExploreRequest, PublicLightExploreResult
from pocketlab.public_pressure_agent_models import (
    PublicPressureExploreRequest,
    PublicPressureExploreResult,
)
from pocketlab.public_pressure_exploration import run_public_pressure_exploration
from pocketlab.public_replay_dataset import list_public_replay_catalog
from pocketlab.public_sensor_agent_models import (
    PublicSensorAgentKind,
    PublicSensorExploreRequest,
    PublicSensorExploreResult,
)
from pocketlab.public_sensor_exploration import run_public_sensor_exploration
from pocketlab.sensor_models import SensorKind

_ID_PATTERN = r"^[a-z0-9][a-z0-9._:-]{2,119}$"


class GeneralPublicComponentValidation(ValueError):
    pass


class GeneralPublicComponentDescriptor(StrictFrozenModel):
    component_id: str = Field(pattern=_ID_PATTERN)
    sensor: SensorKind
    protocol_id: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=3, max_length=180)
    dataset_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    semantic_relationship: Literal["analogue_only"] = "analogue_only"
    supported_scope: str = Field(min_length=10, max_length=700)
    missing_scope: str = Field(min_length=10, max_length=700)
    privacy_ack_required: Literal[True] = True
    counts_as_general_case_evidence: Literal[False] = False
    joint_inference_allowed: Literal[False] = False
    gate_c_eligible: Literal[False] = False

    @model_validator(mode="after")
    def identity_is_closed(self):
        if self.sensor == "bluetooth":
            raise ValueError("Bluetooth has no public numeric component")
        if len(self.dataset_ids) != len(set(self.dataset_ids)):
            raise ValueError("component dataset IDs must be unique")
        return self


class GeneralPublicComponentCatalog(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(pattern=_ID_PATTERN)
    case_revision: int = Field(ge=1)
    protocol_id: str = Field(pattern=_ID_PATTERN)
    requested_sensors: tuple[SensorKind, ...] = Field(min_length=1, max_length=8)
    components: tuple[GeneralPublicComponentDescriptor, ...] = Field(
        min_length=1,
        max_length=12,
    )
    exact_protocol_match_available: Literal[False] = False
    synchronized_joint_dataset_available: Literal[False] = False
    can_complete_general_case_without_live_data: Literal[False] = False
    boundary_messages: tuple[str, ...] = Field(min_length=3, max_length=8)

    @model_validator(mode="after")
    def catalog_covers_requested_sensors(self):
        if len(self.requested_sensors) != len(set(self.requested_sensors)):
            raise ValueError("requested sensors must be unique")
        if "bluetooth" in self.requested_sensors:
            raise ValueError("Bluetooth cannot enter the public numeric catalog")
        component_ids = [item.component_id for item in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component IDs must be unique")
        covered = {item.sensor for item in self.components}
        if covered != set(self.requested_sensors):
            raise ValueError("public components must cover the exact requested sensor set")
        return self


class GeneralPublicComponentRunRequest(StrictFrozenModel):
    expected_revision: int = Field(ge=1)
    component_id: str = Field(pattern=_ID_PATTERN)
    privacy_acknowledged: bool = False


class GeneralPublicPlannerStep(StrictFrozenModel):
    step: int = Field(ge=1, le=2)
    source: Literal["agent", "strong_workflow_fallback"]
    outcome: Literal["accepted", "fallback"]
    selected_candidate_id: str = Field(pattern=_ID_PATTERN)
    rationale_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")


class GeneralPublicComponentRunResult(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(pattern=_ID_PATTERN)
    case_revision: int = Field(ge=1)
    component_id: str = Field(pattern=_ID_PATTERN)
    sensor: SensorKind
    protocol_id: str = Field(pattern=_ID_PATTERN)
    run_id: str = Field(pattern=_ID_PATTERN)
    execution_status: Literal["completed", "limited", "unsupported"]
    conclusion_kind: str = Field(pattern=_ID_PATTERN)
    title: str = Field(min_length=3, max_length=180)
    summary: str = Field(min_length=10, max_length=1_600)
    selected_route_id: str = Field(pattern=_ID_PATTERN)
    planner_status: Literal["accepted", "fallback", "mixed"]
    planner_steps: tuple[GeneralPublicPlannerStep, ...] = Field(min_length=1, max_length=2)
    tool_ids: tuple[str, ...] = Field(default=(), max_length=8)
    evidence_count: int = Field(ge=0, le=8)
    source_ids: tuple[str, ...] = Field(default=(), max_length=4)
    findings: tuple[str, ...] = Field(default=(), max_length=8)
    uncertainties: tuple[str, ...] = Field(min_length=1, max_length=16)
    forbidden_claims: tuple[str, ...] = Field(min_length=1, max_length=16)
    next_live_measurement: str | None = Field(default=None, max_length=1_200)
    semantic_relationship: Literal["analogue_only"] = "analogue_only"
    counts_as_general_case_evidence: Literal[False] = False
    case_revision_changed: Literal[False] = False
    joint_inference_allowed: Literal[False] = False
    gate_c_credited_records: Literal[0] = 0
    gate_e_status: Literal["not_evaluated"] = "not_evaluated"
    gate_h_status: Literal["not_evaluated"] = "not_evaluated"
    public_replay_ready: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def result_never_claims_case_completion(self):
        if self.sensor == "bluetooth":
            raise ValueError("Bluetooth cannot produce a public numeric run")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("result source IDs must be unique")
        return self


PublicExplorationResult = (
    PublicLightExploreResult | PublicPressureExploreResult | PublicSensorExploreResult
)


@dataclass(frozen=True)
class GeneralPublicComponentExecution:
    result: GeneralPublicComponentRunResult
    public_result: PublicExplorationResult


@dataclass(frozen=True)
class _ComponentSpec:
    component_id: str
    sensor: SensorKind
    protocol_id: str
    title: str
    dataset_ids: tuple[str, ...]
    supported_scope: str


_COMMON_MISSING_SCOPE = (
    "缺少当前用户的条件标签、现场控制、独立重复和同一时钟采集，因此不能绑定到当前通用实验，"
    "也不能形成多传感器联合结论。"
)

_COMPONENT_SPECS = (
    _ComponentSpec(
        "public-light-bounded-loop",
        "light",
        "light-public-exploration.v1",
        "Light 公开真实数据 Agent 组件",
        (
            "light-brighter-time-20220701-v1",
            "light-phyphox-snr-20260611-v1",
            "light-privacy-dual-20231127-v1",
        ),
        "可在自然光节律、单次手部遮挡轨迹和已注册触碰—遮挡对照中选择有边界证据。",
    ),
    _ComponentSpec(
        "public-pressure-height-loop",
        "pressure",
        "pressure-public-exploration.v1",
        "Pressure 公开相对高度 Agent 组件",
        ("pressure-nist-perfloc-pixel-20180516-v1",),
        "可分析 NIST Pixel XL 楼梯或电梯压力上行，并与隐藏稀疏高度锚点做服务端评估。",
    ),
    _ComponentSpec(
        "public-accelerometer-cadence-loop",
        "accelerometer",
        "walking-cadence-public-exploration.v1",
        "Accelerometer 公开步频 Agent 组件",
        ("accelerometer-nist-perfloc-pixel-20180516-v1",),
        "可比较三段 NIST 楼梯上行的主频、动态轴 RMS、频谱信噪比和重复性。",
    ),
    _ComponentSpec(
        "public-accelerometer-elevator-loop",
        "accelerometer",
        "elevator-motion-public-exploration.v1",
        "Accelerometer 公开电梯阶段 Agent 组件",
        ("accelerometer-nist-perfloc-pixel-20180516-v1",),
        "可检查 NIST 电梯上行的正加速、稳定与减速阶段序列。",
    ),
    _ComponentSpec(
        "public-accelerometer-vibration-loop",
        "accelerometer",
        "vibration-response-public-exploration.v1",
        "Accelerometer 公开运动响应 Agent 组件",
        ("accelerometer-nist-perfloc-pixel-20180516-v1",),
        "可比较静止锚点与手持运动记录，验证加速度测量链的响应分离。",
    ),
    _ComponentSpec(
        "public-gyroscope-motion-loop",
        "gyroscope",
        "gyroscope-public-exploration.v1",
        "Gyroscope 公开运动状态 Agent 组件",
        ("gyroscope-nist-perfloc-as7-pixel-20180516-v1",),
        "可比较公开静止锚点与手持转动的角速度模长响应。",
    ),
    _ComponentSpec(
        "public-magnetometer-field-loop",
        "magnetometer",
        "magnetometer-public-exploration.v1",
        "Magnetometer 公开磁场变化 Agent 组件",
        ("magnetometer-nist-perfloc-as7-pixel-20180516-v1",),
        "可比较公开稳定背景与局部磁场变化，但不能识别物体或绝对航向。",
    ),
    _ComponentSpec(
        "public-proximity-state-loop",
        "proximity",
        "proximity-public-exploration.v1",
        "Proximity 公开二态事件 Agent 组件",
        ("proximity-nist-perfloc-as7-pixel-20180516-v1",),
        "可检查稀疏 0/5 cm 状态码与前后事件切片的一致性。",
    ),
    _ComponentSpec(
        "public-microphone-level-loop",
        "microphone",
        "microphone-public-exploration.v1",
        "Microphone 公开派生级别 Agent 组件",
        ("microphone-noisecapture-andorra-odbl-v1",),
        "可比较不含原始音频的前后相对级别窗口，不能声称校准声压级或识别内容。",
    ),
    _ComponentSpec(
        "public-location-route-loop",
        "location",
        "location-public-exploration.v1",
        "Location 公开相对路线 Agent 组件",
        ("location-uci-gps-trajectories-20160228-v1",),
        "可比较两次隐私变换路线的长度、相对形状和终点一致性。",
    ),
)


def build_general_public_component_catalog(
    case: GeneralExperimentCase,
    *,
    root: Path,
) -> GeneralPublicComponentCatalog:
    """Resolve source-validated analogues without binding them as case evidence."""

    case = GeneralExperimentCase.model_validate(case.model_dump(mode="python"))
    catalog = list_public_replay_catalog(root)
    by_dataset = {item.dataset_id: item for item in catalog}
    requested_sensors = tuple(
        item.sensor for item in case.protocol.sensors if item.sensor != "bluetooth"
    )
    components: list[GeneralPublicComponentDescriptor] = []
    for spec in _COMPONENT_SPECS:
        if spec.sensor not in requested_sensors:
            continue
        for dataset_id in spec.dataset_ids:
            item = by_dataset.get(dataset_id)
            if item is None:
                raise GeneralPublicComponentValidation(
                    f"registered public component dataset is unavailable: {dataset_id}"
                )
            if item.sensor != spec.sensor or item.public_replay_status != "source_validated":
                raise GeneralPublicComponentValidation(
                    f"public component dataset identity mismatch: {dataset_id}"
                )
        components.append(
            GeneralPublicComponentDescriptor(
                component_id=spec.component_id,
                sensor=spec.sensor,
                protocol_id=spec.protocol_id,
                title=spec.title,
                dataset_ids=spec.dataset_ids,
                supported_scope=spec.supported_scope,
                missing_scope=_COMMON_MISSING_SCOPE,
            )
        )
    covered = {item.sensor for item in components}
    if covered != set(requested_sensors):
        missing = sorted(set(requested_sensors) - covered)
        raise GeneralPublicComponentValidation(
            f"no source-validated public component for sensors: {', '.join(missing)}"
        )
    return GeneralPublicComponentCatalog(
        case_id=case.case_id,
        case_revision=case.revision,
        protocol_id=case.protocol.protocol_id,
        requested_sensors=requested_sensors,
        components=tuple(components),
        boundary_messages=(
            "公开组件只回答其注册数据集中的有边界问题，不等于当前现场实验条件。",
            "每个传感器组件独立运行；当前没有公开的同一时钟多传感器联合数据集。",
            "运行结果单独进入公开回放历史，不增加当前案例 evidence、revision 或真机 Gate C。",
        ),
    )


PlannerCallable = Callable[[object], Awaitable[object]]


async def run_general_public_component(
    case: GeneralExperimentCase,
    request: GeneralPublicComponentRunRequest,
    *,
    root: Path,
    planner: PlannerCallable | None = None,
) -> GeneralPublicComponentExecution:
    """Run one registered public Agent loop while leaving the case immutable."""

    case = GeneralExperimentCase.model_validate(case.model_dump(mode="python"))
    request = GeneralPublicComponentRunRequest.model_validate(
        request.model_dump(mode="python")
    )
    if request.expected_revision != case.revision:
        raise GeneralPublicComponentValidation(
            f"stale general exploration revision: expected {case.revision}, received "
            f"{request.expected_revision}"
        )
    if not request.privacy_acknowledged:
        raise GeneralPublicComponentValidation(
            "运行公开组件前必须确认本地回放、隐私与非真机证据边界。"
        )
    catalog = build_general_public_component_catalog(case, root=root)
    component = next(
        (item for item in catalog.components if item.component_id == request.component_id),
        None,
    )
    if component is None:
        raise GeneralPublicComponentValidation(
            "component_id 不属于当前案例冻结传感器的服务端候选。"
        )

    question = case.protocol.question
    if component.sensor == "light":
        public_result: PublicExplorationResult = await run_public_light_exploration(
            PublicLightExploreRequest(
                research_question=question,
                privacy_acknowledged=True,
            ),
            root=root,
            planner=planner,  # type: ignore[arg-type]
        )
    elif component.sensor == "pressure":
        public_result = await run_public_pressure_exploration(
            PublicPressureExploreRequest(
                research_question=question,
                privacy_acknowledged=True,
            ),
            root=root,
            planner=planner,  # type: ignore[arg-type]
        )
    else:
        public_result = await run_public_sensor_exploration(
            PublicSensorExploreRequest(
                sensor=cast(PublicSensorAgentKind, component.sensor),
                protocol_id=component.protocol_id,
                research_question=question,
                privacy_acknowledged=True,
            ),
            root=root,
            planner=planner,  # type: ignore[arg-type]
        )

    report = public_result.report
    normalized = GeneralPublicComponentRunResult(
        case_id=case.case_id,
        case_revision=case.revision,
        component_id=component.component_id,
        sensor=component.sensor,
        protocol_id=public_result.protocol_id,
        run_id=public_result.run_id,
        execution_status=public_result.execution_status,
        conclusion_kind=report.conclusion_kind,
        title=report.title,
        summary=report.summary,
        selected_route_id=public_result.selected_route_id,
        planner_status=public_result.planner_status,
        planner_steps=tuple(
            GeneralPublicPlannerStep(
                step=item.step,
                source=item.source,
                outcome=item.outcome,
                selected_candidate_id=item.selected_candidate_id,
                rationale_code=item.rationale_code,
            )
            for item in public_result.planner_trace
        ),
        tool_ids=tuple(item.tool_id for item in public_result.tool_trace),
        evidence_count=len(public_result.evidence),
        source_ids=tuple(report.source_ids),
        findings=tuple(item.text for item in report.supported_findings),
        uncertainties=tuple(report.uncertainties),
        forbidden_claims=tuple(report.forbidden_claims),
        next_live_measurement=report.next_live_measurement,
    )
    return GeneralPublicComponentExecution(
        result=normalized,
        public_result=public_result,
    )
