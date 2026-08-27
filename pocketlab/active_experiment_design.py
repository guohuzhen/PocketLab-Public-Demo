from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pocketlab.sensor_models import SensorKind

_IDENTIFIER = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
_METRIC_KEY = r"^[A-Za-z][A-Za-z0-9_]*$"


class _StrictFrozen(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class ActiveExperimentCondition(_StrictFrozen):
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    label: str = Field(min_length=1, max_length=100)
    factor_level: str = Field(min_length=1, max_length=120)
    instruction: str = Field(min_length=1, max_length=800)


class ActiveExperimentDesignSpec(_StrictFrozen):
    """Server-owned design envelope shared by single-sensor bounded experiments."""

    schema_version: Literal["1.0"] = "1.0"
    spec_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    spec_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    exploration_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    sensor: SensorKind
    analyzer_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    primary_metric_key: str = Field(pattern=_METRIC_KEY, max_length=80)
    primary_metric_unit: str = Field(min_length=1, max_length=24)
    independent_factor: str = Field(min_length=1, max_length=100)
    conditions: tuple[ActiveExperimentCondition, ...] = Field(min_length=2, max_length=4)
    required_repeats_per_condition: Literal[3] = 3
    max_corrections: int = Field(default=2, ge=1, le=4)
    max_measurements: int = Field(ge=8, le=16)
    controls: tuple[str, ...] = Field(min_length=2, max_length=12)
    safety_notes: tuple[str, ...] = Field(min_length=1, max_length=10)
    privacy_notes: tuple[str, ...] = Field(default=(), max_length=10)
    claim_boundaries: tuple[str, ...] = Field(min_length=2, max_length=10)
    candidate_actions: tuple[
        Literal["collect_condition", "replicate_condition", "correct_condition"],
        ...,
    ] = (
        "collect_condition",
        "replicate_condition",
        "correct_condition",
    )
    planner_permissions: Literal["select_candidate_id_only"] = "select_candidate_id_only"
    termination_owner: Literal["server"] = "server"
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False
    market_validated: Literal[False] = False

    @model_validator(mode="after")
    def design_contract_is_closed(self) -> Self:
        condition_ids = [item.condition_id for item in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("active design condition IDs must be unique")
        if len(self.controls) != len(set(self.controls)):
            raise ValueError("active design controls must be unique")
        minimum_measurements = len(self.conditions) * self.required_repeats_per_condition
        if self.max_measurements < minimum_measurements + self.max_corrections:
            raise ValueError("measurement budget must include all repeats and corrections")
        expected_actions = {
            "collect_condition",
            "replicate_condition",
            "correct_condition",
        }
        if set(self.candidate_actions) != expected_actions:
            raise ValueError("active design action allowlist changed")
        return self


class ActiveDesignEvidenceDigest(_StrictFrozen):
    evidence_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    confidence: Literal["low", "medium", "high"]
    metric_key: str = Field(pattern=_METRIC_KEY, max_length=80)
    metric_value: float
    metric_unit: str = Field(min_length=1, max_length=24)


class ActiveDesignCandidate(_StrictFrozen):
    candidate_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    action: Literal["collect_condition", "replicate_condition", "correct_condition"]
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    title: str = Field(min_length=1, max_length=160)
    instruction: str = Field(min_length=1, max_length=800)
    reason_code: Literal[
        "missing_condition",
        "replication_required",
        "quality_correction",
    ]
    input_evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)


class ActiveDesignProposal(_StrictFrozen):
    schema_version: Literal["1.0"] = "1.0"
    spec_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    evidence_count: int = Field(ge=0, le=32)
    valid_evidence_count: int = Field(ge=0, le=32)
    correction_count: int = Field(ge=0, le=8)
    condition_counts: dict[str, int]
    candidates: tuple[ActiveDesignCandidate, ...] = Field(default=(), max_length=4)
    fallback_candidate_id: str | None = Field(default=None, pattern=_IDENTIFIER)
    planner_eligible: bool
    conclusion_ready: bool
    forced_stop: bool
    stop_reason_code: Literal[
        "continue",
        "evidence-complete",
        "measurement-budget-exhausted",
        "correction-budget-exhausted",
    ]

    @model_validator(mode="after")
    def proposal_state_is_consistent(self) -> Self:
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("active design candidate IDs must be unique")
        if self.candidates:
            if self.fallback_candidate_id not in candidate_ids:
                raise ValueError("fallback must reference a candidate")
            if self.conclusion_ready or self.forced_stop:
                raise ValueError("terminal proposals cannot retain candidates")
        elif self.fallback_candidate_id is not None:
            raise ValueError("empty candidate sets cannot define fallback")
        if self.planner_eligible != (len(self.candidates) >= 2):
            raise ValueError("Planner is eligible only when multiple candidates exist")
        if self.conclusion_ready != (self.stop_reason_code == "evidence-complete"):
            raise ValueError("conclusion_ready must match evidence-complete")
        if self.forced_stop != self.stop_reason_code.endswith("budget-exhausted"):
            raise ValueError("forced_stop must match a budget stop")
        return self


def _condition(
    condition_id: str,
    label: str,
    factor_level: str,
    instruction: str,
) -> ActiveExperimentCondition:
    return ActiveExperimentCondition(
        condition_id=condition_id,
        label=label,
        factor_level=factor_level,
        instruction=instruction,
    )


_SPECS = (
    ActiveExperimentDesignSpec(
        spec_id="rotation-rate-check.active.v1",
        spec_version="1.0.0",
        exploration_id="rotation-rate-check",
        sensor="gyroscope",
        analyzer_id="pocketlab.gyroscope.v1",
        primary_metric_key="angular_speed_std_rad_s",
        primary_metric_unit="rad/s",
        independent_factor="转台设定状态",
        conditions=(
            _condition("stationary", "静止基线", "静止", "固定手机并记录静止零偏基线。"),
            _condition("steady-rotation", "稳定低速旋转", "低速匀速", "可靠固定手机后记录同一低速设定的稳定旋转段。"),
        ),
        max_measurements=8,
        controls=("保持同一手机和固定位置。", "保持同一采集时长和手机朝向。"),
        safety_notes=("不得手持高速旋转；必须可靠固定并限制转速。",),
        claim_boundaries=("不积分为角位移。", "只比较本次设备、固定方式与测试转速。"),
    ),
    ActiveExperimentDesignSpec(
        spec_id="magnetic-field-walk.active.v1",
        spec_version="1.0.0",
        exploration_id="magnetic-field-walk",
        sensor="magnetometer",
        analyzer_id="pocketlab.magnetometer.v1",
        primary_metric_key="max_field_deviation_ut",
        primary_metric_unit="uT",
        independent_factor="与目标物的相对位置",
        conditions=(
            _condition("background", "远离目标背景", "背景位置", "保持手机方向，记录远离目标物的背景磁场。"),
            _condition("target-near", "目标附近", "固定近距离", "保持手机方向与高度，只改变到目标物的距离并记录。"),
        ),
        max_measurements=8,
        controls=("保持手机朝向与高度。", "一次只改变与目标物的距离。"),
        safety_notes=("不靠近高压设备、强磁体或医疗植入物限制区域。",),
        claim_boundaries=("异常只表示局部磁场变化候选。", "不凭磁场序列识别具体物体或证明因果。"),
    ),
    ActiveExperimentDesignSpec(
        spec_id="pressure-elevator-altitude.active.v1",
        spec_version="1.0.0",
        exploration_id="pressure-elevator-altitude",
        sensor="pressure",
        analyzer_id="pocketlab.pressure.v2",
        primary_metric_key="relative_height_change_m",
        primary_metric_unit="m",
        independent_factor="行程方向",
        conditions=(
            _condition("outbound", "去程", "起点到目标楼层", "从起点静止平台开始，记录到目标楼层再次稳定。"),
            _condition("return", "回程", "目标楼层回到起点", "保持同一手机与路线，记录返回起点并再次稳定。"),
        ),
        max_measurements=8,
        controls=("每次都包含起止稳定平台。", "保持同一路线、楼层区间和手机放置。"),
        safety_notes=("遵守场所规则，不妨碍电梯运行或进入受限区域。",),
        claim_boundaries=("只报告标准大气近似相对高度。", "天气、HVAC 与漂移未排除时不归因为垂直运动。"),
    ),
    ActiveExperimentDesignSpec(
        spec_id="room-acoustic-response.active.v1",
        spec_version="1.0.0",
        exploration_id="room-acoustic-response",
        sensor="microphone",
        analyzer_id="pocketlab.microphone.derived.v1",
        primary_metric_key="mean_relative_level_db",
        primary_metric_unit="dB_relative",
        independent_factor="房间测点",
        conditions=(
            _condition("reference-position", "参考测点", "固定参考位置", "固定声源后，在参考位置记录派生相对级别。"),
            _condition("comparison-position", "对比测点", "第二位置", "只改变手机测点，保持声源、方向和片段不变。"),
        ),
        max_measurements=8,
        controls=("保持声源、音量和测试片段。", "保持手机方向与记录时长。"),
        safety_notes=("使用舒适音量并缩短暴露。",),
        privacy_notes=("只接受派生级别，不保存原始音频；采集前需确认隐私。",),
        claim_boundaries=("dB_relative 不是校准 SPL。", "自动增益未知时只作同设备相对比较。"),
    ),
    ActiveExperimentDesignSpec(
        spec_id="gps-route-consistency.active.v1",
        spec_version="1.0.0",
        exploration_id="gps-route-consistency",
        sensor="location",
        analyzer_id="pocketlab.location.haversine.v1",
        primary_metric_key="path_efficiency_ratio",
        primary_metric_unit="1",
        independent_factor="路线环境",
        conditions=(
            _condition("open-sky", "开阔路线", "开阔环境", "在安全公开路线完成一次开阔环境记录。"),
            _condition("obstructed", "遮挡路线", "建筑遮挡环境", "在同等长度的安全路线记录建筑遮挡条件。"),
        ),
        max_measurements=8,
        controls=("保持相近路线长度与移动方式。", "保持同一手机和定位实验设置。"),
        safety_notes=("不要边走边操作手机，不进入车道或危险区域。",),
        privacy_notes=("位置轨迹默认只在本地保存，禁止在报告中显示绝对坐标。",),
        claim_boundaries=("轨迹指标不是测绘级距离。", "不输出或推断住址与绝对位置。"),
    ),
    ActiveExperimentDesignSpec(
        spec_id="proximity-response.active.v1",
        spec_version="1.0.0",
        exploration_id="proximity-response",
        sensor="proximity",
        analyzer_id="pocketlab.proximity.v2",
        primary_metric_key="transition_count",
        primary_metric_unit="count",
        independent_factor="目标接近方式",
        conditions=(
            _condition("normal-approach", "垂直接近", "正对传感器", "用安全平面目标正对传感器缓慢接近并远离。"),
            _condition("angled-approach", "倾斜接近", "固定倾角", "只改变目标角度，以相同速度接近并远离。"),
        ),
        max_measurements=8,
        controls=("保持同一目标材质与移动速度。", "保持同一手机、起止距离和采集时长。"),
        safety_notes=("不要用尖锐物接触屏幕或传感器开孔。",),
        claim_boundaries=("二态输出只解释为 near/far。", "连续数值也不等同于精密物理距离。"),
    ),
)

_SPEC_BY_ID = {item.spec_id: item for item in _SPECS}
_SPEC_BY_EXPLORATION = {item.exploration_id: item for item in _SPECS}
if len(_SPEC_BY_ID) != len(_SPECS) or len(_SPEC_BY_EXPLORATION) != len(_SPECS):
    raise RuntimeError("active experiment design specs must be unique")


def list_active_experiment_design_specs() -> list[ActiveExperimentDesignSpec]:
    return [item.model_copy(deep=True) for item in _SPECS]


def get_active_experiment_design_spec(spec_id: str) -> ActiveExperimentDesignSpec:
    try:
        return _SPEC_BY_ID[spec_id].model_copy(deep=True)
    except KeyError as exc:
        raise KeyError(f"Unknown active experiment design spec: {spec_id}") from exc


def active_design_spec_for_exploration(
    exploration_id: str,
) -> ActiveExperimentDesignSpec | None:
    item = _SPEC_BY_EXPLORATION.get(exploration_id)
    return item.model_copy(deep=True) if item is not None else None


def propose_active_design_candidates(
    spec: ActiveExperimentDesignSpec,
    evidence: list[ActiveDesignEvidenceDigest],
    *,
    correction_count: int = 0,
) -> ActiveDesignProposal:
    """Generate safe candidates; this function neither calls a model nor mutates state."""

    spec = ActiveExperimentDesignSpec.model_validate(spec.model_dump(mode="python"))
    evidence = [
        ActiveDesignEvidenceDigest.model_validate(item.model_dump(mode="python"))
        for item in evidence
    ]
    if correction_count < 0:
        raise ValueError("correction_count cannot be negative")
    if len(evidence) > spec.max_measurements:
        raise ValueError("evidence exceeds the protocol measurement budget")
    condition_ids = {item.condition_id for item in spec.conditions}
    for item in evidence:
        if item.condition_id not in condition_ids:
            raise ValueError("evidence references a condition outside the spec")
        if item.metric_key != spec.primary_metric_key or item.metric_unit != spec.primary_metric_unit:
            raise ValueError("evidence primary metric does not match the spec")

    counts = {
        condition.condition_id: sum(
            item.condition_id == condition.condition_id and item.confidence != "low"
            for item in evidence
        )
        for condition in spec.conditions
    }
    valid_count = sum(counts.values())
    if len(evidence) >= spec.max_measurements:
        return ActiveDesignProposal(
            spec_id=spec.spec_id,
            evidence_count=len(evidence),
            valid_evidence_count=valid_count,
            correction_count=correction_count,
            condition_counts=counts,
            candidates=(),
            fallback_candidate_id=None,
            planner_eligible=False,
            conclusion_ready=False,
            forced_stop=True,
            stop_reason_code="measurement-budget-exhausted",
        )

    if evidence and evidence[-1].confidence == "low":
        if correction_count >= spec.max_corrections:
            return ActiveDesignProposal(
                spec_id=spec.spec_id,
                evidence_count=len(evidence),
                valid_evidence_count=valid_count,
                correction_count=correction_count,
                condition_counts=counts,
                candidates=(),
                fallback_candidate_id=None,
                planner_eligible=False,
                conclusion_ready=False,
                forced_stop=True,
                stop_reason_code="correction-budget-exhausted",
            )
        condition = next(
            item for item in spec.conditions if item.condition_id == evidence[-1].condition_id
        )
        candidate = ActiveDesignCandidate(
            candidate_id=f"correct-{condition.condition_id}",
            action="correct_condition",
            condition_id=condition.condition_id,
            title=f"纠偏复测 · {condition.label}",
            instruction=condition.instruction,
            reason_code="quality_correction",
            input_evidence_ids=(evidence[-1].evidence_id,),
        )
        return ActiveDesignProposal(
            spec_id=spec.spec_id,
            evidence_count=len(evidence),
            valid_evidence_count=valid_count,
            correction_count=correction_count,
            condition_counts=counts,
            candidates=(candidate,),
            fallback_candidate_id=candidate.candidate_id,
            planner_eligible=False,
            conclusion_ready=False,
            forced_stop=False,
            stop_reason_code="continue",
        )

    incomplete = [
        item
        for item in spec.conditions
        if counts[item.condition_id] < spec.required_repeats_per_condition
    ]
    if not incomplete:
        return ActiveDesignProposal(
            spec_id=spec.spec_id,
            evidence_count=len(evidence),
            valid_evidence_count=valid_count,
            correction_count=correction_count,
            condition_counts=counts,
            candidates=(),
            fallback_candidate_id=None,
            planner_eligible=False,
            conclusion_ready=True,
            forced_stop=False,
            stop_reason_code="evidence-complete",
        )

    minimum_count = min(counts[item.condition_id] for item in incomplete)
    candidate_conditions = [
        item for item in incomplete if counts[item.condition_id] == minimum_count
    ][:3]
    input_ids = tuple(item.evidence_id for item in evidence[-8:] if item.confidence != "low")
    candidates = tuple(
        ActiveDesignCandidate(
            candidate_id=(
                f"collect-{condition.condition_id}"
                if counts[condition.condition_id] == 0
                else f"replicate-{condition.condition_id}-{counts[condition.condition_id] + 1}"
            ),
            action=(
                "collect_condition"
                if counts[condition.condition_id] == 0
                else "replicate_condition"
            ),
            condition_id=condition.condition_id,
            title=(
                f"采集 · {condition.label}"
                if counts[condition.condition_id] == 0
                else f"重复 {counts[condition.condition_id] + 1}/3 · {condition.label}"
            ),
            instruction=condition.instruction,
            reason_code=(
                "missing_condition"
                if counts[condition.condition_id] == 0
                else "replication_required"
            ),
            input_evidence_ids=input_ids,
        )
        for condition in candidate_conditions
    )
    return ActiveDesignProposal(
        spec_id=spec.spec_id,
        evidence_count=len(evidence),
        valid_evidence_count=valid_count,
        correction_count=correction_count,
        condition_counts=counts,
        candidates=candidates,
        fallback_candidate_id=candidates[0].candidate_id,
        planner_eligible=len(candidates) >= 2,
        conclusion_ready=False,
        forced_stop=False,
        stop_reason_code="continue",
    )
