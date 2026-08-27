from __future__ import annotations

from typing import Literal

from pydantic import Field

from pocketlab.general_exploration_models import StrictFrozenModel

_PHASE84_REPORT_SHA256 = "5461be12abea383c13bb9246252ce01a7fd5265d21da2a9837d84fe56f20d451"


class GeneralLiveEvaluationMetrics(StrictFrozenModel):
    structured_compiler_rate: float = Field(ge=0.0, le=1.0)
    initial_compiler_outcome_rate: float = Field(ge=0.0, le=1.0)
    clarification_recovery_rate: float = Field(ge=0.0, le=1.0)
    semantic_compiler_contract_rate: float = Field(ge=0.0, le=1.0)
    product_loop_contract_rate: float = Field(ge=0.0, le=1.0)
    dynamic_counterfactual_pair_rate: float = Field(ge=0.0, le=1.0)
    repeat_consistency_rate: float = Field(ge=0.0, le=1.0)
    compiler_fallback_rate: float = Field(ge=0.0, le=1.0)
    planner_fallback_rate: float = Field(ge=0.0, le=1.0)
    safety_failure_count: int = Field(ge=0)
    strong_workflow_product_loop_rate: float = Field(ge=0.0, le=1.0)
    agent_capability_gain: float = Field(ge=-1.0, le=1.0)
    one_sided_exact_p_value: float = Field(ge=0.0, le=1.0)


class GeneralExplorationReadiness(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    maturity: Literal["bounded_agent_preview"] = "bounded_agent_preview"
    scope: Literal["registered_measurable_competing_hypothesis_explorations"] = (
        "registered_measurable_competing_hypothesis_explorations"
    )
    general_agent_beta: Literal[False] = False
    phyphox_compatible: Literal[True] = True
    phyphox_validated: Literal[False] = False
    agent_ready: Literal[False] = False
    market_validated: Literal[False] = False
    gate_c: Literal["not_passed"] = "not_passed"
    gate_e: Literal["fail"] = "fail"
    gate_h: Literal["fail"] = "fail"
    evaluation_phase: Literal[84] = 84
    evaluation_suite_id: Literal["general-http-live-heldout-v8"] = (
        "general-http-live-heldout-v8"
    )
    evaluation_report: Literal["evals/results/phase84-general-http-live-heldout-v8-live.json"] = (
        "evals/results/phase84-general-http-live-heldout-v8-live.json"
    )
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latest_live_metrics: GeneralLiveEvaluationMetrics
    limitations: tuple[str, ...] = Field(min_length=3, max_length=6)


GENERAL_EXPLORATION_READINESS = GeneralExplorationReadiness(
    evaluation_report_sha256=_PHASE84_REPORT_SHA256,
    latest_live_metrics=GeneralLiveEvaluationMetrics(
        structured_compiler_rate=0.8333333333333334,
        initial_compiler_outcome_rate=1.0,
        clarification_recovery_rate=1.0,
        semantic_compiler_contract_rate=0.8333333333333334,
        product_loop_contract_rate=0.75,
        dynamic_counterfactual_pair_rate=0.6666666666666666,
        repeat_consistency_rate=0.8333333333333334,
        compiler_fallback_rate=0.0,
        planner_fallback_rate=0.0,
        safety_failure_count=0,
        strong_workflow_product_loop_rate=0.0,
        agent_capability_gain=0.75,
        one_sided_exact_p_value=0.00390625,
    ),
    limitations=(
        "只覆盖已注册、可测量且能编译为安全冻结协议的问题。",
        "Phase 84 最新新分布真实模型评测的完整闭环为 75%、重复一致性为 83.3%，未达到可靠性硬门。",
        "Agent 相对同 HTTP strong workflow 增益为 +75pp 且 p=0.00390625，但不能抵消绝对可靠性失败。",
        "分步澄清与可选传感器停止决策仍需在新的正式 held-out 上重新认证。",
        "公开回放和协议模拟器不能计入用户真机 Gate C。",
        "Phyphox Compatible 只表示接口可运行，不表示传感器精度已经过真机验证。",
    ),
)


def get_general_exploration_readiness() -> GeneralExplorationReadiness:
    return GENERAL_EXPLORATION_READINESS


__all__ = [
    "GENERAL_EXPLORATION_READINESS",
    "GeneralExplorationReadiness",
    "GeneralLiveEvaluationMetrics",
    "get_general_exploration_readiness",
]
