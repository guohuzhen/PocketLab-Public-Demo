from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pocketlab.agent_runtime import AgentRuntimeError
from pocketlab.model_run_control import await_model_validation_recovery_decision
from pocketlab.public_light_models import (
    PublicLightEvidenceSnapshot,
    PublicLightEvidenceView,
    PublicLightExploreRequest,
    PublicLightExploreResult,
    PublicLightFact,
    PublicLightFinding,
    PublicLightPlanCandidate,
    PublicLightPlannerDecision,
    PublicLightPlannerRequest,
    PublicLightPlannerTrace,
    PublicLightReport,
    PublicLightReportSource,
    PublicLightRuntimeTrace,
    PublicLightToolExecution,
)
from pocketlab.public_light_tools import (
    BRIGHTER_TIME_DATASET_ID,
    PRIVACY_DUAL_DATASET_ID,
    LightClaimAuditResult,
    NaturalisticLightContextResult,
    PHYPhOX_SNR_DATASET_ID,
    PHYPhOX_SNR_RECORDING_ID,
    PublicLightConditionComparison,
    PublicLightProvenance,
    PublicLightTraceResult,
    audit_light_claim_support,
    compare_registered_light_conditions,
    inspect_public_light_trace,
    summarize_naturalistic_light_context,
)
from pocketlab.public_replay_dataset import get_public_replay_dataset

PlannerCallable = Callable[[PublicLightPlannerRequest], Awaitable[object]]

_COMPARE_ID = "compare_registered_conditions"
_PHONE_TRACE_ID = "inspect_phone_perturbation"
_NATURALISTIC_ID = "summarize_naturalistic_context"
_FINISH_ID = "finish_descriptive"
_LIVE_ID = "request_live_measurement"
_UNSUPPORTED_ID = "stop_unsupported"
_PRIVACY_ACK_ID = "privacy_acknowledgement_required"

_COMPARE_TOOL = "compare_registered_light_conditions"
_TRACE_TOOL = "inspect_public_light_trace"
_CONTEXT_TOOL = "summarize_naturalistic_light_context"
_AUDIT_TOOL = "audit_light_claim_support"

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


class PublicLightPlannerOutcome(Protocol):
    decision: PublicLightPlannerDecision
    runtime_trace: dict[str, Any]


class PublicLightExplorationUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"公开 Light 探索无法安全完成（{reason}）。")
        self.reason = reason


class _PlannerDecisionRejected(ValueError):
    reason = "invalid-agent-decision"


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _question_family(request: PublicLightExploreRequest) -> str:
    question = unicodedata.normalize("NFKC", request.research_question).casefold()
    if _contains_any(
        question,
        (
            "忽略规则",
            "忽略限制",
            "读取密钥",
            "api key",
            "system prompt",
            "调用 shell",
            "删除数据库",
            "ignore previous",
            "ignore all rules",
            "reveal the key",
        ),
    ):
        return "unsupported_injection"
    if _contains_any(
        question,
        (
            "平方反比",
            "反平方",
            "inverse square",
            "distance law",
            "距离定律",
        ),
    ):
        return "unsupported_distance_law"
    if _contains_any(
        question,
        (
            "识别行为",
            "识别动作",
            "识别人",
            "还原图像",
            "推断隐私",
            "behavior identification",
            "identify the person",
            "reconstruct image",
        ),
    ):
        return "unsupported_behavior_identification"
    if _contains_any(
        question,
        (
            "绝对校准",
            "校准成照度计",
            "绝对准确",
            "absolute calibration",
            "calibrated lux meter",
        ),
    ):
        return "unsupported_absolute_calibration"
    if _contains_any(
        question,
        (
            "因果",
            "导致了",
            "造成了",
            "causal",
            "caused by",
        ),
    ):
        return "unsupported_causal_effect"
    if _contains_any(
        question,
        (
            "是否健康",
            "有益健康",
            "健康阈值",
            "照明合格",
            "health threshold",
            "healthy light",
            "safe illuminance",
        ),
    ):
        return "unsupported_health_threshold"
    if request.query_illuminance_lx is not None or _contains_any(
        question,
        (
            "自然环境",
            "日常环境",
            "常见照度",
            "典型照度",
            "大概处于",
            "百分位",
            "naturalistic",
            "everyday light",
            "typical illuminance",
            "percentile",
        ),
    ):
        return "naturalistic_context"
    if _contains_any(
        question,
        (
            "遮挡",
            "触摸",
            "手势",
            "遮住",
            "occlusion",
            "occluder",
            "touch",
        ),
    ):
        return "registered_condition_comparison"
    return "phone_perturbation"


def _claim_kind_for_family(family: str) -> str:
    return {
        "unsupported_injection": "behavior_identification",
        "unsupported_distance_law": "distance_law",
        "unsupported_behavior_identification": "behavior_identification",
        "unsupported_absolute_calibration": "absolute_calibration",
        "unsupported_causal_effect": "causal_effect",
        "unsupported_health_threshold": "naturalistic_context",
        "naturalistic_context": "naturalistic_context",
        "registered_condition_comparison": "descriptive_difference",
        "phone_perturbation": "temporal_pattern",
    }[family]


def _effective_family(original_family: str, selected_candidate_id: str) -> str:
    if original_family.startswith("unsupported_"):
        return original_family
    return {
        _COMPARE_ID: "registered_condition_comparison",
        _PHONE_TRACE_ID: "phone_perturbation",
        _NATURALISTIC_ID: "naturalistic_context",
    }.get(selected_candidate_id, original_family)


def _candidate(
    candidate_id: str,
    title: str,
    reason: str,
    *,
    tool_ids: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    privacy: bool = False,
    terminal: bool = False,
    result_code: str,
) -> PublicLightPlanCandidate:
    return PublicLightPlanCandidate(
        candidate_id=candidate_id,
        title=title,
        server_reason=reason,
        tool_ids=list(tool_ids),
        evidence_refs=list(evidence_refs),
        requires_privacy_acknowledgement=privacy,
        terminal=terminal,
        result_code=result_code,
    )


def _compare_candidate() -> PublicLightPlanCandidate:
    return _candidate(
        _COMPARE_ID,
        "比较两个已注册的遮挡条件",
        "对同一公开场景的两个独立采集做描述性比较，不进行行为识别或因果推断。",
        tool_ids=(_COMPARE_TOOL,),
        evidence_refs=(
            f"{PRIVACY_DUAL_DATASET_ID}/privacy-dual-occluder",
            f"{PRIVACY_DUAL_DATASET_ID}/privacy-dual-touch",
        ),
        privacy=True,
        result_code="registered_condition_comparison",
    )


def _phone_trace_candidate() -> PublicLightPlanCandidate:
    return _candidate(
        _PHONE_TRACE_ID,
        "检查公开 phyphox 手机照度序列",
        "检查一条真实 Android 手机的 phyphox Light 派生序列，只解释其声明仍有效的数值。",
        tool_ids=(_TRACE_TOOL,),
        evidence_refs=(f"{PHYPhOX_SNR_DATASET_ID}/{PHYPhOX_SNR_RECORDING_ID}",),
        result_code="phone_perturbation_trace",
    )


def _naturalistic_candidate() -> PublicLightPlanCandidate:
    return _candidate(
        _NATURALISTIC_ID,
        "查询自然环境照度上下文",
        "使用参与者级照度摘要提供描述性分布上下文，不形成健康或校准阈值。",
        tool_ids=(_CONTEXT_TOOL,),
        evidence_refs=(BRIGHTER_TIME_DATASET_ID,),
        privacy=True,
        result_code="naturalistic_context",
    )


def _terminal_candidate(candidate_id: str) -> PublicLightPlanCandidate:
    values = {
        _FINISH_ID: (
            "结束于当前描述性证据",
            "当前公开证据已经达到允许的描述性结论边界，继续增加来源不会自动提高可信度。",
            "minimal_sufficient_evidence",
        ),
        _LIVE_ID: (
            "请求下一次真实手机测量",
            "公开数据无法代表用户设备或现场控制，下一步应明确转为真实 phyphox 测量。",
            "live_measurement_required",
        ),
        _UNSUPPORTED_ID: (
            "停止不受支持的结论",
            "公开回放缺少该结论所需的控制、重复或校准证据，必须在越界前停止。",
            "unsupported_claim_boundary",
        ),
        _PRIVACY_ACK_ID: (
            "等待隐私确认",
            "候选公开序列保留行为光照特征，只能在本地且经用户明确确认后读取。",
            "privacy_acknowledgement_required",
        ),
    }
    title, reason, result_code = values[candidate_id]
    return _candidate(
        candidate_id,
        title,
        reason,
        terminal=True,
        result_code=result_code,
    )


def _initial_candidates(
    request: PublicLightExploreRequest,
    family: str,
) -> tuple[list[PublicLightPlanCandidate], str]:
    if family.startswith("unsupported_"):
        return (
            [
                _terminal_candidate(_UNSUPPORTED_ID),
                _terminal_candidate(_LIVE_ID),
            ],
            _UNSUPPORTED_ID,
        )
    if (
        family in {"registered_condition_comparison", "naturalistic_context"}
        and not request.privacy_acknowledged
    ):
        return (
            [
                _terminal_candidate(_PRIVACY_ACK_ID),
                _terminal_candidate(_UNSUPPORTED_ID),
            ],
            _PRIVACY_ACK_ID,
        )
    candidates = [_phone_trace_candidate(), _terminal_candidate(_UNSUPPORTED_ID)]
    if request.privacy_acknowledged:
        candidates.insert(0, _compare_candidate())
        candidates.insert(2, _naturalistic_candidate())
    fallback = {
        "registered_condition_comparison": _COMPARE_ID,
        "naturalistic_context": _NATURALISTIC_ID,
        "phone_perturbation": _PHONE_TRACE_ID,
    }[family]
    return candidates, fallback


def _question_mentions_phone_transfer(question: str) -> bool:
    return _contains_any(
        question.casefold(),
        (
            "手机",
            "phyphox",
            "phone",
            "android",
            "迁移",
            "交叉检查",
            "cross-check",
            "crosscheck",
        ),
    )


def _question_requires_live_phone_followup(question: str) -> bool:
    return _contains_any(
        unicodedata.normalize("NFKC", question).casefold(),
        (
            "当前手机",
            "我的手机",
            "我的环境",
            "采样频率",
            "变化频率",
            "原始时间轴",
            "原始数据",
            "current phone",
            "my phone",
            "my room",
            "sampling rate",
            "frequency",
            "raw cadence",
        ),
    )


def _follow_up_candidates(
    request: PublicLightExploreRequest,
    family: str,
    selected_ids: Sequence[str],
) -> tuple[list[PublicLightPlanCandidate], str]:
    selected = set(selected_ids)
    candidates: list[PublicLightPlanCandidate] = []
    if _COMPARE_ID in selected and _PHONE_TRACE_ID not in selected:
        candidates.append(_phone_trace_candidate())
    elif (
        _PHONE_TRACE_ID in selected
        and _COMPARE_ID not in selected
        and family == "registered_condition_comparison"
        and request.privacy_acknowledged
    ):
        candidates.append(_compare_candidate())
    elif (
        _NATURALISTIC_ID in selected
        and _PHONE_TRACE_ID not in selected
        and _question_mentions_phone_transfer(request.research_question)
    ):
        candidates.append(_phone_trace_candidate())
    candidates.extend([_terminal_candidate(_FINISH_ID), _terminal_candidate(_LIVE_ID)])
    if candidates[0].candidate_id in {_PHONE_TRACE_ID, _COMPARE_ID}:
        fallback = (
            candidates[0].candidate_id
            if _question_mentions_phone_transfer(request.research_question)
            else _FINISH_ID
        )
    else:
        fallback = (
            _LIVE_ID
            if _PHONE_TRACE_ID in selected
            and _question_requires_live_phone_followup(request.research_question)
            else _FINISH_ID
        )
    return candidates, fallback


def _evidence_view(
    evidence: Sequence[PublicLightEvidenceSnapshot],
) -> PublicLightEvidenceView:
    limitations: list[str] = []
    facts: list[PublicLightFact] = []
    for item in evidence:
        facts.extend(item.facts)
        limitations.extend(item.claim_boundary)
        limitations.extend(item.processing_disclosures)
    return PublicLightEvidenceView(
        evidence_ids=[item.evidence_id for item in evidence],
        result_codes=[item.evidence_id.removeprefix("evidence-") for item in evidence],
        facts=facts[:24],
        limitations=list(dict.fromkeys(limitations))[:16],
    )


def _planner_request(
    *,
    run_id: str,
    step: int,
    operation: str,
    request: PublicLightExploreRequest,
    evidence: Sequence[PublicLightEvidenceSnapshot],
    candidates: list[PublicLightPlanCandidate],
    fallback_candidate_id: str,
) -> PublicLightPlannerRequest:
    unsigned = {
        "schema_version": "1.0",
        "operation": operation,
        "run_id": run_id,
        "step": step,
        "research_question_untrusted": request.research_question,
        "privacy_acknowledged": request.privacy_acknowledged,
        "evidence_view": _evidence_view(evidence).model_dump(mode="json"),
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "fallback_candidate_id": fallback_candidate_id,
    }
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PublicLightPlannerRequest(
        **unsigned,
        request_sha256=digest,
    )


def _safe_runtime_trace(value: object) -> PublicLightRuntimeTrace | None:
    if not isinstance(value, dict):
        return None
    safe: dict[str, object] = {
        key: item
        for key, item in value.items()
        if key in _SAFE_RUNTIME_KEYS and (item is None or isinstance(item, (str, int, float, bool)))
    }
    tool_events = value.get("tool_events")
    if isinstance(tool_events, list):
        safe_events = []
        for item in tool_events[:4]:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            status = item.get("status")
            if isinstance(name, str) and isinstance(status, str):
                safe_events.append({"name": name[:100], "status": status[:40]})
        safe["tool_events"] = safe_events
    return PublicLightRuntimeTrace.model_validate(safe) if safe else None


def _allowed_rationales(candidate_id: str) -> frozenset[str]:
    return {
        _COMPARE_ID: frozenset({"match_registered_condition_comparison"}),
        _PHONE_TRACE_ID: frozenset(
            {"match_temporal_perturbation_goal", "add_phone_transfer_crosscheck"}
        ),
        _NATURALISTIC_ID: frozenset({"match_naturalistic_context_goal"}),
        _FINISH_ID: frozenset({"minimal_sufficient_evidence"}),
        _LIVE_ID: frozenset({"request_missing_live_evidence"}),
        _UNSUPPORTED_ID: frozenset({"unsupported_claim_boundary"}),
        _PRIVACY_ACK_ID: frozenset({"privacy_not_acknowledged"}),
    }[candidate_id]


def _fallback_reason(error: Exception) -> str:
    value = getattr(error, "reason", None)
    if not isinstance(value, str) or not value:
        value = "planner-unavailable"
    value = re.sub(r"[^a-z0-9-]+", "-", value.casefold()).strip("-")
    return (value or "planner-unavailable")[:100]


async def _select_candidate(
    planner_request: PublicLightPlannerRequest,
    *,
    planner: PlannerCallable | None,
) -> tuple[str, PublicLightPlannerTrace]:
    candidate_ids = [item.candidate_id for item in planner_request.candidates]
    fallback_id = planner_request.fallback_candidate_id
    active_planner = planner
    try:
        if active_planner is None:
            from pocketlab.public_light_planner import run_public_light_planner

            active_planner = run_public_light_planner
        outcome = await active_planner(planner_request)
        decision = getattr(outcome, "decision", None)
        runtime_trace = getattr(outcome, "runtime_trace", None)
        if not isinstance(decision, PublicLightPlannerDecision):
            raise _PlannerDecisionRejected
        if (
            decision.run_id != planner_request.run_id
            or decision.step != planner_request.step
            or decision.request_sha256 != planner_request.request_sha256
            or decision.selected_candidate_id not in candidate_ids
            or decision.rationale_code not in _allowed_rationales(decision.selected_candidate_id)
        ):
            raise _PlannerDecisionRejected
        return decision.selected_candidate_id, PublicLightPlannerTrace(
            step=planner_request.step,
            operation=planner_request.operation,
            request_sha256=planner_request.request_sha256,
            candidate_ids=candidate_ids,
            selected_candidate_id=decision.selected_candidate_id,
            fallback_candidate_id=fallback_id,
            rationale_code=decision.rationale_code,
            source="agent",
            outcome="accepted",
            transport=(
                str(runtime_trace.get("transport"))
                if isinstance(runtime_trace, dict)
                and runtime_trace.get("transport") in {"function_tool", "validated_json_text"}
                else "not_attempted"
            ),
            runtime_trace=_safe_runtime_trace(runtime_trace),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        reason = _fallback_reason(exc)
        if planner is None and reason not in {
            "user-fallback",
            "user-requested-fallback",
        }:
            recovery = await await_model_validation_recovery_decision(
                detail=(
                    "公开照度实验的基模规划未通过候选约束。请选择重试基模、"
                    "切换 Fast，或明确使用冻结的强工作流步骤。"
                ),
                error_kind=reason,
            )
            if recovery in {"retry", "retry_fast"}:
                return await _select_candidate(planner_request, planner=None)
            if recovery != "user_fallback":
                raise AgentRuntimeError(
                    "malformed_model_output",
                    "公开照度规划未完成；PocketLab 未替用户自动启用回退。",
                    retryable=True,
                ) from exc
            reason = "user-requested-fallback"
        elif planner is None:
            reason = "user-requested-fallback"
        return fallback_id, PublicLightPlannerTrace(
            step=planner_request.step,
            operation=planner_request.operation,
            request_sha256=planner_request.request_sha256,
            candidate_ids=candidate_ids,
            selected_candidate_id=fallback_id,
            fallback_candidate_id=fallback_id,
            rationale_code="strong_workflow_fallback",
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason=reason,
            runtime_trace=_safe_runtime_trace(getattr(exc, "runtime_trace", None)),
        )


def _unique_text(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _fact(key: str, value: float, unit: str) -> PublicLightFact:
    return PublicLightFact(key=key, value=float(value), unit=unit)


def _combined_provenance(
    provenances: Sequence[PublicLightProvenance],
) -> tuple[
    str,
    str,
    str | None,
    str,
    str,
    list[str],
    list[str],
]:
    first = provenances[0]
    if any(item.dataset_id != first.dataset_id for item in provenances):
        raise PublicLightExplorationUnavailable("cross-source-provenance")
    for item in provenances:
        if item.requires_user_acknowledgement and "local_replay" not in item.allowed_operations:
            raise PublicLightExplorationUnavailable("privacy-operation-not-allowed")
    device_scope = f"{first.device_class} · {first.acquisition_app}"
    disclosures = _unique_text(
        value for item in provenances for value in item.processing_disclosures
    )
    boundaries = _unique_text(value for item in provenances for value in item.claim_boundary)
    return (
        first.source_title,
        first.source_url,
        first.doi,
        first.license_spdx,
        device_scope,
        disclosures,
        boundaries,
    )


def _comparison_snapshot(
    result: PublicLightConditionComparison,
) -> PublicLightEvidenceSnapshot:
    source_title, source_url, doi, license_spdx, device_scope, disclosures, boundaries = (
        _combined_provenance(result.provenance)
    )
    facts = [
        _fact("privacy_median_difference", result.median_difference_lx, "lx"),
        _fact("privacy_iqr_difference", result.iqr_difference_lx, "lx"),
        _fact(
            "privacy_coefficient_of_variation_difference",
            result.coefficient_of_variation_difference_ratio,
            "ratio",
        ),
        _fact(
            "privacy_zero_fraction_difference",
            result.zero_fraction_difference,
            "ratio",
        ),
        _fact(
            "privacy_repeats_per_condition",
            result.repeats_per_condition,
            "count",
        ),
    ]
    if result.median_ratio is not None:
        facts.append(_fact("privacy_median_ratio", result.median_ratio, "ratio"))
    return PublicLightEvidenceSnapshot(
        evidence_id="evidence-privacy-comparison",
        dataset_id=result.dataset_id,
        recording_ids=[result.left.recording_id, result.right.recording_id],
        data_class=result.provenance[0].data_class,
        device_scope=device_scope,
        source_title=source_title,
        source_url=source_url,
        doi=doi,
        license_spdx=license_spdx,
        facts=facts,
        processing_disclosures=disclosures,
        claim_boundary=_unique_text([*boundaries, *result.limitations]),
    )


def _trace_snapshot(result: PublicLightTraceResult) -> PublicLightEvidenceSnapshot:
    provenance = result.provenance
    (
        source_title,
        source_url,
        doi,
        license_spdx,
        device_scope,
        disclosures,
        boundaries,
    ) = _combined_provenance((provenance,))
    facts = [
        _fact("phone_sample_count", result.sample_count, "count"),
        _fact("phone_unique_levels", result.unique_levels, "count"),
        _fact("phone_zero_fraction", result.zero_fraction, "ratio"),
        _fact(
            "phone_order_only_transition_count",
            result.transition_count,
            "ordered_changes",
        ),
    ]
    facts.extend(
        _fact(f"phone_{item.key}", item.value, item.unit or "value") for item in result.metrics
    )
    limitations = [
        *boundaries,
        *result.warnings,
        (
            "The temporal axis is invalidated; transition count is order-only and no cadence or frequency is reported."
            if not result.temporal_axis_valid
            else "The temporal axis is valid only within the registered raw source boundary."
        ),
    ]
    return PublicLightEvidenceSnapshot(
        evidence_id="evidence-phone-perturbation",
        dataset_id=provenance.dataset_id,
        recording_ids=[provenance.recording_id or PHYPhOX_SNR_RECORDING_ID],
        data_class=provenance.data_class,
        device_scope=device_scope,
        source_title=source_title,
        source_url=source_url,
        doi=doi,
        license_spdx=license_spdx,
        facts=facts[:32],
        processing_disclosures=disclosures,
        claim_boundary=_unique_text(limitations),
    )


def _context_snapshot(
    root: Path,
    result: NaturalisticLightContextResult,
) -> PublicLightEvidenceSnapshot:
    provenance = result.provenance
    (
        source_title,
        source_url,
        doi,
        license_spdx,
        device_scope,
        disclosures,
        boundaries,
    ) = _combined_provenance((provenance,))
    facts = [
        _fact("brighter_participant_series_count", result.participant_count, "count"),
        _fact("brighter_observation_count", result.observation_count, "count"),
        _fact(
            "brighter_participant_median_q25",
            result.participant_median_lux_distribution.q25,
            "lx",
        ),
        _fact(
            "brighter_participant_median",
            result.participant_median_lux_distribution.median,
            "lx",
        ),
        _fact(
            "brighter_participant_median_q75",
            result.participant_median_lux_distribution.q75,
            "lx",
        ),
    ]
    if result.query_percentile_rank_among_participant_medians is not None:
        facts.append(
            _fact(
                "brighter_query_percentile_rank",
                result.query_percentile_rank_among_participant_medians,
                "percent",
            )
        )
    return PublicLightEvidenceSnapshot(
        evidence_id="evidence-naturalistic-context",
        dataset_id=result.dataset_id,
        recording_ids=[],
        data_class=provenance.data_class,
        device_scope=device_scope,
        source_title=source_title,
        source_url=source_url,
        doi=doi,
        license_spdx=license_spdx,
        facts=facts,
        processing_disclosures=disclosures,
        claim_boundary=_unique_text([*boundaries, *result.limitations]),
    )


def _execute_candidate(
    root: Path,
    candidate_id: str,
    request: PublicLightExploreRequest,
) -> tuple[PublicLightEvidenceSnapshot, PublicLightToolExecution]:
    try:
        if candidate_id == _COMPARE_ID:
            _assert_manifest_replay_access(root, PRIVACY_DUAL_DATASET_ID, request)
            result = compare_registered_light_conditions(
                root,
                PRIVACY_DUAL_DATASET_ID,
                "privacy-dual-occluder",
                "privacy-dual-touch",
            )
            evidence = _comparison_snapshot(result)
            tool_id = _COMPARE_TOOL
            result_code = result.result_kind
        elif candidate_id == _PHONE_TRACE_ID:
            _assert_manifest_replay_access(root, PHYPhOX_SNR_DATASET_ID, request)
            result = inspect_public_light_trace(
                root,
                PHYPhOX_SNR_DATASET_ID,
                PHYPhOX_SNR_RECORDING_ID,
            )
            evidence = _trace_snapshot(result)
            tool_id = _TRACE_TOOL
            result_code = result.result_kind
        elif candidate_id == _NATURALISTIC_ID:
            _assert_manifest_replay_access(root, BRIGHTER_TIME_DATASET_ID, request)
            result = summarize_naturalistic_light_context(
                root,
                BRIGHTER_TIME_DATASET_ID,
                request.query_illuminance_lx,
            )
            evidence = _context_snapshot(root, result)
            tool_id = _CONTEXT_TOOL
            result_code = result.result_kind
        else:  # pragma: no cover - callers only execute frozen non-terminal candidates
            raise PublicLightExplorationUnavailable("unknown-execution-candidate")
    except PublicLightExplorationUnavailable:
        raise
    except (KeyError, OSError, ValueError) as exc:
        raise PublicLightExplorationUnavailable("source-validation-failed") from exc
    return evidence, PublicLightToolExecution(
        sequence=1,
        tool_id=tool_id,
        evidence_ids=[evidence.evidence_id],
        result_codes=[result_code],
    )


def _assert_manifest_replay_access(
    root: Path,
    dataset_id: str,
    request: PublicLightExploreRequest,
) -> None:
    try:
        _pack_dir, manifest = get_public_replay_dataset(root, dataset_id)
    except (KeyError, OSError, ValueError) as exc:
        raise PublicLightExplorationUnavailable("source-validation-failed") from exc
    review = manifest.privacy_review
    if "local_replay" not in review.allowed_operations:
        raise PublicLightExplorationUnavailable("privacy-operation-not-allowed")
    if review.requires_user_acknowledgement and not request.privacy_acknowledged:
        raise PublicLightExplorationUnavailable("privacy-not-acknowledged")


def _evidence_refs(evidence: Sequence[PublicLightEvidenceSnapshot]) -> list[str]:
    refs: list[str] = []
    for item in evidence:
        refs.extend(f"{item.dataset_id}/{recording_id}" for recording_id in item.recording_ids)
        if item.dataset_id == BRIGHTER_TIME_DATASET_ID:
            refs = [value for value in refs if not value.startswith(f"{item.dataset_id}/")]
            refs.append(item.dataset_id)
    return _unique_text(refs)


def _audit_claim(
    family: str,
    evidence: Sequence[PublicLightEvidenceSnapshot],
) -> tuple[LightClaimAuditResult, PublicLightToolExecution]:
    claim_kind = _claim_kind_for_family(family)
    audit = audit_light_claim_support(claim_kind, _evidence_refs(evidence))
    return audit, PublicLightToolExecution(
        sequence=1,
        tool_id=_AUDIT_TOOL,
        evidence_ids=[item.evidence_id for item in evidence],
        result_codes=[audit.status, audit.claim_kind],
    )


def _fact_value(
    evidence: Sequence[PublicLightEvidenceSnapshot],
    key: str,
) -> float | None:
    for snapshot in evidence:
        for fact in snapshot.facts:
            if fact.key == key:
                return fact.value
    return None


def _next_live_measurement(family: str) -> str:
    if family == "naturalistic_context":
        return (
            "在自己的目标环境中使用同一台手机的 phyphox Light；固定手机朝向与位置，"
            "记录背景并在至少三个时段各重复三次，同时保存环境与控制条件。"
        )
    if family == "unsupported_distance_law":
        return (
            "使用同一台手机和固定点光源，记录背景；固定朝向，在至少四个已测距离各重复"
            "三次，并在拟合前检查饱和、背景占比和距离跨度。"
        )
    if family == "unsupported_absolute_calibration":
        return (
            "将同一手机与可溯源照度计并置，在多档稳定光照下各重复三次；先冻结几何、"
            "增益与参考仪器，再评估校准误差。"
        )
    return (
        "使用同一台手机的 phyphox Light，固定光源、距离、角度与环境背景；"
        "对基线和一个标准遮挡条件分别独立重复三次，并保留原始导出与控制记录。"
    )


def _privacy_report() -> PublicLightReport:
    return PublicLightReport(
        conclusion_kind="privacy_acknowledgement_required",
        title="需要先确认本地隐私边界",
        summary=(
            "这个问题需要读取保留行为光照节律的公开序列。PocketLab 没有读取这些记录，"
            "也没有调用模型；请只在本机理解风险后显式确认，再重新运行。"
        ),
        uncertainties=["未读取任何隐私敏感公开记录，因此当前不能形成实验结论。"],
        forbidden_claims=["不得绕过确认、导入账号历史、导出或从光照节律识别个人行为。"],
        next_live_measurement=None,
    )


def _report_sources(
    evidence: Sequence[PublicLightEvidenceSnapshot],
) -> list[PublicLightReportSource]:
    sources: list[PublicLightReportSource] = []
    seen: set[str] = set()
    for item in evidence:
        if item.dataset_id in seen:
            continue
        seen.add(item.dataset_id)
        sources.append(
            PublicLightReportSource(
                dataset_id=item.dataset_id,
                data_class=item.data_class,
                device_scope=item.device_scope,
                source_title=item.source_title,
                source_url=item.source_url,
                doi=item.doi,
                license_spdx=item.license_spdx,
            )
        )
    return sources


def _finding(
    finding_id: str,
    text: str,
    evidence_ids: Sequence[str],
) -> PublicLightFinding:
    return PublicLightFinding(
        finding_id=finding_id,
        text=text,
        evidence_ids=list(evidence_ids),
    )


def _report(
    family: str,
    selected_ids: Sequence[str],
    evidence: Sequence[PublicLightEvidenceSnapshot],
    audit: LightClaimAuditResult,
    execution_failures: Sequence[str],
) -> PublicLightReport:
    evidence_ids = [item.evidence_id for item in evidence]
    source_ids = _unique_text(item.dataset_id for item in evidence)
    uncertainties = _unique_text(
        [limitation for item in evidence for limitation in item.claim_boundary]
    )[:16]
    forbidden = list(audit.forbidden_phrasing)[:16]
    findings: list[PublicLightFinding] = []
    conclusion_kind = "limited"
    title = "公开 Light 证据只能形成有边界的结论"
    summary = "公开来源已经过注册、哈希和确定性工具校验，但它们不是当前用户设备的受控重复。"
    if family.startswith("unsupported_") or _UNSUPPORTED_ID in selected_ids:
        conclusion_kind = "unsupported"
        title = "公开回放不足以支持该结论"
        summary = "当前问题越过了这些公开数据的控制、重复或校准边界，因此系统主动停止。"
        uncertainties = list(audit.required_missing_evidence) or [
            "缺少当前结论所需的受控真实手机证据。"
        ]
    elif family == "registered_condition_comparison":
        privacy_evidence = [
            item.evidence_id for item in evidence if item.dataset_id == PRIVACY_DUAL_DATASET_ID
        ]
        phone_evidence = [
            item.evidence_id for item in evidence if item.dataset_id == PHYPhOX_SNR_DATASET_ID
        ]
        difference = _fact_value(evidence, "privacy_median_difference")
        ratio = _fact_value(evidence, "privacy_median_ratio")
        if difference is not None:
            findings.append(
                _finding(
                    "privacy-median-difference",
                    f"两条注册采集的中位照度描述性差值为 {difference:.3g} lx。",
                    privacy_evidence,
                )
            )
        if ratio is not None:
            findings.append(
                _finding(
                    "privacy-median-ratio",
                    f"两条注册采集的中位照度描述性比值为 {ratio:.3g}。",
                    privacy_evidence,
                )
            )
        if _PHONE_TRACE_ID in selected_ids and phone_evidence:
            findings.append(
                _finding(
                    "phone-order-crosscheck",
                    "另一条真实 Android 手机 phyphox 派生序列也包含有序照度变化；"
                    "该交叉检查保持来源分离，不做绝对 lux 合并。",
                    phone_evidence,
                )
            )
        conclusion_kind = "supported_descriptive"
        title = "遮挡条件间存在可复核的描述性差异"
        summary = (
            "公开平板的两个已注册条件在照度统计上不同；这说明 ALS 序列可呈现扰动，"
            "但单次条件不能证明因果、动作类别或对用户手机的泛化。"
        )
    elif family == "naturalistic_context":
        context_evidence = [
            item.evidence_id for item in evidence if item.dataset_id == BRIGHTER_TIME_DATASET_ID
        ]
        q25 = _fact_value(evidence, "brighter_participant_median_q25")
        median = _fact_value(evidence, "brighter_participant_median")
        q75 = _fact_value(evidence, "brighter_participant_median_q75")
        if q25 is not None and median is not None and q75 is not None:
            findings.append(
                _finding(
                    "brighter-participant-distribution",
                    f"公开参与者级中位照度分布的 Q25/中位数/Q75 为 "
                    f"{q25:.3g}/{median:.3g}/{q75:.3g} lx。",
                    context_evidence,
                )
            )
        percentile = _fact_value(evidence, "brighter_query_percentile_rank")
        if percentile is not None:
            findings.append(
                _finding(
                    "brighter-query-percentile",
                    f"查询值位于该公开样本参与者中位照度的约第 {percentile:.1f} 百分位。",
                    context_evidence,
                )
            )
        conclusion_kind = "supported_descriptive"
        title = "获得了自然环境照度的描述性上下文"
        summary = "结果只定位于公开参与者级摘要分布，不能作为健康阈值、设备校准或时间规律。"
    else:
        phone_evidence = [
            item.evidence_id for item in evidence if item.dataset_id == PHYPhOX_SNR_DATASET_ID
        ]
        transitions = _fact_value(evidence, "phone_order_only_transition_count")
        if transitions is not None:
            findings.append(
                _finding(
                    "phone-order-only-changes",
                    f"注册的手机派生序列包含 {transitions:.0f} 次相邻数值变化；"
                    "由于时间轴无效，该数字只表示序列顺序变化。",
                    phone_evidence,
                )
            )
        title = "公开手机序列支持有限的扰动可见性结论"
        summary = (
            "一条真实 Android 手机的 phyphox Light 派生序列包含照度变化，"
            "但作者处理使采样节律无效，且单次采集不能证明鲁棒性或设备泛化。"
        )
    if _LIVE_ID in selected_ids:
        conclusion_kind = "live_measurement_required"
        title = "公开证据已到边界，下一步需要真实手机测量"
    if execution_failures and conclusion_kind == "supported_descriptive":
        conclusion_kind = "limited"
        title = "已保留首份证据，但后续公开来源校验失败"
        uncertainties = _unique_text(
            [
                *uncertainties,
                *(
                    f"后续确定性工具失败：{reason}；系统没有替换来源或伪造证据。"
                    for reason in execution_failures
                ),
            ]
        )[:16]
    return PublicLightReport(
        conclusion_kind=conclusion_kind,
        title=title,
        summary=summary,
        supported_findings=findings[:16],
        uncertainties=uncertainties or ["公开回放不代表当前用户设备和现场条件。"],
        forbidden_claims=forbidden or ["不得将公开回放计入 Gate C 或宣称 agent_ready。"],
        next_live_measurement=_next_live_measurement(family),
        evidence_ids=evidence_ids,
        source_ids=source_ids,
        sources=_report_sources(evidence),
    )


def _planner_status(traces: Sequence[PublicLightPlannerTrace]) -> str:
    sources = {item.source for item in traces}
    if sources == {"agent"}:
        return "accepted"
    if sources == {"strong_workflow_fallback"}:
        return "fallback"
    return "mixed"


async def run_public_light_exploration(
    request: PublicLightExploreRequest,
    *,
    root: Path,
    planner: PlannerCallable | None = None,
) -> PublicLightExploreResult:
    """Run a stateless public-data Light loop with bounded Agent proposals.

    The model can only select frozen candidate IDs. Source validation, privacy,
    deterministic tools, claim audit, termination and the report remain owned by
    this service. No public recording is written to a user account.
    """

    run_id = f"public-light-{uuid4().hex}"
    family = _question_family(request)
    candidates, fallback_id = _initial_candidates(request, family)
    initial_request = _planner_request(
        run_id=run_id,
        step=1,
        operation="select_initial_evidence",
        request=request,
        evidence=(),
        candidates=candidates,
        fallback_candidate_id=fallback_id,
    )
    if fallback_id == _PRIVACY_ACK_ID:
        trace = PublicLightPlannerTrace(
            step=1,
            operation="select_initial_evidence",
            request_sha256=initial_request.request_sha256,
            candidate_ids=[item.candidate_id for item in candidates],
            selected_candidate_id=_PRIVACY_ACK_ID,
            fallback_candidate_id=_PRIVACY_ACK_ID,
            rationale_code="privacy_not_acknowledged",
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason="privacy-not-acknowledged",
        )
        return PublicLightExploreResult(
            run_id=run_id,
            research_question=request.research_question,
            execution_status="limited",
            selected_route_id=_PRIVACY_ACK_ID,
            planner_status="fallback",
            planner_trace=[trace],
            report=_privacy_report(),
        )

    if family == "registered_condition_comparison" and _question_mentions_phone_transfer(
        request.research_question
    ):
        selected_id = _COMPARE_ID
        first_trace = PublicLightPlannerTrace(
            step=1,
            operation="select_initial_evidence",
            request_sha256=initial_request.request_sha256,
            candidate_ids=[item.candidate_id for item in candidates],
            selected_candidate_id=selected_id,
            fallback_candidate_id=fallback_id,
            rationale_code="match_registered_condition_comparison",
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason="protocol-order",
        )
    elif family.startswith("unsupported_"):
        selected_id = _UNSUPPORTED_ID
        first_trace = PublicLightPlannerTrace(
            step=1,
            operation="select_initial_evidence",
            request_sha256=initial_request.request_sha256,
            candidate_ids=[item.candidate_id for item in candidates],
            selected_candidate_id=selected_id,
            fallback_candidate_id=fallback_id,
            rationale_code="unsupported_claim_boundary",
            source="strong_workflow_fallback",
            outcome="fallback",
            fallback_reason="policy-boundary",
        )
    else:
        selected_id, first_trace = await _select_candidate(
            initial_request,
            planner=planner,
        )
    traces = [first_trace]
    selected_ids = [selected_id]
    execution_family = _effective_family(family, selected_id)
    evidence: list[PublicLightEvidenceSnapshot] = []
    tool_trace: list[PublicLightToolExecution] = []
    execution_failures: list[str] = []

    initial_candidate = next(item for item in candidates if item.candidate_id == selected_id)
    if not initial_candidate.terminal:
        snapshot, execution = _execute_candidate(root, selected_id, request)
        evidence.append(snapshot)
        tool_trace.append(execution)

        if selected_id != _NATURALISTIC_ID:
            follow_candidates, follow_fallback = _follow_up_candidates(
                request,
                execution_family,
                selected_ids,
            )
            follow_request = _planner_request(
                run_id=run_id,
                step=2,
                operation="select_follow_up",
                request=request,
                evidence=evidence,
                candidates=follow_candidates,
                fallback_candidate_id=follow_fallback,
            )
            follow_id, follow_trace = await _select_candidate(
                follow_request,
                planner=planner,
            )
            traces.append(follow_trace)
            selected_ids.append(follow_id)
            follow_candidate = next(
                item for item in follow_candidates if item.candidate_id == follow_id
            )
            if not follow_candidate.terminal:
                try:
                    snapshot, execution = _execute_candidate(root, follow_id, request)
                except PublicLightExplorationUnavailable as exc:
                    execution_failures.append(exc.reason)
                    tool_trace.append(
                        PublicLightToolExecution(
                            sequence=len(tool_trace) + 1,
                            tool_id=follow_candidate.tool_ids[0],
                            status="failed",
                            result_codes=[exc.reason],
                        )
                    )
                else:
                    if snapshot.evidence_id not in {item.evidence_id for item in evidence}:
                        execution.sequence = len(tool_trace) + 1
                        evidence.append(snapshot)
                        tool_trace.append(execution)

    audit, audit_execution = _audit_claim(execution_family, evidence)
    audit_execution.sequence = len(tool_trace) + 1
    tool_trace.append(audit_execution)
    if len(tool_trace) > 3:  # pragma: no cover - frozen graph cannot exceed this
        raise PublicLightExplorationUnavailable("tool-budget-exceeded")

    report = _report(
        execution_family,
        selected_ids,
        evidence,
        audit,
        execution_failures,
    )
    execution_status = (
        "unsupported"
        if report.conclusion_kind == "unsupported"
        else "completed"
        if report.conclusion_kind == "supported_descriptive"
        else "limited"
    )
    return PublicLightExploreResult(
        run_id=run_id,
        research_question=request.research_question,
        execution_status=execution_status,
        selected_route_id=selected_ids[-1],
        planner_status=_planner_status(traces),
        planner_trace=traces,
        tool_trace=tool_trace,
        evidence=evidence,
        report=report,
    )


def strong_workflow_route(
    request: PublicLightExploreRequest,
) -> tuple[str, str]:
    """Expose the preregistered baseline route for paired Harness evaluation."""

    family = _question_family(request)
    _, fallback_id = _initial_candidates(request, family)
    return family, fallback_id
