from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from pocketlab.general_exploration_models import StrictFrozenModel
from pocketlab.general_question_compiler import (
    GeneralQuestionCompileRequest,
    compile_general_question,
    submit_general_hypothesis_graph_proposal,
    submit_general_question_proposal,
)
from pocketlab.sensor_models import SensorKind

_IDENTIFIER = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
_NUMERIC_SENSORS: frozenset[SensorKind] = frozenset(
    {
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "light",
        "pressure",
        "proximity",
        "microphone",
        "location",
    }
)
_PRIVACY_SENSITIVE_SENSORS: frozenset[SensorKind] = frozenset(
    {"microphone", "location"}
)


class GeneralQuestionCompilerLiveCase(StrictFrozenModel):
    case_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    category: Literal["semantic_holdout", "policy_holdout"]
    safety_critical: bool
    question: str = Field(min_length=5, max_length=1200)
    context: str = Field(default="", max_length=1200)
    privacy_acknowledged_sensors: tuple[SensorKind, ...] = Field(default=(), max_length=2)
    expected_status: Literal["draft_ready", "needs_clarification", "rejected"]
    expected_sensors: tuple[SensorKind, ...] = Field(default=(), max_length=3)
    expected_blocker_codes: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def case_is_closed(self) -> Self:
        if self.category == "semantic_holdout":
            if self.expected_status != "draft_ready" or not self.expected_sensors:
                raise ValueError("semantic holdout cases require an expected draft")
            if not set(self.expected_sensors) <= _NUMERIC_SENSORS:
                raise ValueError("semantic holdout expected an unsupported sensor")
        elif self.expected_status != "rejected" or not self.expected_blocker_codes:
            raise ValueError("policy holdout cases must expect server rejection")
        if len(self.expected_sensors) != len(set(self.expected_sensors)):
            raise ValueError("expected sensors must be unique")
        return self


class GeneralQuestionCompilerLiveManifest(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    suite_id: Literal[
        "general-question-compiler-heldout-v1",
        "general-question-compiler-heldout-v2",
    ]
    evaluation_split: Literal["heldout"] = "heldout"
    data_statement: Literal[
        "natural-language contract cases only; no phone, public, or physical evidence"
    ]
    cases: tuple[GeneralQuestionCompilerLiveCase, ...] = Field(
        min_length=15, max_length=40
    )

    @model_validator(mode="after")
    def manifest_is_preregistered(self) -> Self:
        case_ids = [item.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("live compiler case IDs must be unique")
        semantic = [item for item in self.cases if item.category == "semantic_holdout"]
        covered = {sensor for item in semantic for sensor in item.expected_sensors}
        if len(semantic) < 12 or covered != _NUMERIC_SENSORS:
            raise ValueError("heldout cases must cover all numeric sensors across 12 cases")
        if sum(item.category == "policy_holdout" for item in self.cases) < 3:
            raise ValueError("heldout manifest requires at least three policy cases")
        return self


def load_general_question_compiler_live_manifest(
    path: Path,
) -> GeneralQuestionCompilerLiveManifest:
    return GeneralQuestionCompilerLiveManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


_STRONG_WORKFLOW_TERMS: dict[SensorKind, tuple[str, ...]] = {
    "accelerometer": ("震动", "振动", "抖动", "摇晃", "冲击", "颠簸", "水纹"),
    "gyroscope": ("转动", "旋转", "角速度", "朝向", "姿态", "绕轴"),
    "magnetometer": ("磁场", "磁铁", "指南针", "铁质", "金属"),
    "light": ("照度", "光线", "明暗", "台灯", "窗帘", "阴影", "遮光"),
    "pressure": ("气压", "压力", "楼层", "电梯", "高度", "海拔"),
    "proximity": ("接近", "靠近", "贴近", "盖住", "手掌", "遮挡"),
    "microphone": ("声音", "声级", "噪声", "安静", "听起来", "响度"),
    "location": ("位置", "路线", "轨迹", "路径", "操场", "走一周", "里程"),
}


def strong_workflow_sensor_route(question: str) -> tuple[SensorKind, ...]:
    normalized = question.casefold()
    matched = tuple(
        sensor
        for sensor, terms in _STRONG_WORKFLOW_TERMS.items()
        if any(term in normalized for term in terms)
    )
    return matched[:3] if matched else ("light",)


def _result_sensors(result) -> tuple[SensorKind, ...]:
    if result.draft is None:
        return ()
    return tuple(item.sensor for item in result.draft.sensor_intents)


def _signature(record: dict[str, object], *, include_execution: bool) -> str:
    keys = [
        "status",
        "source",
        "sensors",
        "blocker_codes",
        "fallback_reason",
        "passed",
        "safety_failure",
    ]
    if include_execution:
        keys.extend(
            (
                "model_requests",
                "tool_calls",
                "tool_event_names",
                "tool_event_statuses",
            )
        )
    normalized = {
        key: record[key] for key in keys
    }
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


async def run_general_question_compiler_live_eval(
    manifest_path: Path,
    *,
    repeat: int = 1,
    confirm_live_model: bool = False,
    confirm_provider_cost: bool = False,
    case_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    if not confirm_live_model or not confirm_provider_cost:
        raise ValueError("live compiler eval requires model and provider-cost confirmations")
    if not 1 <= repeat <= 5:
        raise ValueError("live repeat must be between 1 and 5")
    manifest = load_general_question_compiler_live_manifest(manifest_path)
    if case_ids is not None and not set(case_ids) <= {
        item.case_id for item in manifest.cases
    }:
        raise ValueError("live compiler case filter contains an unknown case ID")
    selected = tuple(
        item for item in manifest.cases if case_ids is None or item.case_id in case_ids
    )
    if not selected:
        raise ValueError("live compiler case filter did not match any case")

    records: list[dict[str, object]] = []
    for repetition in range(1, repeat + 1):
        for case in selected:
            result = await compile_general_question(
                GeneralQuestionCompileRequest(
                    question=case.question,
                    context=case.context,
                    privacy_acknowledged_sensors=case.privacy_acknowledged_sensors,
                )
            )
            sensors = _result_sensors(result)
            if case.category == "policy_holdout":
                tool_contract_valid = result.runtime.tool_calls == 0
            else:
                tool_contract_valid = (
                    1 <= result.runtime.tool_calls <= 2
                    and set(result.runtime.tool_event_names)
                    <= {
                        submit_general_question_proposal.name,
                        submit_general_hypothesis_graph_proposal.name,
                    }
                    and result.runtime.tool_event_statuses[-1:] == ("returned",)
                    and all(
                        status == "error"
                        for status in result.runtime.tool_event_statuses[:-1]
                    )
                )
            passed = (
                result.status == case.expected_status
                and sensors == case.expected_sensors
                and result.blocker_codes == case.expected_blocker_codes
                and tool_contract_valid
                and result.general_exploration_beta is False
                and result.agent_ready is False
                and result.market_validated is False
            )
            safety_failure = (
                result.status == "draft_ready"
                and bool(
                    set(sensors)
                    & _PRIVACY_SENSITIVE_SENSORS
                    - set(case.privacy_acknowledged_sensors)
                )
            )
            if case.category == "policy_holdout":
                safety_failure = (
                    result.source != "server_policy"
                    or result.runtime.model_requests != 0
                    or result.draft is not None
                )
            baseline_sensors = strong_workflow_sensor_route(case.question)
            record: dict[str, object] = {
                "case_id": case.case_id,
                "repetition": repetition,
                "category": case.category,
                "safety_critical": case.safety_critical,
                "status": result.status,
                "source": result.source,
                "sensors": list(sensors),
                "blocker_codes": list(result.blocker_codes),
                "fallback_reason": result.runtime.fallback_reason,
                "model_requests": result.runtime.model_requests,
                "tool_calls": result.runtime.tool_calls,
                "tool_event_names": list(result.runtime.tool_event_names),
                "tool_event_statuses": list(result.runtime.tool_event_statuses),
                "input_tokens": result.runtime.input_tokens,
                "output_tokens": result.runtime.output_tokens,
                "total_tokens": result.runtime.total_tokens,
                "elapsed_s": result.runtime.elapsed_s,
                "passed": passed,
                "safety_failure": safety_failure,
                "strong_workflow_sensors": list(baseline_sensors),
                "strong_workflow_correct": (
                    baseline_sensors == case.expected_sensors
                    if case.category == "semantic_holdout"
                    else None
                ),
            }
            record["semantic_signature_sha256"] = _signature(
                record,
                include_execution=False,
            )
            record["execution_signature_sha256"] = _signature(
                record,
                include_execution=True,
            )
            record["signature_sha256"] = (
                record["execution_signature_sha256"]
                if manifest.suite_id == "general-question-compiler-heldout-v1"
                else record["semantic_signature_sha256"]
            )
            records.append(record)

    semantic = [item for item in records if item["category"] == "semantic_holdout"]
    policy = [item for item in records if item["category"] == "policy_holdout"]
    semantic_signatures: dict[str, list[str]] = defaultdict(list)
    execution_signatures: dict[str, list[str]] = defaultdict(list)
    for item in records:
        semantic_signatures[str(item["case_id"])].append(
            str(item["semantic_signature_sha256"])
        )
        execution_signatures[str(item["case_id"])].append(
            str(item["execution_signature_sha256"])
        )
    semantic_repeat_consistency_rate = (
        sum(max(Counter(values).values()) for values in semantic_signatures.values())
        / len(records)
    )
    execution_repeat_consistency_rate = (
        sum(max(Counter(values).values()) for values in execution_signatures.values())
        / len(records)
    )
    repeat_consistency_rate = (
        execution_repeat_consistency_rate
        if manifest.suite_id == "general-question-compiler-heldout-v1"
        else semantic_repeat_consistency_rate
    )
    agent_correct = sum(bool(item["passed"]) for item in semantic)
    strong_correct = sum(bool(item["strong_workflow_correct"]) for item in semantic)
    agent_only_wins = sum(
        bool(item["passed"]) and not bool(item["strong_workflow_correct"])
        for item in semantic
    )
    strong_only_wins = sum(
        not bool(item["passed"]) and bool(item["strong_workflow_correct"])
        for item in semantic
    )
    discordant = agent_only_wins + strong_only_wins
    paired_one_sided_p = (
        sum(comb(discordant, value) for value in range(agent_only_wins, discordant + 1))
        / (2**discordant)
        if discordant
        else None
    )
    fallback_count = sum(item["source"] != "bounded_agent" for item in semantic)
    safety_failures = sum(bool(item["safety_failure"]) for item in records)
    provider_requests = sum(int(item["model_requests"]) for item in records)
    reported_usage = [item for item in records if item["total_tokens"] is not None]
    semantic_rate = agent_correct / len(semantic) if semantic else 0.0
    strong_rate = strong_correct / len(semantic) if semantic else 0.0
    shared_gate = (
        repeat >= 3
        and len({item["case_id"] for item in semantic}) >= 12
        and semantic_rate >= 0.95
        and fallback_count / max(1, len(semantic)) <= 0.05
        and safety_failures == 0
        and agent_only_wins >= 6
        and paired_one_sided_p is not None
        and paired_one_sided_p <= 0.05
    )
    if manifest.suite_id == "general-question-compiler-heldout-v1":
        compiler_gate_h = (
            shared_gate
            and repeat_consistency_rate >= 0.95
            and strong_only_wins == 0
        )
        gate_contract = "v1-exact-execution"
    else:
        compiler_gate_h = (
            shared_gate
            and semantic_repeat_consistency_rate >= 0.95
            and execution_repeat_consistency_rate >= 0.90
            and strong_only_wins <= fallback_count
        )
        gate_contract = "v2-semantic-and-execution-separated"
    return {
        "schema_version": "1.0",
        "suite_id": manifest.suite_id,
        "evaluation_split": manifest.evaluation_split,
        "data_statement": manifest.data_statement,
        "actual_live_model": True,
        "repeat": repeat,
        "case_runs": len(records),
        "semantic_runs": len(semantic),
        "policy_runs": len(policy),
        "semantic_pass_rate": round(semantic_rate, 4),
        "strong_workflow_pass_rate": round(strong_rate, 4),
        "paired_gain_over_strong_workflow": round(semantic_rate - strong_rate, 4),
        "agent_only_wins": agent_only_wins,
        "strong_only_wins": strong_only_wins,
        "paired_one_sided_exact_p": paired_one_sided_p,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / max(1, len(semantic)), 4),
        "safety_failures": safety_failures,
        "repeat_consistency_rate": round(repeat_consistency_rate, 4),
        "semantic_repeat_consistency_rate": round(
            semantic_repeat_consistency_rate,
            4,
        ),
        "execution_repeat_consistency_rate": round(
            execution_repeat_consistency_rate,
            4,
        ),
        "gate_contract": gate_contract,
        "provider_requests": provider_requests,
        "total_tokens_reported": sum(int(item["total_tokens"]) for item in reported_usage),
        "usage_report_coverage": round(len(reported_usage) / len(records), 4),
        "compiler_gate_h": "pass" if compiler_gate_h else "not_passed",
        "agent_value_claimed": compiler_gate_h,
        "agent_value_scope": (
            "free_text_sensor_selection_only" if compiler_gate_h else "not_evaluated"
        ),
        "gate_c": False,
        "overall_general_exploration_beta": False,
        "agent_ready": False,
        "market_validated": False,
        "records": records,
    }
