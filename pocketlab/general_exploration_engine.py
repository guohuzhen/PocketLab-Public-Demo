from __future__ import annotations

import hashlib
import json
from statistics import median
from typing import Literal

from pocketlab.experiment_guidance import (
    assert_experiment_operation_guide,
    build_experiment_operation_guide,
)
from pocketlab.general_acquisition import (
    GeneralEvidenceEnvelope,
    build_condition_evidence_group,
)
from pocketlab.general_exploration_models import (
    GeneralExperimentProtocol,
    GeneralExplorationCompilation,
)
from pocketlab.general_exploration_state import (
    GeneralAdaptiveSufficiencyAssessment,
    GeneralAuxiliaryObservation,
    GeneralComparisonSeries,
    GeneralCompilerProvenance,
    GeneralConditionMetricSummary,
    GeneralDesignCandidate,
    GeneralDesignDecisionTrace,
    GeneralExperimentCase,
    GeneralExperimentReport,
    GeneralExperimentTask,
    GeneralHypothesisAssessment,
    GeneralHypothesisConclusionAudit,
    GeneralHypothesisObservationAssessment,
    GeneralHypothesisTerminationAudit,
    GeneralMeasurementSubmission,
    GeneralMetricContrast,
    GeneralPlannerDecisionAudit,
    GeneralReasoningCheckpoint,
    GeneralReasoningCheckpointDecision,
    GeneralReasoningReceipt,
    GeneralTerminationVector,
    GeneralVisualizationArtifact,
    GeneralVisualizationPoint,
    PreparedGeneralTransition,
)
from pocketlab.sensor_models import SensorKind

SelectionSource = Literal[
    "deterministic_fallback",
    "deterministic_policy",
    "bounded_agent",
    "reasoning_agent",
]

HYPOTHESIS_TERMINATION_POLICY_ID = "server-hypothesis-termination-gate-v2"
HYPOTHESIS_CONCLUSION_POLICY_ID = "server-hypothesis-conclusion-v1"

_SENSOR_DISPLAY_NAMES = {
    "accelerometer": "加速度计",
    "gyroscope": "陀螺仪",
    "magnetometer": "磁力计",
    "light": "光线",
    "pressure": "气压",
    "proximity": "接近距离",
    "microphone": "麦克风",
    "location": "位置",
    "bluetooth": "蓝牙",
}

_DIRECTION_DISPLAY_NAMES = {
    "increase": "明显升高",
    "decrease": "明显降低",
    "within_observed_repeatability": "未超出已观测重复波动",
}


class GeneralExperimentStateError(ValueError):
    """Raised before mutation when a general experiment transition is unsafe."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_normalized_uncertainty(
    prepared: PreparedGeneralTransition,
    candidate: GeneralDesignCandidate,
) -> float:
    """Return a server-derived, unitless within-slot uncertainty score."""

    case = prepared.base_case
    evidence = (*case.evidence, *prepared.submitted_evidence)
    completed = (*case.completed_tasks, prepared.completed_task)
    valid_ids = {
        evidence_id
        for task in completed
        if task.measurement_valid
        for evidence_id in task.output_evidence_ids
    }
    values_by_sensor: dict[SensorKind, list[float]] = {}
    for item in evidence:
        if (
            item.evidence_id in valid_ids
            and item.condition_id == candidate.condition_id
            and item.sensor in candidate.sensors
        ):
            values_by_sensor.setdefault(item.sensor, []).append(item.metric.value)
    score = 0.0
    for values in values_by_sensor.values():
        if len(values) < 2:
            continue
        scale = max(abs(sum(values) / len(values)), 1e-12)
        score += (max(values) - min(values)) / scale
    return score


def select_deterministic_information_candidate(
    prepared: PreparedGeneralTransition,
) -> str | None:
    """Select the highest observed normalized uncertainty with a stable fallback.

    This is the production strong workflow for numerical replicate ordering. It never
    interprets the user's question and therefore cannot choose an optional sensor probe.
    """

    prepared = PreparedGeneralTransition.model_validate(prepared.model_dump(mode="python"))
    if len(prepared.next_candidates) < 2:
        return prepared.fallback_candidate_id
    fallback = prepared.fallback_candidate_id
    if fallback is None:
        raise GeneralExperimentStateError("multi-candidate transitions require a fallback")
    scores = {
        item.candidate_id: candidate_normalized_uncertainty(prepared, item)
        for item in prepared.next_candidates
    }
    maximum = max(scores.values())
    if maximum - min(scores.values()) <= 1e-9:
        return fallback
    return next(
        item.candidate_id
        for item in prepared.next_candidates
        if scores[item.candidate_id] == maximum
    )


def _validated_compilation(
    compilation: GeneralExplorationCompilation,
) -> GeneralExplorationCompilation:
    return GeneralExplorationCompilation.model_validate(compilation.model_dump(mode="python"))


def _validated_case(case: GeneralExperimentCase) -> GeneralExperimentCase:
    return GeneralExperimentCase.model_validate(case.model_dump(mode="python"))


def _numeric_sensors(protocol: GeneralExperimentProtocol) -> tuple[SensorKind, ...]:
    return tuple(
        item.sensor
        for item in protocol.sensors
        if item.sensor != "bluetooth" and item.activation == "required"
    )


def _optional_probe_requirements(protocol: GeneralExperimentProtocol):
    return tuple(
        item
        for item in protocol.sensors
        if item.sensor != "bluetooth" and item.activation == "optional_probe"
    )


def _required_conditions(protocol: GeneralExperimentProtocol):
    return tuple(item for item in protocol.conditions if item.activation == "required")


def _optional_control_conditions(protocol: GeneralExperimentProtocol):
    return tuple(item for item in protocol.conditions if item.activation == "optional_control")


def _optional_activation_outcome(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
    *,
    target_condition_id: str,
) -> bool | None:
    rule = next(
        (
            item
            for item in protocol.optional_activation_rules
            if item.target_condition_id == target_condition_id
        ),
        None,
    )
    if rule is None:
        return None
    valid_probe_evidence_ids = {
        evidence_id
        for task in tasks
        if task.action == "probe_optional_sensor"
        and task.measurement_valid is True
        and rule.probe_sensor in task.sensors
        for evidence_id in task.output_evidence_ids
    }
    matches = [
        item
        for item in evidence
        if item.evidence_id in valid_probe_evidence_ids
        and item.valid
        and item.sensor == rule.probe_sensor
        and item.metric.key == rule.metric_key
        and item.metric.unit == rule.metric_unit
    ]
    if len(matches) != 1:
        return None
    if rule.comparator == "gt":
        return matches[0].metric.value > rule.threshold
    raise GeneralExperimentStateError("unknown optional activation comparator")


def _valid_slot_count(
    tasks: tuple[GeneralExperimentTask, ...],
    *,
    condition_id: str,
    sensor: SensorKind,
) -> int:
    return sum(
        task.measurement_valid is True
        and task.condition_id == condition_id
        and sensor in task.sensors
        for task in tasks
    )


def _condition_contrast_relation(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
    *,
    sensor: SensorKind,
    metric_key: str,
    metric_unit: str,
    reference_condition_id: str,
    comparison_condition_id: str,
) -> tuple[
    Literal["comparison-higher", "comparison-lower", "within-relative-deadband"],
    tuple[str, ...],
] | None:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    valid_ids = {
        evidence_id
        for task in tasks
        if task.measurement_valid is True
        for evidence_id in task.output_evidence_ids
    }

    def values_for(condition_id: str) -> tuple[tuple[float, str], ...]:
        return tuple(
            (item.metric.value, item.evidence_id)
            for item in evidence
            if item.evidence_id in valid_ids
            and item.condition_id == condition_id
            and item.sensor == sensor
            and item.metric.key == metric_key
            and item.metric.unit == metric_unit
            and evidence_by_id[item.evidence_id].valid
        )

    reference = values_for(reference_condition_id)
    comparison = values_for(comparison_condition_id)
    if not reference or not comparison:
        return None
    reference_center = float(median(value for value, _evidence_id in reference))
    comparison_center = float(median(value for value, _evidence_id in comparison))
    scale = max(abs(reference_center), abs(comparison_center), 1e-12)
    relative_delta = (comparison_center - reference_center) / scale
    relation: Literal[
        "comparison-higher",
        "comparison-lower",
        "within-relative-deadband",
    ] = (
        "within-relative-deadband"
        if abs(relative_delta) <= 0.05
        else "comparison-higher"
        if relative_delta > 0
        else "comparison-lower"
    )
    return relation, tuple(
        evidence_id for _value, evidence_id in (*reference, *comparison)
    )


def _relation_matches_expected(expected: str, observed: str) -> bool:
    if expected == "different_unspecified":
        return observed != "within-relative-deadband"
    return {
        "comparison_higher": "comparison-higher",
        "comparison_lower": "comparison-lower",
        "within_relative_deadband": "within-relative-deadband",
    }[expected] == observed


def _paired_optional_probe_state(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
) -> tuple[set[SensorKind], set[SensorKind], set[SensorKind]]:
    required_conditions = _required_conditions(protocol)
    optional_sensors = {item.sensor for item in _optional_probe_requirements(protocol)}
    started = {
        sensor
        for task in tasks
        for sensor in task.sensors
        if sensor in optional_sensors
    }
    completed = {
        sensor
        for sensor in optional_sensors
        if all(
            _valid_slot_count(tasks, condition_id=condition.condition_id, sensor=sensor) >= 1
            for condition in required_conditions
        )
    }
    predictions_by_observable: dict[
        tuple[SensorKind, str, str, str, str],
        list[str],
    ] = {}
    for hypothesis in protocol.hypotheses:
        for observation in hypothesis.observations:
            if observation.measurement_role != "discriminator" or observation.sensor not in completed:
                continue
            key = (
                observation.sensor,
                observation.metric_key,
                observation.metric_unit,
                observation.reference_condition_id,
                observation.comparison_condition_id,
            )
            predictions_by_observable.setdefault(key, []).append(observation.expected_relation)
    discriminated: set[SensorKind] = set()
    for (
        sensor,
        metric_key,
        metric_unit,
        reference_condition_id,
        comparison_condition_id,
    ), expected_relations in predictions_by_observable.items():
        if len(expected_relations) < 2 or len(set(expected_relations)) < 2:
            continue
        contrast = _condition_contrast_relation(
            protocol,
            tasks,
            evidence,
            sensor=sensor,
            metric_key=metric_key,
            metric_unit=metric_unit,
            reference_condition_id=reference_condition_id,
            comparison_condition_id=comparison_condition_id,
        )
        if contrast is None:
            continue
        matches = {
            _relation_matches_expected(expected_relation, contrast[0])
            for expected_relation in expected_relations
        }
        if matches == {False, True}:
            discriminated.add(sensor)
    return started, completed, discriminated


def _hypothesis_termination_audit(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
) -> GeneralHypothesisTerminationAudit:
    """Close the hypothesis gate or issue the sole evidence-based exemption.

    Merely completing the primary comparison is never enough to close a registered
    competition.  Every discriminator must be observed unless one already-observed
    shared discriminator separates all hypotheses, in which case the server records
    exactly which remaining observations were omitted and which evidence justified it.
    """

    hypothesis_ids = tuple(item.hypothesis_id for item in protocol.hypotheses)
    if not hypothesis_ids:
        return GeneralHypothesisTerminationAudit()
    discriminators = tuple(
        (hypothesis.hypothesis_id, observation)
        for hypothesis in protocol.hypotheses
        for observation in hypothesis.observations
        if observation.measurement_role == "discriminator"
    )
    registered_ids = tuple(observation.observation_id for _hypothesis_id, observation in discriminators)
    observed: dict[str, tuple[str, tuple[str, ...]]] = {}
    for _hypothesis_id, observation in discriminators:
        contrast = _condition_contrast_relation(
            protocol,
            tasks,
            evidence,
            sensor=observation.sensor,
            metric_key=observation.metric_key,
            metric_unit=observation.metric_unit,
            reference_condition_id=observation.reference_condition_id,
            comparison_condition_id=observation.comparison_condition_id,
        )
        if contrast is not None:
            observed[observation.observation_id] = contrast
    observed_ids = tuple(item for item in registered_ids if item in observed)
    if len(observed_ids) == len(registered_ids):
        return GeneralHypothesisTerminationAudit(
            registered_hypothesis_ids=hypothesis_ids,
            registered_discriminator_ids=registered_ids,
            observed_discriminator_ids=observed_ids,
            gate_satisfied=True,
            disposition="all-discriminators-observed",
        )

    by_observable: dict[tuple[str, str, str, str, str], list[tuple[str, object]]] = {}
    for hypothesis_id, observation in discriminators:
        key = (
            observation.sensor,
            observation.metric_key,
            observation.metric_unit,
            observation.reference_condition_id,
            observation.comparison_condition_id,
        )
        by_observable.setdefault(key, []).append((hypothesis_id, observation))
    hypothesis_set = set(hypothesis_ids)
    for entries in by_observable.values():
        if {hypothesis_id for hypothesis_id, _observation in entries} != hypothesis_set:
            continue
        basis_observation_ids = tuple(
            observation.observation_id for _hypothesis_id, observation in entries
        )
        if not all(item in observed for item in basis_observation_ids):
            continue
        observed_relation, source_evidence_ids = observed[basis_observation_ids[0]]
        matching_hypotheses = {
            hypothesis_id
            for hypothesis_id, observation in entries
            if _relation_matches_expected(observation.expected_relation, observed_relation)
        }
        if len(matching_hypotheses) != 1:
            continue
        waived_ids = tuple(item for item in registered_ids if item not in observed)
        return GeneralHypothesisTerminationAudit(
            registered_hypothesis_ids=hypothesis_ids,
            registered_discriminator_ids=registered_ids,
            observed_discriminator_ids=observed_ids,
            waived_discriminator_ids=waived_ids,
            gate_satisfied=True,
            disposition="remaining-discriminators-exempted",
            exemption_reason_code=(
                "observed-shared-discriminator-separates-all-hypotheses"
            ),
            exemption_basis_observation_ids=basis_observation_ids,
            source_evidence_ids=source_evidence_ids,
        )

    return GeneralHypothesisTerminationAudit(
        registered_hypothesis_ids=hypothesis_ids,
        registered_discriminator_ids=registered_ids,
        observed_discriminator_ids=observed_ids,
        unresolved_discriminator_ids=tuple(
            item for item in registered_ids if item not in observed
        ),
        gate_satisfied=False,
        disposition="pending-discriminator-evidence",
    )


def _input_evidence_ids(
    tasks: tuple[GeneralExperimentTask, ...],
) -> tuple[str, ...]:
    values = [
        evidence_id
        for task in tasks
        if task.measurement_valid
        for evidence_id in task.output_evidence_ids
    ]
    return tuple(values[-64:])


def _candidate_id(
    protocol: GeneralExperimentProtocol,
    *,
    condition_id: str,
    sensors: tuple[SensorKind, ...],
    repeat_index: int,
    action: str,
) -> str:
    digest = _canonical_sha256(
        {
            "protocol_id": protocol.protocol_id,
            "condition_id": condition_id,
            "sensors": sensors,
            "repeat_index": repeat_index,
            "action": action,
        }
    )
    return f"candidate-{digest[:16]}"


def _condition_candidate(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    *,
    condition_id: str,
    sensors: tuple[SensorKind, ...],
    repeat_index: int,
    action: Literal[
        "collect_condition",
        "collect_supporting_sensor",
        "replicate_condition",
        "correct_condition",
        "probe_optional_sensor",
        "probe_optional_condition",
    ],
    reason_code: Literal[
        "initial_baseline",
        "missing_condition",
        "missing_supporting_sensor",
        "replication_required",
        "quality_correction",
        "optional_sensor_probe",
        "optional_condition_probe",
    ],
    input_evidence_ids: tuple[str, ...] | None = None,
) -> GeneralDesignCandidate:
    condition = next(item for item in protocol.conditions if item.condition_id == condition_id)
    sensor_labels = "、".join(sensors)
    display_sensor_labels = "、".join(_SENSOR_DISPLAY_NAMES[sensor] for sensor in sensors)
    title = f"{condition.label} · {sensor_labels} · 第 {repeat_index} 次"
    simulation_only = set(protocol.selected_sources) == {"protocol_emulator"}
    instruction = build_experiment_operation_guide(
        core_instruction=(
            (
                f"在 PocketLab 模拟数据区域选择“{condition.label}”场景，"
                f"连续采集 8 秒并生成{display_sensor_labels}模拟记录；"
                "不根据预期结果修改、删减或挑选数据"
            )
            if simulation_only
            else (
                f"{condition.instruction} 使用 {sensor_labels} 完成本轮记录；"
                "不根据预期结果修改、删减或挑选数据"
            )
        ),
        sensors=sensors,
        variable_to_change=(
            f"把模拟场景设为“{condition.label}”"
            if simulation_only
            else f"将{protocol.independent_variable}设为“{condition.factor_level}”"
        ),
        controlled_variables=protocol.controls,
        default_duration_s=8,
        safety_notes=protocol.safety_notes,
        repeat_index=repeat_index,
        task_kind=(
            "baseline"
            if reason_code == "initial_baseline"
            else "correction"
            if reason_code == "quality_correction"
            else "replication"
            if reason_code == "replication_required"
            else "control"
        ),
        execution_mode="simulation" if simulation_only else "physical",
    )
    assert_experiment_operation_guide(
        instruction,
        sensors=sensors,
        execution_mode="simulation" if simulation_only else "physical",
    )
    return GeneralDesignCandidate(
        candidate_id=_candidate_id(
            protocol,
            condition_id=condition_id,
            sensors=sensors,
            repeat_index=repeat_index,
            action=action,
        ),
        action=action,
        condition_id=condition_id,
        sensors=sensors,
        repeat_index=repeat_index,
        title=title,
        instruction=instruction,
        reason_code=reason_code,
        input_evidence_ids=(
            _input_evidence_ids(tasks) if input_evidence_ids is None else input_evidence_ids
        ),
    )


def _next_candidates(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...] = (),
) -> tuple[GeneralDesignCandidate, ...]:
    sensors = _numeric_sensors(protocol)
    required_conditions = _required_conditions(protocol)
    if protocol.alignment == "simultaneous" and len(sensors) > 1:
        for repeat_index in range(
            1,
            protocol.evidence_policy.required_repeats_per_condition + 1,
        ):
            candidates: list[GeneralDesignCandidate] = []
            for condition_index, condition in enumerate(required_conditions):
                if (
                    _valid_slot_count(
                        tasks,
                        condition_id=condition.condition_id,
                        sensor=sensors[0],
                    )
                    >= repeat_index
                ):
                    continue
                action = "collect_condition" if repeat_index == 1 else "replicate_condition"
                reason = (
                    "initial_baseline"
                    if repeat_index == 1 and condition_index == 0
                    else "missing_condition"
                    if repeat_index == 1
                    else "replication_required"
                )
                candidates.append(
                    _condition_candidate(
                        protocol,
                        tasks,
                        condition_id=condition.condition_id,
                        sensors=sensors,
                        repeat_index=repeat_index,
                        action=action,
                        reason_code=reason,
                    )
                )
            if candidates:
                return tuple(candidates)
        return ()

    first_coverage_complete = all(
        _valid_slot_count(
            tasks,
            condition_id=condition.condition_id,
            sensor=sensor,
        )
        >= 1
        for condition in required_conditions
        for sensor in sensors
    )
    optional_requirements = _optional_probe_requirements(protocol)
    paired_optional_mode = (
        protocol.evidence_policy.optional_probe_evidence_mode
        == "paired_condition_contrast"
    )
    paired_started: set[SensorKind] = set()
    paired_completed: set[SensorKind] = set()
    if paired_optional_mode:
        paired_started, paired_completed, _paired_discriminated = _paired_optional_probe_state(
            protocol,
            tasks,
            evidence,
        )
        if first_coverage_complete:
            for requirement in optional_requirements:
                if (
                    requirement.sensor not in paired_started
                    or requirement.sensor in paired_completed
                ):
                    continue
                missing_condition = next(
                    condition
                    for condition in required_conditions
                    if _valid_slot_count(
                        tasks,
                        condition_id=condition.condition_id,
                        sensor=requirement.sensor,
                    )
                    < 1
                )
                return (
                    _condition_candidate(
                        protocol,
                        tasks,
                        condition_id=missing_condition.condition_id,
                        sensors=(requirement.sensor,),
                        repeat_index=1,
                        action="probe_optional_sensor",
                        reason_code="optional_sensor_probe",
                    ),
                )
    optional_sensors = {item.sensor for item in optional_requirements}
    optional_conditions = _optional_control_conditions(protocol)
    optional_condition_ids = {item.condition_id for item in optional_conditions}
    activation_outcomes = {
        condition.condition_id: _optional_activation_outcome(
            protocol,
            tasks,
            evidence,
            target_condition_id=condition.condition_id,
        )
        for condition in optional_conditions
    }
    triggered_optional_conditions = tuple(
        condition
        for condition in optional_conditions
        if activation_outcomes[condition.condition_id] is True
    )
    activation_rules_evaluated_clear = bool(protocol.optional_activation_rules) and all(
        activation_outcomes.get(rule.target_condition_id) is False
        for rule in protocol.optional_activation_rules
    )
    adaptive_window_closed = any(
        task.repeat_index >= 2
        and task.condition_id in {item.condition_id for item in required_conditions}
        and any(sensor in sensors for sensor in task.sensors)
        for task in tasks
    )
    if paired_optional_mode:
        # A registered competition is closed only by its audited evidence gate.
        # Advancing the primary repeats must not silently waive discriminator work.
        optional_sensor_resolved = _hypothesis_termination_audit(
            protocol,
            tasks,
            evidence,
        ).gate_satisfied
    else:
        optional_sensor_resolved = adaptive_window_closed or any(
            task.action == "probe_optional_sensor"
            or any(sensor in optional_sensors for sensor in task.sensors)
            for task in tasks
        )
    available_optional_requirements = (
        tuple(item for item in optional_requirements if item.sensor not in paired_started)
        if paired_optional_mode
        else optional_requirements
    )
    optional_condition_resolved = (
        adaptive_window_closed
        or any(
            task.action == "probe_optional_condition" or task.condition_id in optional_condition_ids
            for task in tasks
        )
        or activation_rules_evaluated_clear
    )
    sensor_choice_available = (
        bool(available_optional_requirements)
        and protocol.evidence_policy.max_optional_probe_count > 0
        and not optional_sensor_resolved
    )
    condition_choice_available = (
        bool(
            triggered_optional_conditions
            if protocol.optional_activation_rules
            else optional_conditions
        )
        and protocol.evidence_policy.max_optional_condition_count > 0
        and not optional_condition_resolved
    )
    if (
        first_coverage_complete
        and condition_choice_available
        and protocol.optional_activation_rules
    ):
        first_required = next(item for item in protocol.sensors if item.sensor in sensors)
        return tuple(
            _condition_candidate(
                protocol,
                tasks,
                condition_id=condition.condition_id,
                sensors=(first_required.sensor,),
                repeat_index=1,
                action="probe_optional_condition",
                reason_code="optional_condition_probe",
            )
            for condition in triggered_optional_conditions
        )
    if first_coverage_complete and (sensor_choice_available or condition_choice_available):
        required_candidates: list[GeneralDesignCandidate] = []
        first_required = next(item for item in protocol.sensors if item.sensor in sensors)
        for condition in required_conditions:
            if (
                _valid_slot_count(
                    tasks,
                    condition_id=condition.condition_id,
                    sensor=first_required.sensor,
                )
                >= 2
            ):
                continue
            required_candidates.append(
                _condition_candidate(
                    protocol,
                    tasks,
                    condition_id=condition.condition_id,
                    sensors=(first_required.sensor,),
                    repeat_index=2,
                    action="replicate_condition",
                    reason_code="replication_required",
                )
            )
        probe_condition = (
            required_conditions[0] if paired_optional_mode else required_conditions[1]
        )
        remaining_slots = max(0, 8 - len(required_candidates))
        optional_candidates: list[GeneralDesignCandidate] = []
        if sensor_choice_available:
            reserved_condition_slots = (
                min(len(optional_conditions), remaining_slots) if condition_choice_available else 0
            )
            sensor_slots = max(0, remaining_slots - reserved_condition_slots)
            optional_candidates.extend(
                _condition_candidate(
                    protocol,
                    tasks,
                    condition_id=probe_condition.condition_id,
                    sensors=(requirement.sensor,),
                    repeat_index=1,
                    action="probe_optional_sensor",
                    reason_code="optional_sensor_probe",
                )
                for requirement in available_optional_requirements[:sensor_slots]
            )
        remaining_slots = max(
            0,
            8 - len(required_candidates) - len(optional_candidates),
        )
        if condition_choice_available:
            available_conditions = (
                triggered_optional_conditions
                if protocol.optional_activation_rules
                else optional_conditions
            )
            optional_candidates.extend(
                _condition_candidate(
                    protocol,
                    tasks,
                    condition_id=condition.condition_id,
                    sensors=(first_required.sensor,),
                    repeat_index=1,
                    action="probe_optional_condition",
                    reason_code="optional_condition_probe",
                )
                for condition in available_conditions[:remaining_slots]
            )
        return (*required_candidates, *optional_candidates)

    for repeat_index in range(
        1,
        protocol.evidence_policy.required_repeats_per_condition + 1,
    ):
        for requirement in protocol.sensors:
            if requirement.sensor == "bluetooth" or requirement.activation != "required":
                continue
            candidates = []
            for condition_index, condition in enumerate(required_conditions):
                if (
                    _valid_slot_count(
                        tasks,
                        condition_id=condition.condition_id,
                        sensor=requirement.sensor,
                    )
                    >= repeat_index
                ):
                    continue
                if repeat_index > 1:
                    action = "replicate_condition"
                    reason = "replication_required"
                elif requirement.role == "primary":
                    action = "collect_condition"
                    reason = "initial_baseline" if condition_index == 0 else "missing_condition"
                else:
                    action = "collect_supporting_sensor"
                    reason = "missing_supporting_sensor"
                candidates.append(
                    _condition_candidate(
                        protocol,
                        tasks,
                        condition_id=condition.condition_id,
                        sensors=(requirement.sensor,),
                        repeat_index=repeat_index,
                        action=action,
                        reason_code=reason,
                    )
                )
            if candidates:
                return tuple(candidates)
    return ()


def build_reasoning_continuation_candidates(
    prepared: PreparedGeneralTransition,
) -> tuple[GeneralDesignCandidate, ...]:
    """Build a bounded extension set after the preregistered evidence window.

    These candidates do not authorize new sensors, conditions, or interventions.
    They only let the evidence reasoner request another repeat from the frozen
    protocol when the first conclusion window is still ambiguous.
    """

    prepared = PreparedGeneralTransition.model_validate(prepared.model_dump(mode="python"))
    protocol = prepared.base_case.protocol
    tasks = (*prepared.base_case.completed_tasks, prepared.completed_task)
    required_conditions = _required_conditions(protocol)
    required_sensors = tuple(
        requirement.sensor
        for requirement in protocol.sensors
        if requirement.sensor != "bluetooth" and requirement.activation == "required"
    )
    if not required_sensors:
        return ()
    sensor_groups: tuple[tuple[SensorKind, ...], ...]
    if protocol.alignment == "simultaneous":
        sensor_groups = (required_sensors,)
    else:
        sensor_groups = tuple((sensor,) for sensor in required_sensors)

    candidates: list[GeneralDesignCandidate] = []
    for sensors in sensor_groups:
        for condition in required_conditions:
            next_repeat = min(
                _valid_slot_count(
                    tasks,
                    condition_id=condition.condition_id,
                    sensor=sensor,
                )
                for sensor in sensors
            ) + 1
            if next_repeat > 32:
                continue
            candidates.append(
                _condition_candidate(
                    protocol,
                    tasks,
                    condition_id=condition.condition_id,
                    sensors=sensors,
                    repeat_index=next_repeat,
                    action="replicate_condition",
                    reason_code="replication_required",
                )
            )
    return tuple(candidates[:8])


def _task_from_candidate(
    candidate: GeneralDesignCandidate,
    *,
    sequence: int,
) -> GeneralExperimentTask:
    digest = _canonical_sha256(
        {
            "candidate_id": candidate.candidate_id,
            "sequence": sequence,
            "input_evidence_ids": candidate.input_evidence_ids,
        }
    )
    return GeneralExperimentTask(
        task_id=f"task-{sequence}-{digest[:12]}",
        sequence=sequence,
        action=candidate.action,
        condition_id=candidate.condition_id,
        sensors=candidate.sensors,
        repeat_index=candidate.repeat_index,
        title=candidate.title,
        instruction=candidate.instruction,
        reason_code=candidate.reason_code,
        input_evidence_ids=candidate.input_evidence_ids,
    )


def _coverage_vector(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
    *,
    correction_count: int,
    reason_code: Literal[
        "continue",
        "evidence-complete",
        "adaptive-evidence-sufficient",
        "correction-budget-exhausted",
        "measurement-budget-exhausted",
    ],
    extra_blocker_codes: tuple[str, ...] = (),
) -> GeneralTerminationVector:
    sensors = _numeric_sensors(protocol)
    conditions = _required_conditions(protocol)
    required_condition_ids = {item.condition_id for item in conditions}
    valid_count = sum(len(task.output_evidence_ids) for task in tasks if task.measurement_valid)
    required_valid_count = sum(
        sum(sensor in sensors for sensor in task.sensors)
        for task in tasks
        if task.measurement_valid and task.condition_id in required_condition_ids
    )
    invalid_count = sum(
        len(task.output_evidence_ids) for task in tasks if task.measurement_valid is False
    )
    covered_conditions = sum(
        all(
            _valid_slot_count(tasks, condition_id=condition.condition_id, sensor=sensor) >= 1
            for sensor in sensors
        )
        for condition in conditions
    )
    covered_sensors = sum(
        any(
            _valid_slot_count(tasks, condition_id=condition.condition_id, sensor=sensor) >= 1
            for condition in conditions
        )
        for sensor in sensors
    )
    blockers: list[str] = []
    if reason_code == "continue":
        for condition in conditions:
            for sensor in sensors:
                count = _valid_slot_count(
                    tasks,
                    condition_id=condition.condition_id,
                    sensor=sensor,
                )
                if count < protocol.evidence_policy.required_repeats_per_condition:
                    blockers.append(f"missing-{condition.condition_id}-{sensor}-repeat-{count + 1}")
        blockers.extend(extra_blocker_codes)
    adaptive_sufficiency = _adaptive_sufficiency_assessment(
        protocol,
        tasks,
        evidence,
        correction_count=correction_count,
    )
    hypothesis_termination = _hypothesis_termination_audit(protocol, tasks, evidence)
    hypothesis_conclusion = _hypothesis_conclusion_audit(
        protocol,
        tasks,
        evidence,
        hypothesis_termination=hypothesis_termination,
    )
    if reason_code == "continue" and not hypothesis_termination.gate_satisfied:
        blockers.append("competition-discriminator-evidence-required")
        blockers.extend(
            f"missing-hypothesis-discriminator-{observation_id}"
            for observation_id in hypothesis_termination.unresolved_discriminator_ids
        )
    complete = reason_code in {
        "evidence-complete",
        "adaptive-evidence-sufficient",
    }
    forced = reason_code in {
        "correction-budget-exhausted",
        "measurement-budget-exhausted",
    }
    if complete:
        blockers.append("evidence-reasoning-required")
    return GeneralTerminationVector(
        required_evidence_count=protocol.evidence_policy.required_recording_count,
        valid_evidence_count=valid_count,
        invalid_evidence_count=invalid_count,
        condition_coverage_ratio=covered_conditions / len(conditions),
        sensor_coverage_ratio=covered_sensors / len(sensors),
        repeat_coverage_ratio=min(
            1.0,
            required_valid_count / protocol.evidence_policy.required_recording_count,
        ),
        correction_count=correction_count,
        completion_basis=(
            "registered-three-repeats"
            if reason_code == "evidence-complete"
            else "adaptive-two-repeat-sufficiency"
            if reason_code == "adaptive-evidence-sufficient"
            else "none"
        ),
        adaptive_sufficiency=adaptive_sufficiency,
        hypothesis_termination=hypothesis_termination,
        hypothesis_conclusion=hypothesis_conclusion,
        evidence_complete=complete,
        reasoning_required=complete,
        guidance_ready=False,
        conclusion_ready=False,
        forced_stop=forced,
        reason_code=reason_code,
        blocker_codes=tuple(blockers[:12]),
    )


def _adaptive_sufficiency_assessment(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
    *,
    correction_count: int,
) -> GeneralAdaptiveSufficiencyAssessment:
    policy = protocol.evidence_policy.adaptive_sufficiency
    conditions = _required_conditions(protocol)
    sensors = _numeric_sensors(protocol)
    minimum = policy.minimum_repeats_per_slot
    slot_counts = {
        (condition.condition_id, sensor): _valid_slot_count(
            tasks,
            condition_id=condition.condition_id,
            sensor=sensor,
        )
        for condition in conditions
        for sensor in sensors
    }
    coverage_met = all(count >= minimum for count in slot_counts.values())
    decision_window_open = coverage_met and all(count == minimum for count in slot_counts.values())
    blockers: list[str] = []
    if not policy.enabled:
        blockers.append("policy-disabled")
    if not coverage_met:
        blockers.append("minimum-repeat-coverage-missing")
    elif not decision_window_open:
        blockers.append("adaptive-decision-window-closed")
    correction_free = correction_count == 0
    if policy.enabled and policy.require_no_corrections and not correction_free:
        blockers.append("correction-history-present")
    if not coverage_met:
        return GeneralAdaptiveSufficiencyAssessment(
            policy_enabled=policy.enabled,
            minimum_coverage_met=False,
            decision_window_open=False,
            all_evidence_high_quality=False,
            correction_free=correction_free,
            eligible=False,
            blocker_codes=tuple(dict.fromkeys(blockers)),
        )

    evidence_by_id = {item.evidence_id: item for item in evidence}
    values_by_slot: dict[tuple[str, SensorKind], tuple[float, ...]] = {}
    qualities: list[str] = []
    for condition in conditions:
        for sensor in sensors:
            slot_evidence = tuple(
                evidence_by_id[evidence_id]
                for task in tasks
                if task.measurement_valid
                and task.condition_id == condition.condition_id
                and sensor in task.sensors
                for evidence_id in task.output_evidence_ids
                if evidence_by_id[evidence_id].sensor == sensor
            )
            values_by_slot[(condition.condition_id, sensor)] = tuple(
                item.metric.value for item in slot_evidence
            )
            qualities.extend(item.quality for item in slot_evidence)

    all_high_quality = bool(qualities) and all(item == "high" for item in qualities)
    if policy.enabled and policy.require_all_high_quality and not all_high_quality:
        blockers.append("required-evidence-not-high-quality")

    within_ranges: list[float] = []
    relative_contrasts: list[float] = []
    contrast_to_uncertainty: list[float] = []
    reference_id = conditions[0].condition_id
    for values in values_by_slot.values():
        center = float(median(values))
        scale = max(abs(center), 1e-12)
        within_ranges.append((max(values) - min(values)) / scale)
    for sensor in sensors:
        reference_values = values_by_slot[(reference_id, sensor)]
        reference_center = float(median(reference_values))
        reference_half_range = (max(reference_values) - min(reference_values)) / 2
        for condition in conditions[1:]:
            comparison_values = values_by_slot[(condition.condition_id, sensor)]
            comparison_center = float(median(comparison_values))
            comparison_half_range = (max(comparison_values) - min(comparison_values)) / 2
            delta = abs(comparison_center - reference_center)
            scale = max(abs(reference_center), abs(comparison_center), 1e-12)
            uncertainty = max(
                reference_half_range + comparison_half_range,
                policy.uncertainty_floor_relative * scale,
            )
            relative_contrasts.append(delta / scale)
            contrast_to_uncertainty.append(delta / uncertainty)

    maximum_within = max(within_ranges)
    minimum_relative = min(relative_contrasts)
    minimum_signal_to_uncertainty = min(contrast_to_uncertainty)
    if maximum_within > policy.maximum_within_slot_relative_range:
        blockers.append("within-slot-variation-too-high")
    if minimum_relative < policy.minimum_relative_contrast:
        blockers.append("relative-contrast-too-small")
    if minimum_signal_to_uncertainty < policy.minimum_contrast_to_uncertainty_ratio:
        blockers.append("contrast-not-above-uncertainty")
    blocker_codes = tuple(dict.fromkeys(blockers))
    return GeneralAdaptiveSufficiencyAssessment(
        policy_enabled=policy.enabled,
        minimum_coverage_met=True,
        decision_window_open=decision_window_open,
        all_evidence_high_quality=all_high_quality,
        correction_free=correction_free,
        observed_max_within_slot_relative_range=maximum_within,
        observed_min_relative_contrast=minimum_relative,
        observed_min_contrast_to_uncertainty_ratio=minimum_signal_to_uncertainty,
        eligible=policy.enabled and not blocker_codes,
        blocker_codes=blocker_codes,
    )


def _median_absolute_deviation(values: tuple[float, ...]) -> float:
    center = median(values)
    return float(median(tuple(abs(value - center) for value in values)))


def _report_boundaries(protocol: GeneralExperimentProtocol) -> tuple[str, ...]:
    source_boundaries = (
        (
            "本次仅为确定性模拟排练；所有序列均为合成 analyzer-contract 数据，不能作为现实世界、手机或公开数据证据。",
            "模拟排练的 Gate C、Phyphox Validated、Agent Ready 与市场验证均为 false。",
        )
        if set(protocol.selected_sources) == {"protocol_emulator"}
        else ()
    )
    values = (
        *protocol.claim_boundaries,
        "仅报告本协议内的描述性相对变化。",
        "公开回放不计用户真机 Gate C。",
        *source_boundaries,
    )
    return tuple(dict.fromkeys(values))[:16]


def _hypothesis_assessments(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
) -> tuple[GeneralHypothesisAssessment, ...]:
    assessments: list[GeneralHypothesisAssessment] = []
    for hypothesis in protocol.hypotheses:
        observations: list[GeneralHypothesisObservationAssessment] = []
        for observation in hypothesis.observations:
            contrast = _condition_contrast_relation(
                protocol,
                tasks,
                evidence,
                sensor=observation.sensor,
                metric_key=observation.metric_key,
                metric_unit=observation.metric_unit,
                reference_condition_id=observation.reference_condition_id,
                comparison_condition_id=observation.comparison_condition_id,
            )
            observed_relation = contrast[0] if contrast is not None else None
            match_code: Literal[
                "not_observed",
                "matches_expected",
                "conflicts_expected",
            ] = (
                "not_observed"
                if contrast is None
                else "matches_expected"
                if _relation_matches_expected(
                    observation.expected_relation,
                    observed_relation,
                )
                else "conflicts_expected"
            )
            observations.append(
                GeneralHypothesisObservationAssessment(
                    observation_id=observation.observation_id,
                    sensor=observation.sensor,
                    metric_key=observation.metric_key,
                    metric_unit=observation.metric_unit,
                    expected_relation=observation.expected_relation,
                    observed_relation=observed_relation,
                    match_code=match_code,
                    source_evidence_ids=() if contrast is None else contrast[1],
                )
            )
        states = {item.match_code for item in observations}
        assessment_code: Literal[
            "untested",
            "observed_prediction_matched",
            "observed_prediction_conflicted",
            "mixed_observations",
        ] = (
            "mixed_observations"
            if {"matches_expected", "conflicts_expected"} <= states
            else "observed_prediction_matched"
            if "matches_expected" in states
            else "observed_prediction_conflicted"
            if "conflicts_expected" in states
            else "untested"
        )
        assessments.append(
            GeneralHypothesisAssessment(
                hypothesis_id=hypothesis.hypothesis_id,
                statement_untrusted=hypothesis.statement_untrusted,
                assessment_code=assessment_code,
                observations=tuple(observations),
            )
        )
    return tuple(assessments)


def _hypothesis_conclusion_audit(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
    *,
    hypothesis_termination: GeneralHypothesisTerminationAudit | None = None,
) -> GeneralHypothesisConclusionAudit:
    hypothesis_ids = tuple(item.hypothesis_id for item in protocol.hypotheses)
    if not hypothesis_ids:
        return GeneralHypothesisConclusionAudit()
    assessments = _hypothesis_assessments(protocol, tasks, evidence)
    compatible = tuple(
        item.hypothesis_id
        for item in assessments
        if item.assessment_code == "observed_prediction_matched"
    )
    weakened = tuple(
        item.hypothesis_id
        for item in assessments
        if item.assessment_code == "observed_prediction_conflicted"
    )
    mixed = tuple(
        item.hypothesis_id
        for item in assessments
        if item.assessment_code == "mixed_observations"
    )
    untested = tuple(
        item.hypothesis_id for item in assessments if item.assessment_code == "untested"
    )
    source_evidence_ids = tuple(
        dict.fromkeys(
            evidence_id
            for assessment in assessments
            for observation in assessment.observations
            for evidence_id in observation.source_evidence_ids
        )
    )
    termination = hypothesis_termination or _hypothesis_termination_audit(
        protocol,
        tasks,
        evidence,
    )
    if not termination.gate_satisfied:
        return GeneralHypothesisConclusionAudit(
            registered_hypothesis_ids=hypothesis_ids,
            compatible_hypothesis_ids=compatible,
            weakened_hypothesis_ids=weakened,
            mixed_hypothesis_ids=mixed,
            untested_hypothesis_ids=untested,
            conclusion_available=False,
            conclusion_code="pending-discriminator-evidence",
            source_evidence_ids=source_evidence_ids,
        )
    uniquely_favored = (
        len(compatible) == 1
        and len(weakened) == len(hypothesis_ids) - 1
        and not mixed
        and not untested
    )
    return GeneralHypothesisConclusionAudit(
        registered_hypothesis_ids=hypothesis_ids,
        compatible_hypothesis_ids=compatible,
        weakened_hypothesis_ids=weakened,
        mixed_hypothesis_ids=mixed,
        untested_hypothesis_ids=untested,
        conclusion_available=True,
        conclusion_code=(
            "one-hypothesis-favored"
            if uniquely_favored
            else "no-unique-hypothesis-favored"
        ),
        favored_hypothesis_id=compatible[0] if uniquely_favored else None,
        source_evidence_ids=source_evidence_ids,
    )


def _descriptive_report(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
    *,
    completion_basis: Literal[
        "registered-three-repeats",
        "adaptive-two-repeat-sufficiency",
    ],
) -> GeneralExperimentReport:
    hypothesis_termination = _hypothesis_termination_audit(protocol, tasks, evidence)
    if not hypothesis_termination.gate_satisfied:
        raise GeneralExperimentStateError(
            "descriptive reports cannot retain unresolved competing hypotheses"
        )
    evidence_by_id = {item.evidence_id: item for item in evidence}
    valid_ids = tuple(
        evidence_id
        for task in tasks
        if task.measurement_valid
        for evidence_id in task.output_evidence_ids
    )
    excluded_ids = tuple(
        evidence_id
        for task in tasks
        if task.measurement_valid is False
        for evidence_id in task.output_evidence_ids
    )
    summaries: list[GeneralConditionMetricSummary] = []
    required_conditions = _required_conditions(protocol)
    for condition in required_conditions:
        for requirement in protocol.sensors:
            if requirement.sensor == "bluetooth" or requirement.activation != "required":
                continue
            values = tuple(
                evidence_by_id[evidence_id].metric.value
                for task in tasks
                if task.measurement_valid
                and task.condition_id == condition.condition_id
                and requirement.sensor in task.sensors
                for evidence_id in task.output_evidence_ids
                if evidence_by_id[evidence_id].sensor == requirement.sensor
            )
            minimum_repeats = (
                2
                if completion_basis == "adaptive-two-repeat-sufficiency"
                else protocol.evidence_policy.required_repeats_per_condition
            )
            if len(values) < minimum_repeats:
                raise GeneralExperimentStateError(
                    "descriptive report repeat count is below its completion basis"
                )
            summaries.append(
                GeneralConditionMetricSummary(
                    condition_id=condition.condition_id,
                    sensor=requirement.sensor,
                    metric_key=requirement.metric_key or "missing_metric",
                    unit=requirement.metric_unit or "missing_unit",
                    values=values,
                    median=float(median(values)),
                    median_absolute_deviation=_median_absolute_deviation(values),
                )
            )

    summary_by_slot = {(item.condition_id, item.sensor): item for item in summaries}
    reference_condition = required_conditions[0]
    contrasts: list[GeneralMetricContrast] = []
    for requirement in protocol.sensors:
        if requirement.sensor == "bluetooth" or requirement.activation != "required":
            continue
        reference = summary_by_slot[(reference_condition.condition_id, requirement.sensor)]
        for condition in required_conditions[1:]:
            comparison = summary_by_slot[(condition.condition_id, requirement.sensor)]
            delta = comparison.median - reference.median
            threshold = max(
                reference.median_absolute_deviation + comparison.median_absolute_deviation,
                0.05 * max(abs(reference.median), abs(comparison.median), 1e-12),
            )
            direction: Literal["increase", "decrease", "within_observed_repeatability"]
            if delta > threshold:
                direction = "increase"
            elif delta < -threshold:
                direction = "decrease"
            else:
                direction = "within_observed_repeatability"
            contrasts.append(
                GeneralMetricContrast(
                    sensor=requirement.sensor,
                    metric_key=requirement.metric_key or "missing_metric",
                    unit=requirement.metric_unit or "missing_unit",
                    reference_condition_id=reference_condition.condition_id,
                    comparison_condition_id=condition.condition_id,
                    absolute_delta=delta,
                    relative_delta_ratio=(
                        delta / reference.median if abs(reference.median) > 1e-12 else None
                    ),
                    descriptive_threshold=threshold,
                    direction=direction,
                )
            )
    condition_labels = {item.condition_id: item.label for item in protocol.conditions}
    phrases = [
        (
            f"{_SENSOR_DISPLAY_NAMES.get(item.sensor, item.sensor)}在“"
            f"{condition_labels.get(item.comparison_condition_id, item.comparison_condition_id)}”"
            f"相对“{condition_labels.get(item.reference_condition_id, item.reference_condition_id)}”"
            f"{_DIRECTION_DISPLAY_NAMES[item.direction]}"
        )
        for item in contrasts[:6]
    ]
    optional_sensors = {
        item.sensor for item in protocol.sensors if item.activation == "optional_probe"
    }
    optional_condition_ids = {
        item.condition_id for item in protocol.conditions if item.activation == "optional_control"
    }
    auxiliary_observations = tuple(
        GeneralAuxiliaryObservation(
            condition_id=task.condition_id,
            sensor=evidence_by_id[evidence_id].sensor,
            metric_key=evidence_by_id[evidence_id].metric.key,
            value=evidence_by_id[evidence_id].metric.value,
            unit=evidence_by_id[evidence_id].metric.unit,
            quality=evidence_by_id[evidence_id].quality,
            interpretation=(
                "single_optional_condition_probe_not_registered_comparison"
                if task.condition_id in optional_condition_ids
                else "paired_optional_probe_descriptive_contrast"
                if (
                    protocol.evidence_policy.optional_probe_evidence_mode
                    == "paired_condition_contrast"
                )
                else "single_optional_probe_not_a_condition_comparison"
            ),
        )
        for task in tasks
        if task.measurement_valid
        for evidence_id in task.output_evidence_ids
        if (
            evidence_by_id[evidence_id].sensor in optional_sensors
            or task.condition_id in optional_condition_ids
        )
    )
    hypothesis_assessments = _hypothesis_assessments(protocol, tasks, evidence)
    hypothesis_conclusion = _hypothesis_conclusion_audit(
        protocol,
        tasks,
        evidence,
        hypothesis_termination=hypothesis_termination,
    )
    metric_labels = {
        (item.sensor, item.metric.key): item.metric.label
        for item in evidence
        if item.evidence_id in valid_ids
    }
    visualization = GeneralVisualizationArtifact(
        artifact_id=f"general-comparison-{protocol.draft_sha256[:16]}",
        title="各传感器的条件对比",
        independent_variable=protocol.independent_variable,
        series=tuple(
            GeneralComparisonSeries(
                sensor=requirement.sensor,
                metric_key=requirement.metric_key or "missing_metric",
                metric_label=metric_labels[
                    (requirement.sensor, requirement.metric_key or "missing_metric")
                ],
                unit=requirement.metric_unit or "missing_unit",
                points=tuple(
                    GeneralVisualizationPoint(
                        condition_id=summary.condition_id,
                        condition_label=condition_labels.get(
                            summary.condition_id, summary.condition_id
                        ),
                        median=summary.median,
                        median_absolute_deviation=summary.median_absolute_deviation,
                        repeat_count=len(summary.values),
                    )
                    for summary in summaries
                    if summary.sensor == requirement.sensor
                ),
            )
            for requirement in protocol.sensors
            if requirement.sensor != "bluetooth" and requirement.activation == "required"
        ),
        source_evidence_ids=valid_ids,
        warnings=(
            "每个传感器使用独立纵轴；不同单位或不同物理量的高度不能直接互相比较。",
            "误差线表示重复记录中位绝对偏差（MAD），不是测量仪器的校准不确定度。",
            "Location 只展示派生相对几何指标，Microphone 只展示派生相对级别。",
        ),
    )
    rehearsal = set(protocol.selected_sources) == {"protocol_emulator"}
    evidence_noun = "条模拟分析证据" if rehearsal else "条有效物理证据"
    answer_parts = [
        (
            f"已完成 {len(valid_ids)} {evidence_noun}；服务端充分度门允许以每槽位两次高质量重复结束。"
            if completion_basis == "adaptive-two-repeat-sufficiency"
            else f"已完成 {len(valid_ids)} {evidence_noun}的三次重复比较。"
        )
    ]
    if phrases:
        answer_parts.append(f"结果显示：{'；'.join(phrases)}。")
    if auxiliary_observations:
        answer_parts.append(
            f"另保留 {len(auxiliary_observations)} 条单次辅助传感器观察，它们不构成条件比较。"
        )
    if hypothesis_assessments:
        assessments_by_id = {
            item.hypothesis_id: item for item in hypothesis_assessments
        }
        if hypothesis_conclusion.conclusion_code == "one-hypothesis-favored":
            favored_id = hypothesis_conclusion.favored_hypothesis_id
            favored = assessments_by_id[favored_id] if favored_id is not None else None
            weakened_labels = "、".join(
                f"“{assessments_by_id[item].statement_untrusted}”"
                for item in hypothesis_conclusion.weakened_hypothesis_ids
            )
            answer_parts.append(
                "竞争假设结论：在本协议预注册预测范围内，"
                f"当前证据更符合 {favored_id}“{favored.statement_untrusted if favored else ''}”；"
                f"相较之下，{weakened_labels}的预测与观测冲突。"
                "这是有边界的相对证据倾向，不等于证明该原因或因果关系。"
            )
        else:
            compatible_count = len(hypothesis_conclusion.compatible_hypothesis_ids)
            if compatible_count > 1:
                detail = f"{compatible_count} 个假设都与当前观测相容"
            elif hypothesis_conclusion.mixed_hypothesis_ids:
                detail = "部分假设内部同时出现匹配与冲突观测"
            elif compatible_count == 0:
                detail = "没有一个预注册假设完整符合当前观测"
            else:
                detail = "仍有假设未被当前判别量稳定排除"
            answer_parts.append(
                f"竞争假设结论：本次判别测量未能区分唯一解释；{detail}。"
                "这是一个明确的非判别结果，不会被包装成因果结论。"
            )
        if hypothesis_termination.disposition == "remaining-discriminators-exempted":
            answer_parts.append(
                "其余预注册判别观察未被静默跳过：服务端已记录豁免收据，"
                f"因为共享判别量已区分全部竞争假设；豁免 {len(hypothesis_termination.waived_discriminator_ids)} 项。"
            )
        else:
            answer_parts.append("竞争假设终止门已完成全部预注册判别观察。")
    answer_parts.append(
        "这些结果只证明模拟排练的软件闭环，不代表现实世界测量。"
        if rehearsal
        else "这些结果是当前控制条件下的描述性反馈，不自动证明因果或绝对校准。"
    )
    answer = "".join(answer_parts)
    confidence: Literal["medium", "high"] = (
        "high"
        if completion_basis == "registered-three-repeats"
        and all(evidence_by_id[item].quality == "high" for item in valid_ids)
        else "medium"
    )
    return GeneralExperimentReport(
        outcome="completed_descriptive",
        answer=answer,
        confidence=confidence,
        evidence_scope="simulated_rehearsal" if rehearsal else "physical_recordings",
        completion_basis=completion_basis,
        termination_reason=(
            "所有必需槽位均取得两次高质量证据，且条件效应通过预注册的波动与不确定性门。"
            if completion_basis == "adaptive-two-repeat-sufficiency"
            else "所有条件与传感器均取得三次通过质量门的独立证据。"
        ),
        summaries=tuple(summaries),
        contrasts=tuple(contrasts),
        auxiliary_observations=auxiliary_observations,
        hypothesis_assessments=hypothesis_assessments,
        hypothesis_termination=hypothesis_termination,
        hypothesis_conclusion=hypothesis_conclusion,
        visualizations=(visualization,),
        evidence_ids=valid_ids,
        excluded_evidence_ids=excluded_ids,
        claim_boundaries=_report_boundaries(protocol),
    )


def _inconclusive_report(
    protocol: GeneralExperimentProtocol,
    tasks: tuple[GeneralExperimentTask, ...],
    evidence: tuple[GeneralEvidenceEnvelope, ...],
    *,
    reason_code: str,
    hypothesis_termination: GeneralHypothesisTerminationAudit,
) -> GeneralExperimentReport:
    valid_ids = tuple(
        evidence_id
        for task in tasks
        if task.measurement_valid
        for evidence_id in task.output_evidence_ids
    )
    excluded_ids = tuple(
        evidence_id
        for task in tasks
        if task.measurement_valid is False
        for evidence_id in task.output_evidence_ids
    )
    reason = (
        "低质量或未对齐记录已经用完纠偏轮次。"
        if reason_code == "correction-budget-exhausted"
        else "测量记录已经达到本协议的安全上限。"
    )
    rehearsal = set(protocol.selected_sources) == {"protocol_emulator"}
    hypothesis_conclusion = _hypothesis_conclusion_audit(
        protocol,
        tasks,
        evidence,
        hypothesis_termination=hypothesis_termination,
    )
    hypothesis_assessments = _hypothesis_assessments(protocol, tasks, evidence)
    return GeneralExperimentReport(
        outcome="completed_inconclusive",
        answer=f"本次实验没有形成足够完整的证据，不能给出比较结论。{reason}",
        confidence="low",
        evidence_scope="simulated_rehearsal" if rehearsal else "physical_recordings",
        termination_reason=reason,
        hypothesis_assessments=hypothesis_assessments,
        hypothesis_termination=hypothesis_termination,
        hypothesis_conclusion=hypothesis_conclusion,
        evidence_ids=valid_ids,
        excluded_evidence_ids=excluded_ids,
        claim_boundaries=_report_boundaries(protocol),
    )


def create_general_experiment_case(
    compilation: GeneralExplorationCompilation,
    *,
    case_id: str,
    compiler_provenance: GeneralCompilerProvenance | None = None,
) -> GeneralExperimentCase:
    compilation = _validated_compilation(compilation)
    if compilation.status != "executable" or compilation.protocol is None:
        raise GeneralExperimentStateError("only executable compilations can create a case")
    protocol = compilation.protocol
    if not any(
        source in {"phyphox_live", "phone_upload", "public_replay", "protocol_emulator"}
        for source in protocol.selected_sources
    ):
        raise GeneralExperimentStateError("an authorized evidence or rehearsal source is required")
    candidates = _next_candidates(protocol, ())
    if not candidates:
        raise GeneralExperimentStateError("protocol produced no safe initial candidate")
    selected = candidates[0]
    return GeneralExperimentCase(
        case_id=case_id,
        revision=1,
        status="collecting",
        compiler_provenance=compiler_provenance or GeneralCompilerProvenance(),
        protocol=protocol,
        current_task=_task_from_candidate(selected, sequence=1),
        decision_trace=(
            GeneralDesignDecisionTrace(
                revision=1,
                candidate_ids=tuple(item.candidate_id for item in candidates),
                selected_candidate_id=selected.candidate_id,
                source="server_initial",
                reason_code=selected.reason_code,
                input_evidence_ids=selected.input_evidence_ids,
            ),
        ),
        correction_count=0,
        termination=_coverage_vector(
            protocol,
            (),
            (),
            correction_count=0,
            reason_code="continue",
        ),
    )


def _expected_evidence_id(
    protocol: GeneralExperimentProtocol,
    evidence: GeneralEvidenceEnvelope,
) -> str:
    digest = _canonical_sha256(
        {
            "protocol_id": protocol.protocol_id,
            "draft_sha256": protocol.draft_sha256,
            "condition_id": evidence.condition_id,
            "sensor": evidence.sensor,
            "content_sha256": evidence.lineage.content_sha256,
            "metric_key": evidence.metric.key,
            "metric_unit": evidence.metric.unit,
        }
    )
    return f"general-evidence-{digest[:16]}"


def _validate_submission_evidence(
    case: GeneralExperimentCase,
    submission: GeneralMeasurementSubmission,
) -> tuple[GeneralEvidenceEnvelope, ...]:
    current = case.current_task
    if current is None:
        raise GeneralExperimentStateError("terminal cases cannot accept measurements")
    by_sensor = {item.sensor: item for item in submission.evidence}
    if set(by_sensor) != set(current.sensors):
        raise GeneralExperimentStateError("submission sensors must exactly match the current task")
    if set(by_sensor).intersection({"bluetooth"}):
        raise GeneralExperimentStateError("Bluetooth cannot submit numeric evidence")
    ordered = tuple(by_sensor[sensor] for sensor in current.sensors)
    existing_ids = {item.evidence_id for item in case.evidence}
    for item in ordered:
        requirement = next(
            (value for value in case.protocol.sensors if value.sensor == item.sensor),
            None,
        )
        if requirement is None:
            raise GeneralExperimentStateError("evidence sensor is outside the protocol")
        if (
            item.protocol_id != case.protocol.protocol_id
            or item.protocol_draft_sha256 != case.protocol.draft_sha256
            or item.condition_id != current.condition_id
            or item.role != requirement.role
            or item.analysis.analyzer_id != requirement.analyzer_id
            or item.metric.key != requirement.metric_key
            or item.metric.unit != requirement.metric_unit
            or item.lineage.source not in case.protocol.selected_sources
        ):
            raise GeneralExperimentStateError("evidence does not match the current protocol task")
        if item.evidence_id != _expected_evidence_id(case.protocol, item):
            raise GeneralExperimentStateError("evidence ID does not match its immutable content")
        if item.evidence_id in existing_ids:
            raise GeneralExperimentStateError("evidence replay is not allowed")
    return ordered


def _prepared_digest(
    *,
    base_case: GeneralExperimentCase,
    completed_task: GeneralExperimentTask,
    submitted_evidence: tuple[GeneralEvidenceEnvelope, ...],
    next_candidates: tuple[GeneralDesignCandidate, ...],
    fallback_candidate_id: str | None,
    termination: GeneralTerminationVector,
    report: GeneralExperimentReport | None,
    correction_count: int,
) -> str:
    values = {
        "schema_version": "1.0",
        "base_case": base_case.model_dump(mode="json"),
        "completed_task": completed_task.model_dump(mode="json"),
        "submitted_evidence": [item.model_dump(mode="json") for item in submitted_evidence],
        "next_candidates": [item.model_dump(mode="json") for item in next_candidates],
        "fallback_candidate_id": fallback_candidate_id,
        "termination": termination.model_dump(mode="json"),
        "report": None if report is None else report.model_dump(mode="json"),
        "correction_count": correction_count,
    }
    return _canonical_sha256(values)


def prepare_reasoning_continuation(
    prepared: PreparedGeneralTransition,
) -> PreparedGeneralTransition:
    """Replace a provisional terminal template with a bounded extra-measurement set."""

    prepared = PreparedGeneralTransition.model_validate(prepared.model_dump(mode="python"))
    if prepared.report is None or not prepared.termination.reasoning_required:
        raise GeneralExperimentStateError(
            "reasoning continuation requires a provisionally complete evidence window"
        )
    candidates = build_reasoning_continuation_candidates(prepared)
    if not candidates:
        raise GeneralExperimentStateError("no frozen continuation candidate is available")
    case = prepared.base_case
    tasks = (*case.completed_tasks, prepared.completed_task)
    evidence = (*case.evidence, *prepared.submitted_evidence)
    termination = _coverage_vector(
        case.protocol,
        tasks,
        evidence,
        correction_count=prepared.correction_count,
        reason_code="continue",
        extra_blocker_codes=("reasoning-agent-requested-more-evidence",),
    )
    fallback_candidate_id = candidates[0].candidate_id
    digest = _prepared_digest(
        base_case=case,
        completed_task=prepared.completed_task,
        submitted_evidence=prepared.submitted_evidence,
        next_candidates=candidates,
        fallback_candidate_id=fallback_candidate_id,
        termination=termination,
        report=None,
        correction_count=prepared.correction_count,
    )
    return PreparedGeneralTransition(
        base_case=case,
        completed_task=prepared.completed_task,
        submitted_evidence=prepared.submitted_evidence,
        next_candidates=candidates,
        fallback_candidate_id=fallback_candidate_id,
        termination=termination,
        report=None,
        correction_count=prepared.correction_count,
        prepared_sha256=digest,
    )


def prepare_reasoned_report(
    prepared: PreparedGeneralTransition,
    report: GeneralExperimentReport,
) -> PreparedGeneralTransition:
    """Bind a validated evidence-reasoning receipt into a prepared terminal report."""

    prepared = PreparedGeneralTransition.model_validate(prepared.model_dump(mode="python"))
    report = GeneralExperimentReport.model_validate(report.model_dump(mode="python"))
    if prepared.report is None or not prepared.termination.reasoning_required:
        raise GeneralExperimentStateError("reasoned reports require a provisional terminal report")
    if report.reasoning is None or report.reasoning.decision not in {"finalize", "user_stop"}:
        raise GeneralExperimentStateError("reasoned reports require a final reasoning receipt")
    if (
        report.reasoning.case_id != prepared.base_case.case_id
        or report.reasoning.expected_revision != prepared.base_case.revision
    ):
        raise GeneralExperimentStateError("reasoned report receipt does not bind the active case")
    termination = GeneralTerminationVector.model_validate(
        prepared.termination.model_copy(
            update={
                "reasoning_required": False,
                "guidance_ready": True,
                "conclusion_ready": report.reasoning.decision == "finalize",
                "blocker_codes": (),
            }
        ).model_dump(mode="python")
    )
    digest = _prepared_digest(
        base_case=prepared.base_case,
        completed_task=prepared.completed_task,
        submitted_evidence=prepared.submitted_evidence,
        next_candidates=(),
        fallback_candidate_id=None,
        termination=termination,
        report=report,
        correction_count=prepared.correction_count,
    )
    return PreparedGeneralTransition(
        base_case=prepared.base_case,
        completed_task=prepared.completed_task,
        submitted_evidence=prepared.submitted_evidence,
        next_candidates=(),
        fallback_candidate_id=None,
        termination=termination,
        report=report,
        correction_count=prepared.correction_count,
        prepared_sha256=digest,
    )


def commit_general_reasoning_checkpoint(
    prepared: PreparedGeneralTransition,
    receipt: GeneralReasoningReceipt,
) -> GeneralExperimentCase:
    """Commit completed evidence while pausing the loop for an explicit user choice."""

    prepared = PreparedGeneralTransition.model_validate(prepared.model_dump(mode="python"))
    expected_sha = _prepared_digest(
        base_case=prepared.base_case,
        completed_task=prepared.completed_task,
        submitted_evidence=prepared.submitted_evidence,
        next_candidates=prepared.next_candidates,
        fallback_candidate_id=prepared.fallback_candidate_id,
        termination=prepared.termination,
        report=prepared.report,
        correction_count=prepared.correction_count,
    )
    if prepared.prepared_sha256 != expected_sha or prepared.report is None:
        raise GeneralExperimentStateError("checkpoint requires an intact provisional report")
    if (
        receipt.decision != "continue"
        or receipt.case_id != prepared.base_case.case_id
        or receipt.expected_revision != prepared.base_case.revision
        or receipt.selected_candidate_id is None
    ):
        raise GeneralExperimentStateError("checkpoint requires a bound continuation receipt")
    case = prepared.base_case
    tasks = (*case.completed_tasks, prepared.completed_task)
    evidence = (*case.evidence, *prepared.submitted_evidence)
    task_count = len(tasks)
    continue_allowed = task_count < case.protocol.evidence_policy.hard_task_count
    candidates = build_reasoning_continuation_candidates(prepared) if continue_allowed else ()
    candidate_ids = {item.candidate_id for item in candidates}
    if continue_allowed and receipt.selected_candidate_id not in candidate_ids:
        raise GeneralExperimentStateError("checkpoint recommendation left the frozen candidate set")
    offered = GeneralReasoningReceipt.model_validate(
        receipt.model_copy(
            update={
                "decision": "offer_user_choice",
                "selected_candidate_id": None,
            }
        ).model_dump(mode="python")
    )
    termination = _coverage_vector(
        case.protocol,
        tasks,
        evidence,
        correction_count=prepared.correction_count,
        reason_code="continue",
        extra_blocker_codes=("user-decision-required",),
    )
    checkpoint = GeneralReasoningCheckpoint(
        checkpoint_id=f"checkpoint-{case.revision + 1}-{receipt.request_sha256[:12]}",
        triggered_at_task_count=task_count,
        continue_allowed=continue_allowed,
        recommended_candidate_id=(
            receipt.selected_candidate_id if continue_allowed else None
        ),
        continuation_candidates=candidates,
        reasoning=offered,
        provisional_report=prepared.report,
        prompt=(
            "证据仍未达到清晰终止标准。你可以继续执行 Agent 推荐的判别测量，"
            "也可以依据当前证据收手并生成较低确定性的报告。"
            if continue_allowed
            else "实验已达到安全任务上限；请依据当前证据收手并生成有边界报告。"
        ),
    )
    return GeneralExperimentCase(
        case_id=case.case_id,
        revision=case.revision + 1,
        status="awaiting_user_decision",
        compiler_provenance=case.compiler_provenance,
        protocol=case.protocol,
        current_task=None,
        completed_tasks=tasks,
        evidence=evidence,
        decision_trace=case.decision_trace,
        planner_trace=case.planner_trace,
        reasoning_trace=(*case.reasoning_trace, offered),
        reasoning_checkpoint_count=case.reasoning_checkpoint_count + 1,
        reasoning_checkpoint=checkpoint,
        correction_count=prepared.correction_count,
        termination=termination,
    )


def continue_general_reasoning_checkpoint(
    case: GeneralExperimentCase,
    decision: GeneralReasoningCheckpointDecision,
) -> GeneralExperimentCase:
    case = GeneralExperimentCase.model_validate(case.model_dump(mode="python"))
    decision = GeneralReasoningCheckpointDecision.model_validate(
        decision.model_dump(mode="python")
    )
    checkpoint = case.reasoning_checkpoint
    if (
        case.status != "awaiting_user_decision"
        or checkpoint is None
        or decision.action != "continue"
        or decision.expected_revision != case.revision
        or not checkpoint.continue_allowed
        or checkpoint.recommended_candidate_id is None
    ):
        raise GeneralExperimentStateError("checkpoint continuation is stale or unavailable")
    chosen = next(
        item
        for item in checkpoint.continuation_candidates
        if item.candidate_id == checkpoint.recommended_candidate_id
    )
    termination = _coverage_vector(
        case.protocol,
        case.completed_tasks,
        case.evidence,
        correction_count=case.correction_count,
        reason_code="continue",
        extra_blocker_codes=("reasoning-agent-requested-more-evidence",),
    )
    trace = GeneralDesignDecisionTrace(
        revision=case.revision + 1,
        candidate_ids=tuple(
            item.candidate_id for item in checkpoint.continuation_candidates
        ),
        selected_candidate_id=chosen.candidate_id,
        source="user_checkpoint",
        reason_code=chosen.reason_code,
        input_evidence_ids=chosen.input_evidence_ids,
    )
    return GeneralExperimentCase(
        case_id=case.case_id,
        revision=case.revision + 1,
        status="collecting",
        compiler_provenance=case.compiler_provenance,
        protocol=case.protocol,
        current_task=_task_from_candidate(
            chosen,
            sequence=len(case.completed_tasks) + 1,
        ),
        completed_tasks=case.completed_tasks,
        evidence=case.evidence,
        decision_trace=(*case.decision_trace, trace),
        planner_trace=case.planner_trace,
        reasoning_trace=case.reasoning_trace,
        reasoning_checkpoint_count=case.reasoning_checkpoint_count,
        correction_count=case.correction_count,
        termination=termination,
    )


def complete_general_reasoning_checkpoint(
    case: GeneralExperimentCase,
    decision: GeneralReasoningCheckpointDecision,
    report: GeneralExperimentReport,
) -> GeneralExperimentCase:
    case = GeneralExperimentCase.model_validate(case.model_dump(mode="python"))
    checkpoint = case.reasoning_checkpoint
    if (
        case.status != "awaiting_user_decision"
        or checkpoint is None
        or decision.action != "stop"
        or decision.expected_revision != case.revision
    ):
        raise GeneralExperimentStateError("checkpoint stop decision is stale or unavailable")
    report = GeneralExperimentReport.model_validate(report.model_dump(mode="python"))
    if report.reasoning is None or report.reasoning.decision != "user_stop":
        raise GeneralExperimentStateError("checkpoint stop requires a user-stop report receipt")
    termination = _coverage_vector(
        case.protocol,
        case.completed_tasks,
        case.evidence,
        correction_count=case.correction_count,
        reason_code="evidence-complete",
    )
    termination = GeneralTerminationVector.model_validate(
        termination.model_copy(
            update={
                "reasoning_required": False,
                "guidance_ready": True,
                "conclusion_ready": False,
                "blocker_codes": (),
            }
        ).model_dump(mode="python")
    )
    if report.completion_basis != termination.completion_basis:
        raise GeneralExperimentStateError("checkpoint report completion basis is inconsistent")
    return GeneralExperimentCase(
        case_id=case.case_id,
        revision=case.revision + 1,
        status=report.outcome,
        compiler_provenance=case.compiler_provenance,
        protocol=case.protocol,
        current_task=None,
        completed_tasks=case.completed_tasks,
        evidence=case.evidence,
        decision_trace=case.decision_trace,
        planner_trace=case.planner_trace,
        reasoning_trace=(
            case.reasoning_trace
            if report.reasoning in case.reasoning_trace
            else (*case.reasoning_trace, report.reasoning)
        ),
        reasoning_checkpoint_count=case.reasoning_checkpoint_count,
        correction_count=case.correction_count,
        termination=termination,
        report=report,
    )


def prepare_general_measurement(
    case: GeneralExperimentCase,
    submission: GeneralMeasurementSubmission,
) -> PreparedGeneralTransition:
    case = _validated_case(case)
    submission = GeneralMeasurementSubmission.model_validate(submission.model_dump(mode="python"))
    current = case.current_task
    if case.status != "collecting" or current is None:
        raise GeneralExperimentStateError("terminal cases cannot accept measurements")
    if (
        submission.case_id != case.case_id
        or submission.task_id != current.task_id
        or submission.expected_revision != case.revision
    ):
        raise GeneralExperimentStateError("stale or foreign measurement submission")
    submitted_evidence = _validate_submission_evidence(case, submission)
    policy = case.protocol.evidence_policy
    effective_evidence_budget = min(
        256,
        max(
            policy.max_measurements,
            policy.hard_task_count * max(1, len(_numeric_sensors(case.protocol))),
        ),
    )
    if len(case.evidence) + len(submitted_evidence) > effective_evidence_budget:
        raise GeneralExperimentStateError("measurement submission exceeds the frozen budget")

    reasons: tuple[str, ...]
    if case.protocol.alignment == "simultaneous" and len(current.sensors) > 1:
        group = build_condition_evidence_group(
            case.protocol,
            condition_id=current.condition_id,
            evidence=list(submitted_evidence),
        )
        measurement_valid = group.valid
        reasons = group.blocker_codes
    else:
        measurement_valid = all(item.valid for item in submitted_evidence)
        reasons = tuple(
            dict.fromkeys(
                reason for item in submitted_evidence for reason in item.rejection_reasons
            )
        )
    completed_task = current.model_copy(
        update={
            "status": "completed",
            "output_evidence_ids": tuple(item.evidence_id for item in submitted_evidence),
            "measurement_valid": measurement_valid,
            "rejection_reasons": reasons,
        }
    )
    completed_task = GeneralExperimentTask.model_validate(completed_task.model_dump(mode="python"))
    tasks = (*case.completed_tasks, completed_task)
    all_evidence = (*case.evidence, *submitted_evidence)
    correction_count = case.correction_count + (not measurement_valid)
    required_complete = all(
        _valid_slot_count(
            tasks,
            condition_id=condition.condition_id,
            sensor=sensor,
        )
        >= case.protocol.evidence_policy.required_repeats_per_condition
        for condition in _required_conditions(case.protocol)
        for sensor in _numeric_sensors(case.protocol)
    )
    adaptive_sufficiency = _adaptive_sufficiency_assessment(
        case.protocol,
        tasks,
        all_evidence,
        correction_count=correction_count,
    )
    hypothesis_termination = _hypothesis_termination_audit(
        case.protocol,
        tasks,
        all_evidence,
    )
    hypothesis_conclusion = _hypothesis_conclusion_audit(
        case.protocol,
        tasks,
        all_evidence,
        hypothesis_termination=hypothesis_termination,
    )
    if (
        required_complete
        and hypothesis_termination.gate_satisfied
        and hypothesis_conclusion.conclusion_available
    ):
        reason_code = "evidence-complete"
    elif (
        adaptive_sufficiency.eligible
        and hypothesis_termination.gate_satisfied
        and hypothesis_conclusion.conclusion_available
    ):
        reason_code = "adaptive-evidence-sufficient"
    elif correction_count >= case.protocol.evidence_policy.max_corrections:
        reason_code = "correction-budget-exhausted"
    elif len(tasks) >= policy.hard_task_count or len(all_evidence) >= effective_evidence_budget:
        reason_code = "measurement-budget-exhausted"
    else:
        reason_code = "continue"
    termination = _coverage_vector(
        case.protocol,
        tasks,
        all_evidence,
        correction_count=correction_count,
        reason_code=reason_code,
    )

    report: GeneralExperimentReport | None = None
    candidates: tuple[GeneralDesignCandidate, ...] = ()
    fallback_candidate_id: str | None = None
    if reason_code in {"evidence-complete", "adaptive-evidence-sufficient"}:
        report = _descriptive_report(
            case.protocol,
            tasks,
            all_evidence,
            completion_basis=(
                "adaptive-two-repeat-sufficiency"
                if reason_code == "adaptive-evidence-sufficient"
                else "registered-three-repeats"
            ),
        )
    elif reason_code != "continue":
        report = _inconclusive_report(
            case.protocol,
            tasks,
            all_evidence,
            reason_code=reason_code,
            hypothesis_termination=hypothesis_termination,
        )
    elif not measurement_valid:
        candidate = _condition_candidate(
            case.protocol,
            tasks,
            condition_id=current.condition_id,
            sensors=current.sensors,
            repeat_index=current.repeat_index,
            action="correct_condition",
            reason_code="quality_correction",
            input_evidence_ids=completed_task.output_evidence_ids,
        )
        candidates = (candidate,)
        fallback_candidate_id = candidate.candidate_id
    else:
        candidates = _next_candidates(case.protocol, tasks, all_evidence)
        if not candidates:
            raise GeneralExperimentStateError(
                "incomplete evidence graph produced no safe next candidate"
            )
        fallback_candidate_id = candidates[0].candidate_id

    prepared_sha256 = _prepared_digest(
        base_case=case,
        completed_task=completed_task,
        submitted_evidence=submitted_evidence,
        next_candidates=candidates,
        fallback_candidate_id=fallback_candidate_id,
        termination=termination,
        report=report,
        correction_count=correction_count,
    )
    return PreparedGeneralTransition(
        base_case=case,
        completed_task=completed_task,
        submitted_evidence=submitted_evidence,
        next_candidates=candidates,
        fallback_candidate_id=fallback_candidate_id,
        termination=termination,
        report=report,
        correction_count=correction_count,
        prepared_sha256=prepared_sha256,
    )


def commit_general_measurement(
    prepared: PreparedGeneralTransition,
    *,
    selected_candidate_id: str | None = None,
    selection_source: SelectionSource = "deterministic_fallback",
    planner_audit: GeneralPlannerDecisionAudit | None = None,
    reasoning_receipt: GeneralReasoningReceipt | None = None,
) -> GeneralExperimentCase:
    prepared = PreparedGeneralTransition.model_validate(prepared.model_dump(mode="python"))
    expected_sha = _prepared_digest(
        base_case=prepared.base_case,
        completed_task=prepared.completed_task,
        submitted_evidence=prepared.submitted_evidence,
        next_candidates=prepared.next_candidates,
        fallback_candidate_id=prepared.fallback_candidate_id,
        termination=prepared.termination,
        report=prepared.report,
        correction_count=prepared.correction_count,
    )
    if prepared.prepared_sha256 != expected_sha:
        raise GeneralExperimentStateError("prepared transition integrity check failed")
    case = prepared.base_case
    tasks = (*case.completed_tasks, prepared.completed_task)
    evidence = (*case.evidence, *prepared.submitted_evidence)
    revision = case.revision + 1
    if prepared.report is not None:
        if selected_candidate_id is not None or planner_audit is not None:
            raise GeneralExperimentStateError(
                "terminal transitions cannot retain a planner decision"
            )
        if reasoning_receipt is not None and prepared.report.reasoning != reasoning_receipt:
            raise GeneralExperimentStateError(
                "terminal reasoning receipt must be embedded in the report"
            )
        return GeneralExperimentCase(
            case_id=case.case_id,
            revision=revision,
            status=prepared.report.outcome,
            compiler_provenance=case.compiler_provenance,
            protocol=case.protocol,
            current_task=None,
            completed_tasks=tasks,
            evidence=evidence,
            decision_trace=case.decision_trace,
            planner_trace=case.planner_trace,
            reasoning_trace=(
                case.reasoning_trace
                if prepared.report.reasoning is None
                else (*case.reasoning_trace, prepared.report.reasoning)
            ),
            reasoning_checkpoint_count=case.reasoning_checkpoint_count,
            correction_count=prepared.correction_count,
            termination=prepared.termination,
            report=prepared.report,
        )

    chosen_id = selected_candidate_id or prepared.fallback_candidate_id
    chosen = next(
        (item for item in prepared.next_candidates if item.candidate_id == chosen_id),
        None,
    )
    if chosen is None:
        raise GeneralExperimentStateError("selected candidate is outside the frozen set")
    if selection_source == "bounded_agent":
        if (
            planner_audit is None
            or planner_audit.outcome != "accepted"
            or planner_audit.source != "agent"
        ):
            raise GeneralExperimentStateError(
                "bounded Agent selections require an accepted planner audit"
            )
    elif selection_source == "reasoning_agent":
        if (
            planner_audit is not None
            or reasoning_receipt is None
            or reasoning_receipt.decision != "continue"
            or reasoning_receipt.case_id != case.case_id
            or reasoning_receipt.expected_revision != case.revision
            or reasoning_receipt.selected_candidate_id != chosen.candidate_id
        ):
            raise GeneralExperimentStateError(
                "reasoning Agent selections require a bound continuation receipt"
            )
    elif planner_audit is not None:
        deterministic_audit = (
            selection_source == "deterministic_policy"
            and planner_audit.outcome == "deterministic"
            and planner_audit.source == "deterministic_policy"
        )
        fallback_audit = (
            selection_source == "deterministic_fallback"
            and planner_audit.outcome == "fallback"
            and planner_audit.source == "deterministic_fallback"
        )
        if not deterministic_audit and not fallback_audit:
            raise GeneralExperimentStateError("planner audit does not match selection source")
    if selection_source == "deterministic_fallback" and (
        chosen.candidate_id != prepared.fallback_candidate_id
    ):
        raise GeneralExperimentStateError("deterministic fallback must use the frozen fallback")
    if planner_audit is not None and (
        planner_audit.expected_revision != case.revision
        or planner_audit.commit_revision != revision
        or planner_audit.completed_task_id != prepared.completed_task.task_id
        or planner_audit.prepared_sha256 != prepared.prepared_sha256
        or planner_audit.candidate_ids
        != tuple(item.candidate_id for item in prepared.next_candidates)
        or planner_audit.selected_candidate_id != chosen.candidate_id
        or planner_audit.fallback_candidate_id != prepared.fallback_candidate_id
    ):
        raise GeneralExperimentStateError("planner audit does not bind the prepared transition")
    trace = GeneralDesignDecisionTrace(
        revision=revision,
        candidate_ids=tuple(item.candidate_id for item in prepared.next_candidates),
        selected_candidate_id=chosen.candidate_id,
        source=selection_source,
        reason_code=chosen.reason_code,
        input_evidence_ids=chosen.input_evidence_ids,
    )
    return GeneralExperimentCase(
        case_id=case.case_id,
        revision=revision,
        status="collecting",
        compiler_provenance=case.compiler_provenance,
        protocol=case.protocol,
        current_task=_task_from_candidate(chosen, sequence=len(tasks) + 1),
        completed_tasks=tasks,
        evidence=evidence,
        decision_trace=(*case.decision_trace, trace),
        planner_trace=(
            case.planner_trace if planner_audit is None else (*case.planner_trace, planner_audit)
        ),
        reasoning_trace=(
            case.reasoning_trace
            if reasoning_receipt is None
            else (*case.reasoning_trace, reasoning_receipt)
        ),
        reasoning_checkpoint_count=case.reasoning_checkpoint_count,
        correction_count=prepared.correction_count,
        termination=prepared.termination,
    )


def submit_general_measurement(
    case: GeneralExperimentCase,
    submission: GeneralMeasurementSubmission,
    *,
    selected_candidate_id: str | None = None,
    selection_source: SelectionSource = "deterministic_fallback",
    planner_audit: GeneralPlannerDecisionAudit | None = None,
    reasoning_receipt: GeneralReasoningReceipt | None = None,
) -> GeneralExperimentCase:
    prepared = prepare_general_measurement(case, submission)
    return commit_general_measurement(
        prepared,
        selected_candidate_id=selected_candidate_id,
        selection_source=selection_source,
        planner_audit=planner_audit,
        reasoning_receipt=reasoning_receipt,
    )
