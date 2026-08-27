from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pocketlab.general_exploration_models import (
    GeneralAdaptiveSufficiencyPolicy,
    GeneralCompileContext,
    GeneralEvidencePolicy,
    GeneralExpectedObservation,
    GeneralExperimentProtocol,
    GeneralExplorationCompilation,
    GeneralExplorationDraft,
    GeneralHypothesisSpec,
    GeneralMetricCapability,
    GeneralOptionalActivationRule,
    GeneralSensorCapabilityContract,
    GeneralSensorRequirement,
)


def _metric(key: str, unit: str, label: str) -> GeneralMetricCapability:
    return GeneralMetricCapability(metric_key=key, unit=unit, label=label)


_CAPABILITIES = (
    GeneralSensorCapabilityContract(
        sensor="accelerometer",
        analyzer_id="pocketlab.acceleration.v2",
        metrics=(
            _metric("selected_axis_rms_m_s2", "m/s^2", "主动态轴 RMS"),
            _metric("selected_axis_peak_to_peak_m_s2", "m/s^2", "主动态轴峰峰值"),
            _metric("dominant_frequency_hz", "Hz", "主频"),
            _metric("spectral_snr_db", "dB", "频谱信噪比"),
        ),
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=(
            "频谱与振幅只能描述运动响应，不能单独识别故障原因。",
            "位移积分和楼层推断不属于通用加速度合同。",
        ),
    ),
    GeneralSensorCapabilityContract(
        sensor="gyroscope",
        analyzer_id="pocketlab.gyroscope.v1",
        metrics=(
            _metric("mean_angular_speed_rad_s", "rad/s", "平均角速度模长"),
            _metric("angular_speed_std_rad_s", "rad/s", "角速度波动"),
            _metric("peak_angular_speed_rad_s", "rad/s", "峰值角速度"),
        ),
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=("未校正零偏时不积分角位移或输出绝对姿态。",),
    ),
    GeneralSensorCapabilityContract(
        sensor="magnetometer",
        analyzer_id="pocketlab.magnetometer.v1",
        metrics=(
            _metric("mean_field_magnitude_ut", "uT", "平均磁场模长"),
            _metric("field_magnitude_std_ut", "uT", "磁场模长波动"),
            _metric("field_peak_to_peak_ut", "uT", "磁场模长峰峰值"),
            _metric("max_field_deviation_ut", "uT", "相对背景最大偏差"),
        ),
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=("只能报告相对局部磁场变化，不能识别具体物体或证明因果。",),
    ),
    GeneralSensorCapabilityContract(
        sensor="light",
        analyzer_id="pocketlab.light.v2",
        metrics=(
            _metric("median_illuminance_lx", "lx", "照度中位数"),
            _metric("illuminance_iqr_lx", "lx", "照度四分位距"),
            _metric("coefficient_of_variation_ratio", "ratio", "照度变异系数"),
            _metric("upper_plateau_fraction", "ratio", "观测上限平台比例"),
        ),
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=("没有外部参考照度计时不声称绝对校准准确度。",),
    ),
    GeneralSensorCapabilityContract(
        sensor="pressure",
        analyzer_id="pocketlab.pressure.v2",
        metrics=(
            _metric("pressure_change_hpa", "hPa", "压力变化"),
            _metric("relative_height_change_m", "m", "近似相对高度变化"),
            _metric("pressure_trend_hpa_per_min", "hPa/min", "压力趋势"),
            _metric("pressure_mad_hpa", "hPa", "压力中位绝对偏差"),
        ),
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=("标准大气近似不等于绝对海拔，且不能单独排除天气和 HVAC。",),
    ),
    GeneralSensorCapabilityContract(
        sensor="proximity",
        analyzer_id="pocketlab.proximity.v2",
        metrics=(
            _metric("observed_level_count", "count", "观测离散级数"),
            _metric("signal_mode_code", "code", "常量、二态或多级模式"),
            _metric("transition_count", "count", "状态转换次数"),
        ),
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=("二态 near/far 编码不能解释为连续厘米距离。",),
    ),
    GeneralSensorCapabilityContract(
        sensor="microphone",
        analyzer_id="pocketlab.microphone.derived.v1",
        metrics=(
            _metric("mean_relative_level_db", "dB_relative", "平均相对级别"),
            _metric("peak_relative_level_db", "dB_relative", "峰值相对级别"),
            _metric("relative_level_span_db", "dB_relative", "相对级别范围"),
        ),
        privacy_ack_required=True,
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=(
            "只接受派生相对级别，不保存原始音频或转写。",
            "dB_relative 不是校准声压级。",
        ),
    ),
    GeneralSensorCapabilityContract(
        sensor="location",
        analyzer_id="pocketlab.location.haversine.v1",
        metrics=(
            _metric("trajectory_distance_m", "m", "相对轨迹长度"),
            _metric("displacement_m", "m", "端点相对位移"),
            _metric("average_path_speed_m_s", "m/s", "平均路径速率"),
            _metric("path_efficiency_ratio", "ratio", "路径效率"),
        ),
        privacy_ack_required=True,
        supports_live_capture=True,
        supports_file_upload=True,
        supports_public_replay=True,
        supports_bounded_agent=True,
        limitations=("默认只输出相对几何，不在通用报告中展示绝对坐标。",),
    ),
    GeneralSensorCapabilityContract(
        sensor="bluetooth",
        analyzer_id=None,
        metrics=(),
        supports_live_capture=False,
        supports_file_upload=False,
        supports_public_replay=False,
        supports_bounded_agent=False,
        limitations=("必须先登记具体设备、GATT、字节序、缩放、单位和时钟协议。",),
    ),
)

_CAPABILITY_BY_SENSOR = {item.sensor: item for item in _CAPABILITIES}
if len(_CAPABILITY_BY_SENSOR) != len(_CAPABILITIES):
    raise RuntimeError("general exploration sensor capabilities must be unique")

_OPTIONAL_ACTIVATION_THRESHOLDS: dict[tuple[str, str, str], float] = {
    ("accelerometer", "selected_axis_rms_m_s2", "m/s^2"): 1.0,
    ("gyroscope", "mean_angular_speed_rad_s", "rad/s"): 0.3,
    ("magnetometer", "mean_field_magnitude_ut", "uT"): 80.0,
    ("light", "median_illuminance_lx", "lx"): 50.0,
    ("pressure", "pressure_mad_hpa", "hPa"): 0.015,
    ("proximity", "transition_count", "count"): 8.0,
    ("microphone", "mean_relative_level_db", "dB_relative"): 55.0,
    ("location", "trajectory_distance_m", "m"): 30.0,
}

_REJECTED_CLAIMS = {
    "medical_diagnosis": (
        "medical-claim-outside-scope",
        "手机传感器实验不能用于医疗诊断。",
    ),
    "person_identification": (
        "identity-claim-outside-scope",
        "PocketLab 不执行人员识别。",
    ),
    "surveillance": (
        "surveillance-claim-outside-scope",
        "PocketLab 不把传感器实验用于人员监控。",
    ),
    "dangerous_operation": (
        "dangerous-operation-rejected",
        "该实验要求危险操作，不能生成可执行协议。",
    ),
}


def list_general_sensor_capabilities() -> list[GeneralSensorCapabilityContract]:
    return [item.model_copy(deep=True) for item in _CAPABILITIES]


def normalize_general_exploration_draft_for_protocol(
    draft: GeneralExplorationDraft,
) -> GeneralExplorationDraft:
    """Translate a safe research intent into the strongest executable claim.

    PocketLab can compare controlled conditions and rank mechanisms, but the
    phone-only protocol cannot promise an absolute causal proof.  Treating every
    causal wording as a hard blocker created an impossible compiler loop: the
    model had to label the intent honestly, while the protocol then rejected the
    same label.  Preserve the question and hypotheses, but freeze the executable
    claim at a relative comparison with an explicit boundary.

    Invalid simultaneous layouts are likewise a transport detail, not a reason
    to ask the user to restate an otherwise complete experiment.  Sequential
    capture is the safe lossless representation when there are not two required
    numeric sensors or when optional probes/conditions are present.
    """

    draft = GeneralExplorationDraft.model_validate(draft.model_dump(mode="python"))
    payload = draft.model_dump(mode="python")
    changed = False
    boundaries = list(draft.claim_boundaries)

    if draft.requested_claim == "causal":
        payload["requested_claim"] = "relative_comparison"
        causal_boundary = (
            "本协议检验受控条件下哪种机制更受支持，不把手机测量包装成绝对因果证明。"
        )
        if causal_boundary not in boundaries:
            boundaries.append(causal_boundary)
        changed = True

    if draft.alignment == "simultaneous":
        required_numeric = {
            item.sensor
            for item in draft.sensor_intents
            if item.sensor != "bluetooth" and item.activation == "required"
        }
        has_optional = any(
            item.activation == "optional_probe" for item in draft.sensor_intents
        ) or any(item.activation == "optional_control" for item in draft.conditions)
        if len(required_numeric) < 2 or has_optional:
            payload["alignment"] = "sequential"
            changed = True

    if not changed:
        return draft
    payload["claim_boundaries"] = tuple(dict.fromkeys(boundaries))[:12]
    return GeneralExplorationDraft.model_validate(payload)


def _canonical_sha256(draft: GeneralExplorationDraft) -> str:
    payload = json.dumps(
        draft.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def general_exploration_draft_sha256(draft: GeneralExplorationDraft) -> str:
    draft = GeneralExplorationDraft.model_validate(draft.model_dump(mode="python"))
    return _canonical_sha256(draft)


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _visualizations(draft: GeneralExplorationDraft) -> tuple[str, ...]:
    by_objective: dict[str, tuple[str, ...]] = {
        "compare_conditions": ("comparison", "time_series"),
        "characterize_trend": ("time_series", "scatter"),
        "detect_event": ("timeline", "time_series"),
        "estimate_relationship": ("scatter", "comparison"),
        "check_repeatability": ("comparison", "time_series"),
        "map_relative_pattern": ("relative_map", "comparison"),
        "combine_signals": ("time_series", "comparison"),
    }
    values = list(by_objective[draft.objective])
    if any(item.sensor == "location" for item in draft.sensor_intents):
        values.insert(0, "relative_map")
    return _unique(values)[:4]


def _rejected(code: str, message: str) -> GeneralExplorationCompilation:
    return GeneralExplorationCompilation(
        status="rejected",
        protocol=None,
        blocker_codes=(code,),
        user_messages=(message,),
        can_run_with_current_context=False,
    )


def _validated_public_match(
    draft: GeneralExplorationDraft,
    context: GeneralCompileContext,
    draft_sha256: str,
) -> tuple[str | None, str | None]:
    match = context.public_replay_match
    if match is None:
        return None, "public-replay-semantic-match-required"
    selected_sensors = {item.sensor for item in draft.sensor_intents if item.sensor != "bluetooth"}
    if (
        match.draft_sha256 != draft_sha256
        or match.objective != draft.objective
        or set(match.sensors) != selected_sensors
    ):
        return None, "public-replay-match-does-not-bind-draft"
    return match.match_id, None


def compile_general_exploration_protocol(
    draft: GeneralExplorationDraft,
    context: GeneralCompileContext,
) -> GeneralExplorationCompilation:
    """Compile a strict draft into a server-owned protocol without model execution.

    This function is deterministic and mutation-free. A protocol-emulator-only draft
    is an executable simulated rehearsal, never physical evidence. Public replay is
    never authorized by sensor name alone; it must bind to the exact draft digest.
    """

    draft = GeneralExplorationDraft.model_validate(draft.model_dump(mode="python"))
    context = GeneralCompileContext.model_validate(context.model_dump(mode="python"))
    draft_sha256 = _canonical_sha256(draft)

    rejected_claim = _REJECTED_CLAIMS.get(draft.requested_claim)
    if rejected_claim is not None:
        return _rejected(*rejected_claim)

    requirements: list[GeneralSensorRequirement] = []
    for intent in draft.sensor_intents:
        capability = _CAPABILITY_BY_SENSOR[intent.sensor]
        if intent.sensor != "bluetooth":
            metric = next(
                (
                    item
                    for item in capability.metrics
                    if item.metric_key == intent.metric_key and item.unit == intent.metric_unit
                ),
                None,
            )
            if metric is None:
                return _rejected(
                    "unsupported-sensor-metric",
                    f"{intent.sensor} 当前不支持所请求的指标与单位组合。",
                )
        requirements.append(
            GeneralSensorRequirement(
                sensor=intent.sensor,
                role=intent.role,
                activation=intent.activation,
                analyzer_id=capability.analyzer_id,
                metric_key=intent.metric_key,
                metric_unit=intent.metric_unit,
                measurement_purpose=intent.measurement_purpose,
                privacy_ack_required=capability.privacy_ack_required,
                bounded_agent_supported=capability.supports_bounded_agent,
            )
        )

    blockers: list[str] = []
    messages: list[str] = []
    sensors = {item.sensor for item in requirements}
    numeric_sensors = sensors - {"bluetooth"}
    sensitive_sensors = {item.sensor for item in requirements if item.privacy_ack_required}
    acknowledged = set(context.privacy_acknowledged_sensors)

    if "bluetooth" in sensors:
        blockers.append("bluetooth-capability-check-only")
        messages.append("Bluetooth 目前只能检查能力，不能进入通用数值实验闭环。")
    if draft.alignment == "simultaneous" and len(numeric_sensors) < 2:
        blockers.append("simultaneous-capture-requires-multiple-sensors")
        messages.append("同步采集至少需要两个必需数值传感器；单传感器实验应使用顺序采集。")
    if (
        draft.alignment == "simultaneous"
        and len(numeric_sensors) > 1
        and not context.supports_simultaneous_capture
    ):
        blockers.append("simultaneous-multi-sensor-capture-not-available")
        messages.append("该问题要求同步多传感器采集，但当前数据桥尚未提供可信同步。")
    if draft.alignment == "simultaneous" and (
        any(item.activation == "optional_probe" for item in requirements)
        or any(item.activation == "optional_control" for item in draft.conditions)
    ):
        blockers.append("optional-probe-requires-sequential-alignment")
        messages.append("首版可选传感器或对照条件探测只允许顺序采集，不能混入同步采集协议。")
    if draft.requested_claim == "causal":
        blockers.append("causal-claim-requires-registered-design")
        messages.append("通用协议只能先形成相对比较；因果结论需要注册的控制与随机化设计。")
    if draft.requested_claim == "absolute_calibration" and not context.external_reference_available:
        blockers.append("absolute-calibration-requires-external-reference")
        messages.append("绝对校准需要可追溯外部参考仪器，手机单独测量不足以支持该结论。")

    privacy_ready = sensitive_sensors <= acknowledged
    if not privacy_ready:
        blockers.append("privacy-acknowledgement-required")
        messages.append("麦克风或位置实验必须在运行前完成对应的隐私确认。")

    authorized_sources: list[str] = []
    source_messages: list[str] = []
    public_match_id: str | None = None
    for source in context.selected_sources:
        if source == "protocol_emulator":
            authorized_sources.append(source)
            continue
        if not privacy_ready:
            continue
        if source == "phyphox_live":
            live_supported = all(
                _CAPABILITY_BY_SENSOR[sensor].supports_live_capture for sensor in sensors
            )
            detection_ready = numeric_sensors <= set(context.detected_sensors)
            if live_supported and (detection_ready or context.allow_deferred_live_detection):
                authorized_sources.append(source)
            else:
                source_messages.append("当前 phyphox 实验没有检测到全部必需传感器。")
        elif source == "phone_upload":
            if all(_CAPABILITY_BY_SENSOR[sensor].supports_file_upload for sensor in sensors):
                authorized_sources.append(source)
            else:
                source_messages.append("所选传感器组合当前不支持可信文件导入。")
        elif source == "public_replay":
            public_supported = all(
                _CAPABILITY_BY_SENSOR[sensor].supports_public_replay for sensor in sensors
            )
            if public_supported:
                public_match_id, public_error = _validated_public_match(
                    draft,
                    context,
                    draft_sha256,
                )
                if public_error is None:
                    authorized_sources.append(source)
                else:
                    source_messages.append("公开回放尚未通过与当前问题的精确语义匹配。")
            else:
                source_messages.append("所选传感器组合当前没有可授权的公开回放。")

    physical_sources = {
        source
        for source in authorized_sources
        if source in {"phyphox_live", "phone_upload", "public_replay"}
    }
    rehearsal_only = set(authorized_sources) == {"protocol_emulator"}
    if not physical_sources and not rehearsal_only:
        blockers.append("no-physical-evidence-source")
        messages.extend(source_messages)
        if not context.selected_sources:
            messages.append("请选择真机、手机文件上传或已验证的公开回放证据来源。")

    required_numeric_sensors = {
        item.sensor
        for item in requirements
        if item.sensor != "bluetooth" and item.activation == "required"
    }
    required_sensor_count = max(1, len(required_numeric_sensors))
    optional_sensor_count = sum(
        item.sensor != "bluetooth" and item.activation == "optional_probe" for item in requirements
    )
    required_condition_count = sum(item.activation == "required" for item in draft.conditions)
    optional_condition_count = sum(
        item.activation == "optional_control" for item in draft.conditions
    )
    optional_activation_rules: tuple[GeneralOptionalActivationRule, ...] = ()
    if context.enable_server_owned_optional_activation and optional_condition_count:
        optional_requirements = [
            item for item in requirements if item.activation == "optional_probe"
        ]
        optional_conditions = [
            item for item in draft.conditions if item.activation == "optional_control"
        ]
        if len(optional_requirements) != 1 or len(optional_conditions) != 1:
            blockers.append("optional-activation-requires-one-probe-and-one-control")
            messages.append(
                "带可选对照的服务端证据门槛要求一个可选传感器精确对应一个可选对照；"
                "多个竞争解释应先只注册可选传感器探查。"
            )
        else:
            requirement = optional_requirements[0]
            threshold = _OPTIONAL_ACTIVATION_THRESHOLDS.get(
                (
                    requirement.sensor,
                    str(requirement.metric_key),
                    str(requirement.metric_unit),
                )
            )
            if threshold is None:
                blockers.append("optional-activation-metric-not-registered")
                messages.append("该可选指标尚无服务端注册的证据门槛，不能交给模型自行解释。")
            else:
                condition = optional_conditions[0]
                optional_activation_rules = (
                    GeneralOptionalActivationRule(
                        rule_id=f"activation-{requirement.sensor}-{condition.condition_id}",
                        probe_sensor=requirement.sensor,
                        metric_key=str(requirement.metric_key),
                        metric_unit=str(requirement.metric_unit),
                        threshold=threshold,
                        target_condition_id=condition.condition_id,
                    ),
                )
    required_recordings = required_condition_count * 3 * required_sensor_count
    max_corrections = min(8, max(2, required_sensor_count))
    optional_probe_evidence_mode = (
        "paired_condition_contrast" if draft.hypotheses else "single_observation"
    )
    max_optional_probes = min(2 if draft.hypotheses else 1, optional_sensor_count)
    max_optional_conditions = min(1, optional_condition_count)
    correction_recording_allowance = max_corrections * (
        required_sensor_count if draft.alignment == "simultaneous" else 1
    )
    evidence_policy = GeneralEvidencePolicy(
        required_recording_count=required_recordings,
        max_corrections=max_corrections,
        max_optional_probe_count=max_optional_probes,
        optional_probe_evidence_mode=optional_probe_evidence_mode,
        max_optional_condition_count=max_optional_conditions,
        adaptive_sufficiency=GeneralAdaptiveSufficiencyPolicy(
            enabled=context.enable_adaptive_sufficiency,
        ),
        max_measurements=(
            required_recordings
            + correction_recording_allowance
            + max_optional_probes
            * (required_condition_count if draft.hypotheses else 1)
            + max_optional_conditions
        ),
    )
    protocol = GeneralExperimentProtocol(
        protocol_id=f"general-exploration-{draft_sha256[:16]}",
        draft_sha256=draft_sha256,
        title=draft.title,
        question=draft.question,
        objective=draft.objective,
        requested_claim=draft.requested_claim,
        independent_variable=draft.independent_variable,
        conditions=draft.conditions,
        sensors=tuple(requirements),
        alignment=draft.alignment,
        controls=draft.controls,
        expected_pattern=draft.expected_pattern,
        hypotheses=tuple(
            GeneralHypothesisSpec(
                hypothesis_id=hypothesis.hypothesis_id,
                statement_untrusted=hypothesis.statement_untrusted,
                observations=tuple(
                    GeneralExpectedObservation(
                        observation_id=prediction.prediction_id,
                        sensor=prediction.sensor,
                        metric_key=prediction.metric_key,
                        metric_unit=prediction.metric_unit,
                        reference_condition_id=prediction.reference_condition_id,
                        comparison_condition_id=prediction.comparison_condition_id,
                        expected_relation=prediction.expected_relation,
                        measurement_role=prediction.measurement_role,
                    )
                    for prediction in hypothesis.predictions
                ),
            )
            for hypothesis in draft.hypotheses
        ),
        safety_notes=draft.safety_notes,
        privacy_notes=draft.privacy_notes,
        claim_boundaries=draft.claim_boundaries,
        selected_sources=tuple(authorized_sources),
        public_replay_match_id=public_match_id if "public_replay" in authorized_sources else None,
        optional_activation_rules=optional_activation_rules,
        evidence_policy=evidence_policy,
        visualization_kinds=_visualizations(draft),
    )

    blocker_codes = _unique(blockers)
    user_messages = _unique(messages)
    if blocker_codes:
        return GeneralExplorationCompilation(
            status="plan_only",
            protocol=protocol,
            blocker_codes=blocker_codes,
            user_messages=user_messages,
            can_run_with_current_context=False,
        )
    return GeneralExplorationCompilation(
        status="executable",
        protocol=protocol,
        blocker_codes=(),
        user_messages=(),
        can_run_with_current_context=True,
    )
