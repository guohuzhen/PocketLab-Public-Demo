from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator

from pocketlab.reality_feedback import RealityFeedbackRecord
from pocketlab.sensor_models import (
    AnalysisMetric,
    PhyphoxSensorProfile,
    SensorKind,
    SensorRecordingCreated,
    SensorSample,
)


class AccelerationSample(BaseModel):
    timestamp_ms: float = Field(description="Monotonic timestamp in milliseconds")
    x: float = Field(description="X-axis acceleration in m/s^2")
    y: float = Field(description="Y-axis acceleration in m/s^2")
    z: float = Field(description="Z-axis acceleration in m/s^2")


class SessionUpload(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    device: str = Field(default="HUAWEI Mate 70 Pro+", max_length=120)
    sensor: Literal["accelerometer"] = "accelerometer"
    notes: str = Field(default="", max_length=500)
    samples: list[AccelerationSample] = Field(min_length=64, max_length=60_000)

    @model_validator(mode="after")
    def timestamps_must_increase(self) -> "SessionUpload":
        timestamps = [sample.timestamp_ms for sample in self.samples]
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("timestamp_ms must be strictly increasing")
        return self


class VibrationAnalysis(BaseModel):
    sample_count: int
    duration_s: float
    sampling_rate_hz: float
    sampling_jitter_ratio: float
    max_sampling_gap_ratio: float
    nyquist_frequency_hz: float
    recommended_frequency_limit_hz: float
    selected_axis: Literal["x", "y", "z"]
    rms_acceleration_m_s2: float
    peak_to_peak_m_s2: float
    dominant_frequency_hz: float
    spectral_snr_db: float
    confidence: Literal["low", "medium", "high"]
    warnings: list[str] = Field(default_factory=list)


class SessionCreated(BaseModel):
    session_id: str
    label: str
    analysis: VibrationAnalysis
    created_at: str


class SessionHistoryItem(BaseModel):
    session_id: str
    label: str
    device: str
    notes: str
    sample_count: int
    analysis: VibrationAnalysis
    created_at: str


class SessionRecord(BaseModel):
    session_id: str
    upload: SessionUpload
    analysis: VibrationAnalysis
    created_at: str


class AgentRunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    session_ids: list[str] = Field(min_length=1, max_length=4)


class AgentRunResponse(BaseModel):
    answer: str
    model: str
    session_ids: list[str]


class EvidenceWorkbenchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    recording_ids: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def recording_ids_must_be_unique(self) -> "EvidenceWorkbenchRequest":
        if len(self.recording_ids) != len(set(self.recording_ids)):
            raise ValueError("recording_ids must be unique")
        return self


class EvidenceWorkbenchResponse(BaseModel):
    answer: str
    model: str
    recording_ids: list[str]
    sensor_kinds: list[SensorKind]


class AuthRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: SecretStr = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, min_length=1, max_length=60)
    claim_local_data: bool = True


class AuthLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: SecretStr = Field(min_length=1, max_length=128)


class AuthUser(BaseModel):
    user_id: str
    username: str
    display_name: str
    created_at: str
    updated_at: str


class AuthSessionResponse(BaseModel):
    user: AuthUser
    claimed_local_data: bool = False


class AuthStatusResponse(BaseModel):
    legacy_data_available: bool


HypothesisStatus = Literal["unverified", "supported", "weakened", "inconclusive"]
TaskStatus = Literal["pending", "completed"]
DiagnosticCaseStatus = Literal[
    "planning",
    "collecting",
    "awaiting_user_decision",
    "completed_with_conclusion",
    "completed_inconclusive",
]
TaskKind = Literal["baseline", "control", "replication", "correction", "exploration"]
ExpectedEffect = Literal["increase", "decrease", "change", "no_change", "unknown"]
EffectMetric = Literal["rms", "frequency", "either"]
TaskAnalyzerStatus = Literal["ready", "detection_only", "not_implemented"]
DiagnosticSensorRole = Literal["primary", "supporting", "optional"]


class DiagnosticCaseCreate(BaseModel):
    title: str = Field(min_length=2, max_length=80)
    problem_statement: str = Field(min_length=10, max_length=1000)
    context: str = Field(default="", max_length=1000)


class DiagnosticSensorPlanDraft(BaseModel):
    sensor: SensorKind
    role: DiagnosticSensorRole
    rationale: str = Field(min_length=5, max_length=400)
    target_metric_key: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        max_length=80,
    )


class DiagnosticSensorPlanItem(DiagnosticSensorPlanDraft):
    analyzer_status: TaskAnalyzerStatus = "ready"
    measurement_quantity: str = Field(min_length=2, max_length=160)
    recommended_phyphox_experiment: str = Field(min_length=2, max_length=240)


class DiagnosticHypothesisDraft(BaseModel):
    statement: str = Field(min_length=3, max_length=240)
    rationale: str = Field(min_length=3, max_length=500)
    critical_prediction: str = Field(
        min_length=3,
        max_length=500,
        description="One observable prediction whose test can distinguish this hypothesis.",
    )
    critical_sensor: SensorKind | None = None
    critical_expected_effect: ExpectedEffect = "unknown"


class AgentMeasurementTaskDraft(BaseModel):
    """Provider-friendly task fields; deterministic metadata is added by the backend."""

    title: str = Field(min_length=2, max_length=100)
    instruction: str = Field(min_length=5, max_length=800)
    variable_to_change: str = Field(min_length=2, max_length=200)
    controlled_variables: list[str] = Field(default_factory=list, max_length=8)
    required_sensor: SensorKind = Field(
        default="accelerometer",
        description="The phone sensor that directly measures the quantity requested by this task.",
    )
    target_metric_key: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        max_length=80,
    )


class MeasurementTaskDraft(BaseModel):
    title: str = Field(min_length=2, max_length=100)
    instruction: str = Field(min_length=5, max_length=800)
    variable_to_change: str = Field(min_length=2, max_length=200)
    controlled_variables: list[str] = Field(default_factory=list, max_length=8)
    required_sensor: SensorKind = "accelerometer"
    target_metric_key: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        max_length=80,
    )
    task_kind: TaskKind
    comparison_task_id: str | None = Field(
        default=None,
        max_length=40,
        description="For a control or replication task, the completed task used as baseline.",
    )
    target_hypothesis_ids: list[str] = Field(default_factory=list, max_length=3)
    expected_effect: ExpectedEffect = "unknown"
    effect_metric: EffectMetric = "either"

    @model_validator(mode="after")
    def comparison_task_is_required_for_control(self) -> "MeasurementTaskDraft":
        if self.task_kind in {"control", "replication"} and not self.comparison_task_id:
            raise ValueError("对照或重复验证任务必须指定 comparison_task_id。")
        if self.task_kind in {"control", "replication"} and self.expected_effect == "unknown":
            raise ValueError("对照或重复验证任务必须声明 expected_effect。")
        return self


class HypothesisAssessmentDraft(BaseModel):
    hypothesis_id: str = Field(min_length=1, max_length=40)
    status: HypothesisStatus
    reasoning: str = Field(min_length=3, max_length=500)
    critical_prediction_tested: bool = Field(
        description="Whether this measurement actually tested the hypothesis's critical prediction."
    )


class DiagnosticHypothesis(BaseModel):
    hypothesis_id: str
    statement: str
    rationale: str
    critical_prediction: str
    critical_sensor: SensorKind | None = None
    critical_expected_effect: ExpectedEffect = "unknown"
    status: HypothesisStatus = "unverified"
    latest_reasoning: str = "尚未绑定测量证据。"
    evidence_ids: list[str] = Field(default_factory=list)


class DiagnosticMeasurementTask(BaseModel):
    task_id: str
    title: str
    instruction: str
    variable_to_change: str
    controlled_variables: list[str] = Field(default_factory=list)
    required_sensor: SensorKind = "accelerometer"
    target_metric_key: str | None = None
    sensor_role: DiagnosticSensorRole = "primary"
    measurement_quantity: str = "三轴加速度、振动 RMS 与主频"
    recommended_phyphox_experiment: str = "“加速度（不含重力）”或“加速度”实验"
    analyzer_status: TaskAnalyzerStatus = "ready"
    task_kind: TaskKind
    comparison_task_id: str | None = None
    target_hypothesis_ids: list[str] = Field(default_factory=list)
    expected_effect: ExpectedEffect = "unknown"
    effect_metric: EffectMetric = "either"
    status: TaskStatus = "pending"


class DiagnosticControlEffect(BaseModel):
    baseline_task_id: str
    baseline_session_id: str
    sensor: SensorKind = "accelerometer"
    metric_key: str = "selected_axis_rms_m_s2"
    metric_unit: str = "m/s^2"
    baseline_value: float = 0.0
    current_value: float = 0.0
    absolute_delta: float = 0.0
    relative_change_ratio: float | None = None
    # Retained for old stored cases and the old acceleration-only UI.
    rms_change_ratio: float = 0.0
    frequency_shift_hz: float = 0.0
    frequency_resolution_hz: float = 0.0
    observed_effect: Literal["increase", "decrease", "change", "no_change"]
    matches_expected_effect: bool
    comparable: bool = True
    comparison_warnings: list[str] = Field(default_factory=list)


class DiagnosticMeasurementFact(BaseModel):
    fact_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", max_length=80)
    sensor: SensorKind
    metric_key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=80)
    metric_label: str = Field(min_length=1, max_length=100)
    metric_unit: str = Field(max_length=24)
    value: float
    quality: Literal["low", "medium", "high"]
    source_session_ids: list[str] = Field(min_length=1, max_length=8)
    provenance_source: Literal[
        "phyphox_remote",
        "phone_upload",
        "file_import",
        "public_replay",
        "test_fixture",
        "legacy_session",
    ]
    baseline_value: float | None = None
    absolute_delta: float | None = None
    relative_delta_ratio: float | None = None
    relation: Literal[
        "single_observation",
        "increase",
        "decrease",
        "within_repeatability",
    ] = "single_observation"
    analyzer_id: str = Field(default="", max_length=120)
    analyzer_version: str = Field(default="", max_length=40)
    sample_count: int = Field(default=0, ge=0)
    duration_s: float = Field(default=0.0, ge=0)
    sampling_rate_hz: float = Field(default=0.0, ge=0)
    analysis_warnings: list[str] = Field(default_factory=list, max_length=20)
    companion_metrics: list[AnalysisMetric] = Field(default_factory=list, max_length=16)


DiagnosticActionId = Literal[
    "preserve-and-observe",
    "repeat-controlled-measurement",
    "redistribute-balanced-load",
    "remove-external-contact",
    "stabilize-external-support",
    "reduce-user-adjustable-source",
    "reposition-within-safe-use",
    "improve-light-path",
    "reduce-acoustic-exposure",
    "reduce-magnetic-interference",
    "clear-sensor-path",
    "verify-environmental-context",
    "isolate-operating-source",
    "check-manufacturer-guidance",
    "request-professional-inspection",
]


class DiagnosticReasoningReceipt(BaseModel):
    model_name: str = Field(min_length=1, max_length=160)
    answer_headline: str = Field(min_length=8, max_length=300)
    mechanism_explanation: str = Field(min_length=12, max_length=1600)
    confidence: Literal["low", "medium", "high"]
    ranked_hypothesis_ids: list[str] = Field(min_length=2, max_length=3)
    source_fact_ids: list[str] = Field(min_length=1, max_length=32)
    next_measurement_reason: str = Field(default="", max_length=800)
    solution_rationale: str = Field(default="", max_length=1000)
    recommended_action_ids: list[DiagnosticActionId] = Field(min_length=1, max_length=3)
    transport: Literal[
        "agent_tool",
        "validated_json_chat",
        "deterministic_fallback",
    ] = "agent_tool"
    model_requests: int = Field(default=0, ge=0, le=8)
    elapsed_ms: int = Field(default=0, ge=0, le=86_400_000)
    fallback_reason: str | None = Field(default=None, max_length=160)


class DiagnosticEvidence(BaseModel):
    evidence_id: str
    task_id: str
    session_id: str
    quality: Literal["low", "medium", "high"]
    summary: str
    observation_notes: str = ""
    hypothesis_assessments: list[HypothesisAssessmentDraft] = Field(default_factory=list)
    control_effect: DiagnosticControlEffect | None = None
    sensor: SensorKind = "accelerometer"
    facts: list[DiagnosticMeasurementFact] = Field(default_factory=list)
    reasoning_receipt: DiagnosticReasoningReceipt | None = None


class DiagnosticTerminationVector(BaseModel):
    effective_evidence_count: int = 0
    effective_control_count: int = 0
    matched_control_count: int = 0
    hypothesis_coverage_ratio: float = 0.0
    support_scores: dict[str, float] = Field(default_factory=dict)
    leading_hypothesis_id: str | None = None
    runner_up_hypothesis_id: str | None = None
    leading_support: float = 0.5
    runner_up_support: float = 0.5
    leading_margin: float = 0.0
    leading_positive_weight: float = 0.0
    leading_negative_weight: float = 0.0
    high_quality_contradictions: int = 0
    consecutive_low_quality_count: int = 0
    recent_information_gain: float = 0.0
    hypothesis_set_state: Literal[
        "unverified",
        "active_leader",
        "tied",
        "mixed",
        "all_weakened",
    ] = "unverified"
    hypothesis_revision_required: bool = False
    intervention_effect_ready: bool = False
    conclusion_ready: bool = False
    forced_stop: bool = False
    completed_task_count: int = 0
    distinct_sensor_count: int = 0
    required_sensor_diversity: int = 1
    soft_checkpoint_reached: bool = False
    hard_stop_reached: bool = False
    user_decision_required: bool = False
    bounded_rehearsal_stop: bool = False
    stop_reason_code: Literal["public-replay-evidence-exhausted"] | None = None
    blockers: list[str] = Field(default_factory=list)


SolutionRiskLevel = Literal["low", "caution", "professional"]


class DiagnosticSolutionAction(BaseModel):
    action_id: DiagnosticActionId
    action_role: Literal["resolve", "verify", "escalate"] = "resolve"
    title: str
    rationale: str
    preparation: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    expected_result: str
    how_to_verify: str = ""
    if_not_improved: str = ""
    estimated_time: str = ""
    tools_needed: list[str] = Field(default_factory=list)
    do_not_do: list[str] = Field(default_factory=list)
    risk_level: SolutionRiskLevel = "low"
    safety_notes: list[str] = Field(default_factory=list)


class DiagnosticOptionalRetest(BaseModel):
    optional: Literal[True] = True
    title: str
    purpose: str
    instruction: str
    controlled_variables: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    result_use: str


class DiagnosticSolutionPlan(BaseModel):
    basis: Literal["evidence_supported", "inconclusive_safe_next_steps"]
    summary: str
    evidence_summary: str = ""
    first_action_reason: str = ""
    actions: list[DiagnosticSolutionAction] = Field(default_factory=list)
    escalation_conditions: list[str] = Field(default_factory=list)
    optional_retest: DiagnosticOptionalRetest | None = None


class DiagnosticFinalReport(BaseModel):
    outcome: Literal["completed_with_conclusion", "completed_inconclusive"]
    confidence: Literal["low", "medium", "high"]
    leading_hypothesis_id: str | None = None
    conclusion: str
    evidence_basis: list[str] = Field(default_factory=list)
    remaining_uncertainties: list[str] = Field(default_factory=list)
    termination_reason: str
    vector: DiagnosticTerminationVector
    solution_plan: DiagnosticSolutionPlan | None = None
    answer_headline: str = ""
    mechanism_explanation: str = ""
    ranked_hypothesis_ids: list[str] = Field(default_factory=list)
    source_fact_ids: list[str] = Field(default_factory=list)
    user_takeaway: str = ""
    evidence_explanation: list[str] = Field(default_factory=list)
    confidence_explanation: str = ""
    scope_boundary: str = ""
    finalization_source: Literal[
        "model_generated",
        "deterministic_fallback",
        "legacy_unattributed",
    ] = "legacy_unattributed"
    finalization_model: str = ""
    finalization_transport: str = ""
    finalization_model_requests: int = Field(default=0, ge=0, le=16)
    finalization_elapsed_ms: int = Field(default=0, ge=0, le=86_400_000)
    finalization_fallback_reason: str | None = Field(default=None, max_length=500)
    finalization_retryable: bool = False


class DiagnosticCase(BaseModel):
    case_id: str
    title: str
    problem_statement: str
    context: str = ""
    status: DiagnosticCaseStatus = "planning"
    sensor_plan: list[DiagnosticSensorPlanItem] = Field(default_factory=list, max_length=4)
    hypotheses: list[DiagnosticHypothesis] = Field(default_factory=list)
    current_task: DiagnosticMeasurementTask | None = None
    completed_tasks: list[DiagnosticMeasurementTask] = Field(default_factory=list)
    evidence: list[DiagnosticEvidence] = Field(default_factory=list)
    termination_vector: DiagnosticTerminationVector = Field(
        default_factory=DiagnosticTerminationVector
    )
    final_report: DiagnosticFinalReport | None = None
    termination_invalidated: bool = False
    termination_invalidation_reason: str = Field(default="", max_length=500)
    checkpoint_pending: bool = False
    continued_after_checkpoint: bool = False
    checkpoint_next_task: DiagnosticMeasurementTask | None = None
    intake_transport: Literal[
        "validated_json_chat",
        "deterministic_fallback",
    ] | None = None
    intake_model: str = ""
    intake_model_requests: int = Field(default=0, ge=0, le=16)
    intake_elapsed_ms: int = Field(default=0, ge=0, le=86_400_000)
    intake_fallback_reason: str | None = Field(default=None, max_length=300)
    revision_parent_case_id: str | None = Field(default=None, max_length=80)
    revision_feedback: RealityFeedbackRecord | None = None
    superseded_by_case_id: str | None = Field(default=None, max_length=80)


class DiagnosticCaseHistoryItem(BaseModel):
    case_id: str
    title: str
    problem_statement: str
    status: DiagnosticCaseStatus
    current_task_title: str | None = None
    evidence_count: int
    superseded_by_case_id: str | None = None
    created_at: str
    updated_at: str


class DiagnosticCaseSnapshot(BaseModel):
    case: DiagnosticCase
    latest_agent_message: str = ""
    created_at: str
    updated_at: str


class DiagnosticMeasurementSubmit(BaseModel):
    task_id: str = Field(min_length=1, max_length=40)
    session_id: str = Field(min_length=1, max_length=40)
    observation_notes: str = Field(default="", max_length=500)


class DiagnosticRecordingSubmit(BaseModel):
    recording_id: str = Field(min_length=1, max_length=40)
    observation_notes: str = Field(default="", max_length=500)


class DiagnosticSensorTaskResponse(BaseModel):
    session: SensorRecordingCreated
    case: DiagnosticCase
    agent_message: str
    model: str
    preview_samples: list[SensorSample] = Field(default_factory=list, max_length=1001)


class DiagnosticTaskPhyphoxRequest(BaseModel):
    base_url: str = Field(min_length=10, max_length=200)
    duration_s: float = Field(default=5.0, ge=1.0, le=120.0)
    label: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=500)
    observation_notes: str = Field(default="", max_length=500)
    privacy_acknowledged: bool = False


class DiagnosticPublicReplaySubmit(BaseModel):
    dataset_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
        max_length=100,
    )
    recording_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
        max_length=80,
    )
    observation_notes: str = Field(default="公开数据仅用于诊断闭环演示。", max_length=500)
    privacy_acknowledged: bool = False


class DiagnosticCheckpointDecision(BaseModel):
    decision: Literal["continue", "stop"]
    expected_completed_task_count: int = Field(ge=20, le=32)


class DiagnosticAgentResponse(BaseModel):
    case: DiagnosticCase
    agent_message: str
    model: str


class MobileTaskResponse(BaseModel):
    case_id: str
    case_title: str
    problem_statement: str
    status: DiagnosticCaseStatus
    task: DiagnosticMeasurementTask | None = None
    hypotheses: list[DiagnosticHypothesis]
    evidence_count: int
    final_report: DiagnosticFinalReport | None = None


ExplorationReadiness = Literal[
    "ready",
    "analysis_ready",
    "capability_detectable",
    "planned",
]
ExplorationActionKind = Literal[
    "diagnostic_agent",
    "bounded_agent",
    "sensor_analysis",
    "capability_check",
]


class ExplorationSource(BaseModel):
    label: str
    url: str
    role: str


class ExplorationTemplate(BaseModel):
    exploration_id: str
    title: str
    question: str
    category: str
    primary_sensor: SensorKind
    secondary_sensors: list[SensorKind] = Field(default_factory=list)
    readiness: ExplorationReadiness
    readiness_note: str
    action_kind: ExplorationActionKind
    executable_protocol_id: str | None = None
    simulation_question: str | None = Field(default=None, min_length=5, max_length=800)
    simulation_scope_note: str | None = Field(default=None, min_length=5, max_length=500)
    duration_minutes: int = Field(ge=1, le=120)
    difficulty: Literal["入门", "进阶"]
    protocol: list[str]
    expected_signal: str
    output_value: str
    safety_notes: list[str] = Field(default_factory=list)
    privacy_notes: list[str] = Field(default_factory=list)
    sources: list[ExplorationSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def action_matches_evidence_boundary(self) -> "ExplorationTemplate":
        if self.action_kind == "bounded_agent" and not self.executable_protocol_id:
            raise ValueError("bounded_agent explorations require an executable protocol")
        if self.action_kind == "bounded_agent" and (
            not self.simulation_question or not self.simulation_scope_note
        ):
            raise ValueError(
                "bounded_agent explorations require a scoped simulation question"
            )
        if self.executable_protocol_id and self.action_kind != "bounded_agent":
            raise ValueError("executable protocols must use action_kind=bounded_agent")
        if self.action_kind == "diagnostic_agent" and self.readiness != "ready":
            raise ValueError("diagnostic_agent explorations must have readiness=ready")
        if self.action_kind == "sensor_analysis" and self.readiness != "analysis_ready":
            raise ValueError("sensor_analysis explorations must have analysis_ready evidence")
        return self


class TaskSampleUpload(SessionUpload):
    observation_notes: str = Field(default="", max_length=500)


class TaskSampleResponse(BaseModel):
    session: SessionCreated
    case: DiagnosticCase
    agent_message: str
    model: str


class PhyphoxBufferMapping(BaseModel):
    timestamp: str = Field(default="acc_time", min_length=1, max_length=80)
    x: str = Field(default="accX", min_length=1, max_length=80)
    y: str = Field(default="accY", min_length=1, max_length=80)
    z: str = Field(default="accZ", min_length=1, max_length=80)


class PhyphoxConnectionRequest(BaseModel):
    base_url: str = Field(min_length=10, max_length=200)
    buffer_mapping: PhyphoxBufferMapping = Field(default_factory=PhyphoxBufferMapping)


class PhyphoxProbeResponse(BaseModel):
    base_url: str
    experiment_title: str
    remote_session: str | None = None
    measuring: bool
    compatible: bool
    buffer_mapping: PhyphoxBufferMapping
    available_buffers: list[str]
    missing_buffers: list[str]
    detected_sensors: list[SensorKind] = Field(default_factory=list)
    export_buffers: list[str] = Field(default_factory=list)
    exploration_matches: list[str] = Field(default_factory=list)
    config_sha256: str = ""
    sensor_profiles: dict[SensorKind, PhyphoxSensorProfile] = Field(default_factory=dict)


class PhyphoxCaptureRequest(PhyphoxConnectionRequest):
    duration_s: float = Field(default=5.0, ge=3.0, le=60.0)
    label: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=500)
    observation_notes: str = Field(default="", max_length=500)


class PhyphoxCaptureMetadata(BaseModel):
    source: Literal["phyphox_remote"] = "phyphox_remote"
    experiment_title: str
    remote_session: str | None = None
    requested_duration_s: float
    actual_duration_s: float
    sample_count: int
    buffer_mapping: PhyphoxBufferMapping


class PhyphoxTaskResponse(TaskSampleResponse):
    capture: PhyphoxCaptureMetadata
    preview_samples: list[AccelerationSample]


class LocalProfile(BaseModel):
    user_id: str
    display_name: str
    created_at: str
    updated_at: str


class LocalProfileUpdate(BaseModel):
    display_name: str = Field(min_length=1, max_length=60)


class SavedPhyphoxDevice(BaseModel):
    device_id: str
    name: str
    base_url: str
    buffer_mapping: PhyphoxBufferMapping
    experiment_title: str = ""
    compatible: bool = False
    is_default: bool = True
    last_seen_at: str | None = None
    created_at: str
    updated_at: str


class PhyphoxDeviceSaveRequest(PhyphoxConnectionRequest):
    name: str = Field(default="我的手机", min_length=1, max_length=80)


class PocketLabSettings(BaseModel):
    profile: LocalProfile
    default_phyphox_device: SavedPhyphoxDevice | None = None


class PhyphoxDeviceSaveResponse(BaseModel):
    device: SavedPhyphoxDevice
    probe: PhyphoxProbeResponse
