from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from pocketlab.auth import get_current_user_id
from pocketlab.experiment_guidance import ExperimentGuidanceError
from pocketlab.general_acquisition import (
    AcquisitionAlignmentAttestation,
    GeneralAcquisitionReference,
    GeneralEvidenceEnvelope,
    StoredRecordingAcquisitionSource,
    bind_general_evidence,
)
from pocketlab.general_exploration_engine import (
    create_general_experiment_case,
    prepare_general_measurement,
)
from pocketlab.general_exploration_models import (
    GeneralCompileContext,
    GeneralExplorationDraft,
    StrictFrozenModel,
)
from pocketlab.general_exploration_protocol import (
    compile_general_exploration_protocol,
    general_exploration_draft_sha256,
)
from pocketlab.general_exploration_state import (
    GeneralCompilerProvenance,
    GeneralExperimentCase,
    GeneralMeasurementSubmission,
    PreparedGeneralTransition,
)
from pocketlab.general_question_compiler import (
    GENERAL_CLARIFICATION_CODES,
    GeneralQuestionClarificationReceipt,
    GeneralQuestionCompilationReceipt,
    GeneralQuestionCompileRequest,
    GeneralQuestionCompileResult,
    general_clarification_request_sha256,
    general_clarification_resolution_sha256,
)
from pocketlab.general_simulation import (
    GeneralSimulationMeasurementRequest,
    build_general_simulated_evidence,
)
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.reality_feedback import (
    RealityEvidenceReuseAudit,
    RealityEvidenceReuseCandidate,
    RealityFeedbackRecord,
    RealityFeedbackRequest,
    build_reality_evidence_reuse_audit,
)
from pocketlab.sensor_models import SensorKind
from pocketlab.store import SessionStore, session_store


class GeneralExplorationNotFound(KeyError):
    pass


class GeneralExplorationConflict(RuntimeError):
    pass


class GeneralExplorationValidation(ValueError):
    def __init__(
        self,
        message: str,
        *,
        blocker_codes: tuple[str, ...] = (),
        user_messages: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.blocker_codes = blocker_codes
        self.user_messages = user_messages


def build_general_reality_evidence_reuse_audit(
    case: GeneralExperimentCase,
    request: RealityFeedbackRequest,
) -> RealityEvidenceReuseAudit:
    """Select old deterministic facts for planning context without importing evidence."""

    condition_labels = {
        item.condition_id: item.label for item in case.protocol.conditions
    }
    rehearsal = set(case.protocol.selected_sources) == {"protocol_emulator"}
    candidates: list[RealityEvidenceReuseCandidate] = []
    for evidence in case.evidence:
        eligible = evidence.valid and evidence.quality in {"medium", "high"}
        blocker = None
        if not eligible:
            blocker = "low-quality-or-invalid"
        elif rehearsal:
            if evidence.lineage.source != "protocol_emulator":
                eligible = False
                blocker = "non-user-evidence-source"
        elif evidence.lineage.source not in {"phyphox_live", "phone_upload"}:
            eligible = False
            blocker = "non-user-evidence-source"
        scope_label = "模拟排练" if evidence.lineage.simulated else "现场记录"
        condition_label = condition_labels.get(evidence.condition_id, evidence.condition_id)
        summary = (
            f"{scope_label}“{condition_label}”中，{evidence.metric.label}为 "
            f"{evidence.metric.value:.4g} {evidence.metric.unit or '（无单位）'}。"
        )
        candidates.append(
            RealityEvidenceReuseCandidate(
                evidence_id=evidence.evidence_id,
                sensor=evidence.sensor,
                planning_summary=summary,
                eligible=eligible,
                exclusion_reason_code=blocker,
            )
        )
    return build_reality_evidence_reuse_audit(
        tuple(candidates),
        confirm_sensitive_sensor_reuse=request.confirm_sensitive_sensor_reuse,
    )


class GeneralClarificationReceiptReservation(StrictFrozenModel):
    receipt: GeneralQuestionClarificationReceipt
    reservation_token: str = Field(pattern=r"^clarify-reservation-[0-9a-f]{24}$")


class GeneralExplorationCaseCreate(StrictFrozenModel):
    draft: GeneralExplorationDraft
    compilation_receipt_id: str | None = Field(
        default=None,
        pattern=r"^general-compile-[0-9a-f]{20}$",
    )
    source: Literal["phone_upload", "protocol_emulator"] = "phone_upload"
    privacy_acknowledged_sensors: tuple[SensorKind, ...] = Field(
        default=(),
        max_length=2,
    )

    @field_validator("draft", mode="before")
    @classmethod
    def normalize_json_draft_arrays(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for field_name in (
            "conditions",
            "sensor_intents",
            "controls",
            "safety_notes",
            "privacy_notes",
            "claim_boundaries",
        ):
            field_value = normalized.get(field_name)
            if isinstance(field_value, list):
                normalized[field_name] = tuple(field_value)
        return normalized

    @field_validator("privacy_acknowledged_sensors", mode="before")
    @classmethod
    def normalize_privacy_array(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def acknowledgements_are_unique_and_relevant(self) -> Self:
        if len(self.privacy_acknowledged_sensors) != len(
            set(self.privacy_acknowledged_sensors)
        ):
            raise ValueError("privacy acknowledgements must be unique")
        requested = {item.sensor for item in self.draft.sensor_intents}
        if not set(self.privacy_acknowledged_sensors) <= requested:
            raise ValueError("privacy acknowledgement references an unrequested sensor")
        return self


class GeneralRecordingMeasurementSubmit(StrictFrozenModel):
    expected_revision: int = Field(ge=1)
    task_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", max_length=80)
    recording_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    controls_confirmed: bool

    @field_validator("recording_ids", mode="before")
    @classmethod
    def normalize_recording_array(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def recording_ids_are_unique(self) -> Self:
        if len(self.recording_ids) != len(set(self.recording_ids)):
            raise ValueError("measurement recording IDs must be unique")
        return self


class GeneralPhyphoxCaptureRequest(StrictFrozenModel):
    expected_revision: int = Field(ge=1)
    task_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", max_length=80)
    duration_s: float = Field(default=8.0, ge=1.0, le=300.0)
    controls_confirmed: bool
    privacy_acknowledged: bool


class GeneralPhyphoxCaptureMetadata(StrictFrozenModel):
    source: Literal["phyphox_remote"] = "phyphox_remote"
    recording_id: str = Field(min_length=6, max_length=80)
    experiment_title: str = Field(min_length=1, max_length=120)
    remote_session: str | None = Field(default=None, max_length=120)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_duration_s: float = Field(ge=1.0, le=300.0)
    actual_duration_s: float = Field(gt=0.0, le=360.0)
    sample_count: int = Field(ge=2, le=120_000)
    sensor: SensorKind
    analyzer_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", max_length=80)


class GeneralPhyphoxCaptureResponse(StrictFrozenModel):
    case: GeneralExperimentCase
    evidence: GeneralEvidenceEnvelope
    capture: GeneralPhyphoxCaptureMetadata

    @model_validator(mode="after")
    def capture_is_bound_to_returned_case(self) -> Self:
        if self.capture.recording_id != self.evidence.lineage.recording_id:
            raise ValueError("capture recording must match returned evidence")
        if self.capture.sensor != self.evidence.sensor:
            raise ValueError("capture sensor must match returned evidence")
        if self.evidence.lineage.source != "phyphox_live":
            raise ValueError("general phyphox capture must preserve live source identity")
        if not any(
            item.evidence_id == self.evidence.evidence_id for item in self.case.evidence
        ):
            raise ValueError("capture evidence must already be present in returned case")
        return self


class GeneralPhyphoxSynchronizedCaptureResponse(StrictFrozenModel):
    case: GeneralExperimentCase
    evidence: tuple[GeneralEvidenceEnvelope, ...] = Field(min_length=2, max_length=3)
    captures: tuple[GeneralPhyphoxCaptureMetadata, ...] = Field(min_length=2, max_length=3)
    alignment: AcquisitionAlignmentAttestation

    @model_validator(mode="after")
    def synchronized_capture_is_closed(self) -> Self:
        evidence_by_recording = {item.lineage.recording_id: item for item in self.evidence}
        if len(evidence_by_recording) != len(self.evidence):
            raise ValueError("synchronized evidence cannot duplicate recordings")
        if {item.recording_id for item in self.captures} != set(evidence_by_recording):
            raise ValueError("synchronized capture metadata must match every evidence record")
        if {item.sensor for item in self.captures} != {item.sensor for item in self.evidence}:
            raise ValueError("synchronized capture sensors must match evidence sensors")
        if any(item.lineage.alignment != self.alignment for item in self.evidence):
            raise ValueError("synchronized evidence must retain one alignment attestation")
        case_evidence = {item.evidence_id for item in self.case.evidence}
        if not {item.evidence_id for item in self.evidence} <= case_evidence:
            raise ValueError("synchronized evidence must already be committed to the case")
        return self


class GeneralExplorationCaseHistoryItem(StrictFrozenModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", max_length=80)
    revision: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=160)
    status: Literal[
        "collecting",
        "awaiting_user_decision",
        "completed_descriptive",
        "completed_inconclusive",
    ]
    primary_sensor: SensorKind
    current_task_title: str | None = Field(default=None, max_length=180)
    evidence_count: int = Field(ge=0, le=256)
    superseded_by_case_id: str | None = Field(default=None, max_length=80)
    compiler_source: Literal["manual", "bounded_agent_compiler"]
    execution_mode: Literal["physical_exploration", "simulated_rehearsal"]
    created_at: str = Field(min_length=10, max_length=64)
    updated_at: str = Field(min_length=10, max_length=64)


class GeneralAcquisitionSourcePlan(StrictFrozenModel):
    source: Literal[
        "phyphox_live",
        "account_recording",
        "public_analogue",
        "protocol_simulator",
    ]
    status: Literal[
        "available",
        "setup_required",
        "no_matching_recording",
        "not_authorized",
        "analogue_only",
        "terminal",
    ]
    candidate_recording_ids: tuple[str, ...] = Field(default=(), max_length=200)
    recoverable_recording_ids: tuple[str, ...] = Field(default=(), max_length=8)
    counts_as_case_evidence: bool
    may_supply_user_phone_evidence: bool
    gate_c_credited_records: Literal[0] = 0
    boundary_message: str = Field(min_length=10, max_length=500)

    @model_validator(mode="after")
    def source_semantics_are_closed(self) -> Self:
        if len(self.candidate_recording_ids) != len(set(self.candidate_recording_ids)):
            raise ValueError("acquisition candidate IDs must be unique")
        if not set(self.recoverable_recording_ids) <= set(self.candidate_recording_ids):
            raise ValueError("recoverable recordings must be bindable candidates")
        if self.source == "public_analogue":
            if (
                self.counts_as_case_evidence
                or self.may_supply_user_phone_evidence
                or self.candidate_recording_ids
                or self.recoverable_recording_ids
                or self.status not in {"analogue_only", "terminal"}
            ):
                raise ValueError("public analogues cannot become general-case evidence")
        elif self.source == "protocol_simulator":
            if (
                not self.counts_as_case_evidence
                or self.may_supply_user_phone_evidence
                or self.candidate_recording_ids
                or self.recoverable_recording_ids
                or self.status not in {"available", "not_authorized", "terminal"}
            ):
                raise ValueError("protocol simulator must remain non-physical rehearsal evidence")
        elif not self.counts_as_case_evidence or not self.may_supply_user_phone_evidence:
            raise ValueError("live and account sources must retain their evidence role")
        if self.source != "account_recording" and (
            self.candidate_recording_ids or self.recoverable_recording_ids
        ):
            raise ValueError("only the account-recording option may expose recording IDs")
        return self


class GeneralAcquisitionPlan(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$", max_length=80)
    case_revision: int = Field(ge=1)
    task_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        max_length=80,
    )
    required_sensors: tuple[SensorKind, ...] = Field(default=(), max_length=3)
    sources: tuple[GeneralAcquisitionSourcePlan, ...] = Field(min_length=4, max_length=4)
    case_evidence_required: bool
    public_analogue_can_complete_case: Literal[False] = False
    phyphox_validated: Literal[False] = False

    @model_validator(mode="after")
    def plan_has_exact_source_partition(self) -> Self:
        if {item.source for item in self.sources} != {
            "phyphox_live",
            "account_recording",
            "public_analogue",
            "protocol_simulator",
        }:
            raise ValueError("acquisition plan requires four explicit source roles")
        terminal = self.task_id is None
        if terminal != (not self.case_evidence_required):
            raise ValueError("terminal acquisition plans cannot require new evidence")
        if terminal and (self.required_sensors or any(item.status != "terminal" for item in self.sources)):
            raise ValueError("terminal acquisition plans cannot retain active sources")
        if not terminal and not self.required_sensors:
            raise ValueError("active acquisition plans require task sensors")
        return self


class GeneralExplorationStore:
    """Persist one user's immutable general-exploration graph with revision CAS."""

    def __init__(
        self,
        storage: SQLiteDatabase | None = None,
        recordings: SessionStore | None = None,
        *,
        user_id: str | None = DEFAULT_USER_ID,
    ) -> None:
        self._database = storage or SQLiteDatabase(":memory:")
        self._recordings = recordings or SessionStore(self._database, user_id=user_id)
        self._acquisitions = StoredRecordingAcquisitionSource(self._recordings)
        self._user_id = user_id

    @property
    def _active_user_id(self) -> str:
        return self._user_id or get_current_user_id()

    def create(self, request: GeneralExplorationCaseCreate) -> GeneralExperimentCase:
        selected_sources = (
            ("protocol_emulator",)
            if request.source == "protocol_emulator"
            else ("phyphox_live",)
            if request.draft.alignment == "simultaneous"
            else (request.source, "phyphox_live")
        )
        compilation = compile_general_exploration_protocol(
            request.draft,
            GeneralCompileContext(
                selected_sources=selected_sources,
                privacy_acknowledged_sensors=request.privacy_acknowledged_sensors,
                supports_simultaneous_capture=True,
                allow_deferred_live_detection=True,
                enable_adaptive_sufficiency=True,
                enable_server_owned_optional_activation=True,
            ),
        )
        if compilation.status != "executable":
            raise GeneralExplorationValidation(
                "当前问题尚未形成可执行的通用实验协议。",
                blocker_codes=compilation.blocker_codes,
                user_messages=compilation.user_messages,
            )
        case_id = f"general-{uuid4().hex[:16]}"
        now = utc_now()
        try:
            with self._database.transaction() as connection:
                provenance = GeneralCompilerProvenance()
                if request.compilation_receipt_id is not None:
                    row = connection.execute(
                        """
                        SELECT draft_sha256, receipt_json, consumed_case_id
                        FROM general_compilation_receipts
                        WHERE receipt_id = ? AND user_id = ?
                        """,
                        (request.compilation_receipt_id, self._active_user_id),
                    ).fetchone()
                    if row is None:
                        raise GeneralExplorationValidation(
                            "编译凭证不存在、已失效或不属于当前账号。"
                        )
                    if row["consumed_case_id"] is not None:
                        raise GeneralExplorationConflict("该编译凭证已经创建过实验。")
                    receipt = GeneralQuestionCompilationReceipt.model_validate_json(
                        row["receipt_json"]
                    )
                    if (
                        receipt.receipt_id != request.compilation_receipt_id
                        or receipt.draft_sha256 != compilation.protocol.draft_sha256
                        or row["draft_sha256"] != compilation.protocol.draft_sha256
                    ):
                        raise GeneralExplorationValidation(
                            "当前草案已修改，不能沿用旧 Agent 编译凭证。"
                        )
                    provenance = GeneralCompilerProvenance(
                        source="bounded_agent_compiler",
                        receipt_id=receipt.receipt_id,
                        draft_sha256=receipt.draft_sha256,
                        compiler_model=receipt.compiler_model,
                        transport=receipt.transport,
                        tool_event_names=receipt.tool_event_names,
                        created_at=receipt.created_at,
                    )
                try:
                    case = create_general_experiment_case(
                        compilation,
                        case_id=case_id,
                        compiler_provenance=provenance,
                    )
                except ExperimentGuidanceError as exc:
                    raise GeneralExplorationValidation(
                        "实验草案中的操作仍是占位描述，请明确写出用户实际要做的单一变化。",
                        blocker_codes=exc.blocker_codes,
                        user_messages=(
                            "请把“目标条件”改成具体动作，例如“关闭台灯”或“把手机移到距声源 1 米处”。",
                        ),
                    ) from exc
                connection.execute(
                    """
                    INSERT INTO general_exploration_cases(
                        case_id, user_id, revision, case_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case.case_id,
                        self._active_user_id,
                        case.revision,
                        case.model_dump_json(),
                        now,
                        now,
                    ),
                )
                if request.compilation_receipt_id is not None:
                    cursor = connection.execute(
                        """
                        UPDATE general_compilation_receipts
                        SET consumed_case_id = ?, consumed_at = ?
                        WHERE receipt_id = ? AND user_id = ?
                          AND consumed_case_id IS NULL
                        """,
                        (
                            case.case_id,
                            now,
                            request.compilation_receipt_id,
                            self._active_user_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise GeneralExplorationConflict(
                            "该编译凭证已被其他请求使用，请重新编译。"
                        )
        except sqlite3.IntegrityError as exc:  # pragma: no cover - UUID collision guard
            raise GeneralExplorationConflict("通用探索编号冲突，请重试。") from exc
        return case

    def issue_compilation_receipt(
        self,
        result: GeneralQuestionCompileResult,
    ) -> GeneralQuestionCompilationReceipt:
        """Persist only a safe hash-bound receipt for an accepted Agent draft."""

        result = GeneralQuestionCompileResult.model_validate(
            result.model_dump(mode="python")
        )
        if (
            result.status != "draft_ready"
            or result.source != "bounded_agent"
            or result.draft is None
            or result.runtime.status != "completed"
            or result.runtime.fallback_reason != "none"
            or result.runtime.transport == "deterministic_fallback"
        ):
            raise GeneralExplorationValidation(
                "只有成功的受限 Agent 草案可以获得编译凭证。"
            )
        created_at = utc_now()
        receipt = GeneralQuestionCompilationReceipt(
            receipt_id=f"general-compile-{uuid4().hex[:20]}",
            draft_sha256=general_exploration_draft_sha256(result.draft),
            compiler_model=result.runtime.model,
            transport=result.runtime.transport,
            tool_event_names=result.runtime.tool_event_names,
            created_at=created_at,
        )
        try:
            self._database.execute(
                """
                INSERT INTO general_compilation_receipts(
                    receipt_id, user_id, draft_sha256, receipt_json,
                    consumed_case_id, created_at, consumed_at
                ) VALUES (?, ?, ?, ?, NULL, ?, NULL)
                """,
                (
                    receipt.receipt_id,
                    self._active_user_id,
                    receipt.draft_sha256,
                    receipt.model_dump_json(),
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:  # pragma: no cover - UUID collision guard
            raise GeneralExplorationConflict("编译凭证编号冲突，请重新编译。") from exc
        return receipt

    def issue_clarification_receipt(
        self,
        request: GeneralQuestionCompileRequest,
        result: GeneralQuestionCompileResult,
    ) -> GeneralQuestionClarificationReceipt:
        """Persist only an opaque question-bound receipt for a real clarification."""

        request = GeneralQuestionCompileRequest.model_validate(request.model_dump(mode="python"))
        result = GeneralQuestionCompileResult.model_validate(result.model_dump(mode="python"))
        reason_codes = tuple(
            code for code in GENERAL_CLARIFICATION_CODES if code in result.blocker_codes
        )
        if (
            not request.use_agent
            or result.status != "needs_clarification"
            or result.source not in {"bounded_agent", "server_policy"}
            or result.runtime.fallback_reason != "none"
            or not reason_codes
            or set(reason_codes) != set(result.blocker_codes)
        ):
            raise GeneralExplorationValidation(
                "只有真实、有限且未回退的澄清结果可以获得澄清凭证。"
            )
        created_at = utc_now()
        receipt = GeneralQuestionClarificationReceipt(
            receipt_id=f"general-clarify-{uuid4().hex[:20]}",
            request_sha256=general_clarification_request_sha256(request),
            reason_codes=reason_codes,
            source=result.source,
            compiler_model=result.runtime.model,
            created_at=created_at,
        )
        try:
            self._database.execute(
                """
                INSERT INTO general_clarification_receipts(
                    receipt_id, user_id, request_sha256, receipt_json,
                    reservation_token, reserved_at, consumed_resolution_sha256,
                    created_at, consumed_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, NULL)
                """,
                (
                    receipt.receipt_id,
                    self._active_user_id,
                    receipt.request_sha256,
                    receipt.model_dump_json(),
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:  # pragma: no cover - UUID collision guard
            raise GeneralExplorationConflict("澄清凭证编号冲突，请重试。") from exc
        return receipt

    def reserve_clarification_receipt(
        self,
        request: GeneralQuestionCompileRequest,
    ) -> GeneralClarificationReceiptReservation:
        """Atomically reserve a user/question-bound receipt before model execution."""

        request = GeneralQuestionCompileRequest.model_validate(request.model_dump(mode="python"))
        if request.clarification_receipt_id is None:
            raise GeneralExplorationValidation("补充澄清时必须提供上一轮的一次性澄清凭证。")
        resolved_codes = {
            *(item.reason_code for item in request.clarification_answers),
            *(request.condition_resolution.reason_codes if request.condition_resolution else ()),
            *(
                (request.mechanism_resolution.reason_code,)
                if request.mechanism_resolution is not None
                else ()
            ),
        }
        if not resolved_codes:
            raise GeneralExplorationValidation("澄清凭证必须和至少一项结构化补充一起提交。")
        reservation_token = f"clarify-reservation-{uuid4().hex[:24]}"
        reserved_at = utc_now()
        with self._database.transaction() as connection:
            row = connection.execute(
                """
                SELECT request_sha256, receipt_json, reservation_token, consumed_at
                FROM general_clarification_receipts
                WHERE receipt_id = ? AND user_id = ?
                """,
                (request.clarification_receipt_id, self._active_user_id),
            ).fetchone()
            if row is None:
                raise GeneralExplorationValidation(
                    "澄清凭证不存在、已失效或不属于当前账号。"
                )
            if row["consumed_at"] is not None:
                raise GeneralExplorationConflict("该澄清凭证已经使用过，请重新开始编译。")
            if row["reservation_token"] is not None:
                raise GeneralExplorationConflict("该澄清凭证正在被另一个请求使用。")
            receipt = GeneralQuestionClarificationReceipt.model_validate_json(
                row["receipt_json"]
            )
            request_sha256 = general_clarification_request_sha256(request)
            if (
                receipt.receipt_id != request.clarification_receipt_id
                or receipt.request_sha256 != request_sha256
                or row["request_sha256"] != request_sha256
            ):
                raise GeneralExplorationValidation(
                    "问题、上下文、传感器或隐私设置已改变；旧澄清凭证不能复用。"
                )
            if not resolved_codes <= set(receipt.reason_codes):
                raise GeneralExplorationValidation(
                    "补充内容超出了上一轮要求澄清的有限范围。"
                )
            cursor = connection.execute(
                """
                UPDATE general_clarification_receipts
                SET reservation_token = ?, reserved_at = ?
                WHERE receipt_id = ? AND user_id = ?
                  AND reservation_token IS NULL AND consumed_at IS NULL
                """,
                (
                    reservation_token,
                    reserved_at,
                    receipt.receipt_id,
                    self._active_user_id,
                ),
            )
            if cursor.rowcount != 1:
                raise GeneralExplorationConflict("该澄清凭证已被其他请求占用。")
        return GeneralClarificationReceiptReservation(
            receipt=receipt,
            reservation_token=reservation_token,
        )

    def release_clarification_receipt(
        self,
        reservation: GeneralClarificationReceiptReservation,
    ) -> None:
        reservation = GeneralClarificationReceiptReservation.model_validate(
            reservation.model_dump(mode="python")
        )
        self._database.execute(
            """
            UPDATE general_clarification_receipts
            SET reservation_token = NULL, reserved_at = NULL
            WHERE receipt_id = ? AND user_id = ?
              AND reservation_token = ? AND consumed_at IS NULL
            """,
            (
                reservation.receipt.receipt_id,
                self._active_user_id,
                reservation.reservation_token,
            ),
        )

    def consume_clarification_receipt(
        self,
        reservation: GeneralClarificationReceiptReservation,
        request: GeneralQuestionCompileRequest,
    ) -> None:
        reservation = GeneralClarificationReceiptReservation.model_validate(
            reservation.model_dump(mode="python")
        )
        request = GeneralQuestionCompileRequest.model_validate(request.model_dump(mode="python"))
        consumed_at = utc_now()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE general_clarification_receipts
                SET consumed_resolution_sha256 = ?, consumed_at = ?,
                    reservation_token = NULL, reserved_at = NULL
                WHERE receipt_id = ? AND user_id = ?
                  AND reservation_token = ? AND consumed_at IS NULL
                """,
                (
                    general_clarification_resolution_sha256(request),
                    consumed_at,
                    reservation.receipt.receipt_id,
                    self._active_user_id,
                    reservation.reservation_token,
                ),
            )
            if cursor.rowcount != 1:
                raise GeneralExplorationConflict(
                    "澄清凭证状态已经改变；本次结果不会被接受。"
                )

    def get(self, case_id: str) -> GeneralExperimentCase:
        row = self._database.fetch_one(
            """
            SELECT revision, case_json FROM general_exploration_cases
            WHERE case_id = ? AND user_id = ?
            """,
            (case_id, self._active_user_id),
        )
        if row is None:
            raise GeneralExplorationNotFound(f"Unknown general exploration: {case_id}")
        case = GeneralExperimentCase.model_validate_json(row["case_json"])
        if case.revision != int(row["revision"]):
            raise RuntimeError("general exploration revision column and JSON diverged")
        return case

    def list(self, *, limit: int = 100) -> list[GeneralExplorationCaseHistoryItem]:
        rows = self._database.fetch_all(
            """
            SELECT revision, case_json, created_at, updated_at
            FROM general_exploration_cases
            WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?
            """,
            (self._active_user_id, limit),
        )
        result: list[GeneralExplorationCaseHistoryItem] = []
        for row in rows:
            case = GeneralExperimentCase.model_validate_json(row["case_json"])
            primary = next(
                item.sensor for item in case.protocol.sensors if item.role == "primary"
            )
            result.append(
                GeneralExplorationCaseHistoryItem(
                    case_id=case.case_id,
                    revision=int(row["revision"]),
                    title=case.protocol.title,
                    status=case.status,
                    primary_sensor=primary,
                    current_task_title=(
                        case.current_task.title
                        if case.current_task is not None
                        else "等待继续/收手选择"
                        if case.reasoning_checkpoint is not None
                        else None
                    ),
                    evidence_count=len(case.evidence),
                    superseded_by_case_id=case.superseded_by_case_id,
                    compiler_source=case.compiler_provenance.source,
                    execution_mode=(
                        "simulated_rehearsal"
                        if set(case.protocol.selected_sources) == {"protocol_emulator"}
                        else "physical_exploration"
                    ),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return result

    def delete(self, case_id: str) -> None:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM general_exploration_cases WHERE case_id = ? AND user_id = ?",
                (case_id, self._active_user_id),
            )
        if cursor.rowcount != 1:
            raise GeneralExplorationNotFound(f"Unknown general exploration: {case_id}")

    def link_reality_feedback_revision(
        self,
        source_case_id: str,
        revised_case_id: str,
        request: RealityFeedbackRequest,
        *,
        evidence_reuse: RealityEvidenceReuseAudit | None = None,
    ) -> GeneralExperimentCase:
        """Atomically link an old evidence graph to a freshly compiled replacement."""

        source = self.get(source_case_id)
        revised = self.get(revised_case_id)
        if source.superseded_by_case_id is not None:
            raise GeneralExplorationConflict("该实验已经根据现场反馈生成过新版本。")
        if source.report is not None:
            raise GeneralExplorationValidation("已结束实验请新建探索；此入口用于修正进行中的计划。")
        if request.expected_revision != source.revision:
            raise GeneralExplorationConflict("实验版本已经变化，请刷新后再提交现场反馈。")
        current_task_id = source.current_task.task_id if source.current_task else None
        if request.expected_task_id is not None and request.expected_task_id != current_task_id:
            raise GeneralExplorationConflict("当前任务已经变化，请刷新后再提交现场反馈。")
        known_hypotheses = {
            item.hypothesis_id: item.statement_untrusted
            for item in source.protocol.hypotheses
        }
        if set(request.hypothesis_ids) - set(known_hypotheses):
            raise GeneralExplorationValidation("反馈引用了当前协议中不存在的候选解释。")
        reuse_audit = evidence_reuse or build_general_reality_evidence_reuse_audit(
            source,
            request,
        )
        feedback = RealityFeedbackRecord(
            feedback_id=f"feedback-{uuid4().hex[:16]}",
            feedback_type=request.feedback_type,
            message=request.message,
            hypothesis_ids=request.hypothesis_ids,
            source_case_id=source.case_id,
            source_task_id=current_task_id,
            preserved_evidence_ids=tuple(item.evidence_id for item in source.evidence),
            evidence_reuse=reuse_audit,
            created_at=utc_now(),
        )
        linked_source = source.model_copy(
            update={
                "revision": source.revision + 1,
                "superseded_by_case_id": revised.case_id,
            }
        )
        linked_revision = revised.model_copy(
            update={
                "revision_parent_case_id": source.case_id,
                "revision_feedback": feedback,
            }
        )
        now = utc_now()
        with self._database.transaction() as connection:
            source_cursor = connection.execute(
                """
                UPDATE general_exploration_cases
                SET revision = ?, case_json = ?, updated_at = ?
                WHERE case_id = ? AND user_id = ? AND revision = ?
                """,
                (
                    linked_source.revision,
                    linked_source.model_dump_json(),
                    now,
                    linked_source.case_id,
                    self._active_user_id,
                    source.revision,
                ),
            )
            revised_cursor = connection.execute(
                """
                UPDATE general_exploration_cases SET case_json = ?, updated_at = ?
                WHERE case_id = ? AND user_id = ? AND revision = ?
                """,
                (
                    linked_revision.model_dump_json(),
                    now,
                    linked_revision.case_id,
                    self._active_user_id,
                    linked_revision.revision,
                ),
            )
            if source_cursor.rowcount != 1 or revised_cursor.rowcount != 1:
                raise GeneralExplorationConflict("实验版本已经变化，请刷新后再试。")
        return linked_revision

    @staticmethod
    def _reject_superseded(case: GeneralExperimentCase) -> None:
        if case.superseded_by_case_id is not None:
            raise GeneralExplorationConflict("该实验已按现场反馈重规划，请进入新版本继续。")

    def acquisition_plan(
        self,
        case_id: str,
        *,
        default_phyphox_device_saved: bool,
    ) -> GeneralAcquisitionPlan:
        case = self.get(case_id)
        self._reject_superseded(case)
        task = case.current_task
        if case.status != "collecting" or task is None:
            terminal_sources = tuple(
                GeneralAcquisitionSourcePlan(
                    source=source,
                    status="terminal",
                    counts_as_case_evidence=source != "public_analogue",
                    may_supply_user_phone_evidence=source != "public_analogue",
                    boundary_message="实验已经结束，不再接受新的现场或公开类比证据。",
                )
                for source in ("phyphox_live", "account_recording", "public_analogue")
            )
            terminal_sources = (
                *terminal_sources,
                GeneralAcquisitionSourcePlan(
                    source="protocol_simulator",
                    status="terminal",
                    counts_as_case_evidence=True,
                    may_supply_user_phone_evidence=False,
                    boundary_message="模拟排练已经结束，不再生成新的合成序列。",
                ),
            )
            return GeneralAcquisitionPlan(
                case_id=case.case_id,
                case_revision=case.revision,
                sources=terminal_sources,
                case_evidence_required=False,
            )

        requirements = {
            item.sensor: (item.metric_key, item.metric_unit)
            for item in case.protocol.sensors
            if item.sensor in task.sensors
        }
        selected_sources = set(case.protocol.selected_sources)
        recordings = []
        for recording in self._recordings.list_sensor_recordings():
            provenance_source = recording.upload.provenance.source
            acquisition_source = (
                "phyphox_live"
                if provenance_source == "phyphox_remote"
                else "phone_upload"
                if provenance_source in {"phone_upload", "file_import"}
                else "protocol_emulator"
                if provenance_source == "test_fixture"
                else None
            )
            requirement = requirements.get(recording.upload.sensor)
            if acquisition_source not in selected_sources or requirement is None:
                continue
            metric_key, metric_unit = requirement
            if not any(
                metric.key == metric_key and metric.unit == metric_unit
                for metric in recording.analysis.metrics
            ):
                continue
            if len(task.sensors) > 1 and (
                recording.upload.provenance.capture_group_id is None
                or recording.upload.provenance.clock_id is None
            ):
                continue
            recordings.append(recording)
        if len(task.sensors) > 1:
            groups: dict[tuple[str, str], set[SensorKind]] = {}
            for recording in recordings:
                key = (
                    str(recording.upload.provenance.capture_group_id),
                    str(recording.upload.provenance.clock_id),
                )
                groups.setdefault(key, set()).add(recording.upload.sensor)
            complete_groups = {
                key for key, sensors in groups.items() if sensors == set(task.sensors)
            }
            recordings = [
                recording
                for recording in recordings
                if (
                    str(recording.upload.provenance.capture_group_id),
                    str(recording.upload.provenance.clock_id),
                )
                in complete_groups
            ]
        candidate_ids = tuple(recording.session_id for recording in recordings)
        referenced_ids = self._referenced_recording_ids()
        recovery_ids = tuple(
            recording.session_id
            for recording in recordings
            if recording.session_id not in referenced_ids
            and recording.upload.provenance.general_case_id == case.case_id
            and recording.upload.provenance.general_task_id == task.task_id
        )
        if "phyphox_live" not in selected_sources:
            live_status = "not_authorized"
        elif default_phyphox_device_saved:
            live_status = "available"
        else:
            live_status = "setup_required"
        account_status = (
            "not_authorized"
            if selected_sources == {"protocol_emulator"}
            else "available"
            if candidate_ids
            else "no_matching_recording"
        )
        return GeneralAcquisitionPlan(
            case_id=case.case_id,
            case_revision=case.revision,
            task_id=task.task_id,
            required_sensors=task.sensors,
            sources=(
                GeneralAcquisitionSourcePlan(
                    source="phyphox_live",
                    status=live_status,
                    counts_as_case_evidence=True,
                    may_supply_user_phone_evidence=True,
                    boundary_message=(
                        "手机采集会先保存带 case/task lineage 的记录，再通过当前 revision 绑定。"
                    ),
                ),
                GeneralAcquisitionSourcePlan(
                    source="account_recording",
                    status=account_status,
                    candidate_recording_ids=candidate_ids,
                    recoverable_recording_ids=recovery_ids,
                    counts_as_case_evidence=True,
                    may_supply_user_phone_evidence=True,
                    boundary_message=(
                        "这里只列出当前协议来源、传感器、指标与同步合同都能通过的账号记录。"
                    ),
                ),
                GeneralAcquisitionSourcePlan(
                    source="public_analogue",
                    status="analogue_only",
                    counts_as_case_evidence=False,
                    may_supply_user_phone_evidence=False,
                    boundary_message=(
                        "公开真实数据只运行独立类比组件，不写入当前案例且 Gate C 计入 0。"
                    ),
                ),
                GeneralAcquisitionSourcePlan(
                    source="protocol_simulator",
                    status=(
                        "available"
                        if selected_sources == {"protocol_emulator"}
                        else "not_authorized"
                    ),
                    counts_as_case_evidence=True,
                    may_supply_user_phone_evidence=False,
                    boundary_message=(
                        "只为显式模拟排练协议生成合成 analyzer-contract 序列；"
                        "可完成软件闭环，但不是物理、手机或 Gate C 证据。"
                    ),
                ),
            ),
            case_evidence_required=True,
        )

    def prepare_recording_submission(
        self,
        case_id: str,
        request: GeneralRecordingMeasurementSubmit,
    ) -> PreparedGeneralTransition:
        case = self.get(case_id)
        self._reject_superseded(case)
        task = case.current_task
        if case.status != "collecting" or task is None:
            raise GeneralExplorationConflict("该通用探索已经结束，不能继续绑定记录。")
        if request.expected_revision != case.revision or request.task_id != task.task_id:
            raise GeneralExplorationConflict("任务或 revision 已变化，请刷新后重试。")
        if not request.controls_confirmed:
            raise GeneralExplorationValidation("绑定记录前必须确认本任务的控制条件。")
        if len(request.recording_ids) != len(task.sensors):
            raise GeneralExplorationValidation("当前任务要求每个传感器恰好绑定一条记录。")

        recordings_by_sensor = {}
        for recording_id in request.recording_ids:
            try:
                recording = self._recordings.get_sensor_recording(recording_id)
            except KeyError as exc:
                raise GeneralExplorationValidation(str(exc)) from exc
            if recording.upload.sensor in recordings_by_sensor:
                raise GeneralExplorationValidation("一次任务不能重复绑定同一传感器。")
            recordings_by_sensor[recording.upload.sensor] = recording
        if set(recordings_by_sensor) != set(task.sensors):
            raise GeneralExplorationValidation("记录传感器集合与当前任务不一致。")

        selected_sources = set(case.protocol.selected_sources)
        if not selected_sources <= {"phone_upload", "phyphox_live"}:
            raise GeneralExplorationValidation(
                "该提交接口只接受账号上传或 phyphox 实时协议记录。"
            )
        evidence = []
        for sensor in task.sensors:
            recording = recordings_by_sensor[sensor]
            if recording.upload.provenance.source == "phyphox_remote":
                acquisition_source: Literal["phyphox_live", "phone_upload"] = (
                    "phyphox_live"
                )
            elif recording.upload.provenance.source in {"phone_upload", "file_import"}:
                acquisition_source = "phone_upload"
            else:
                raise GeneralExplorationValidation(
                    "测试夹具或公开回放不能通过账号记录接口冒充现场物理证据。"
                )
            if acquisition_source not in selected_sources:
                raise GeneralExplorationValidation("记录来源不在当前冻结协议的允许范围内。")
            try:
                acquisition = self._acquisitions.load(
                    GeneralAcquisitionReference(
                        source=acquisition_source,
                        recording_id=recording.session_id,
                    )
                )
                evidence.append(
                    bind_general_evidence(
                        case.protocol,
                        condition_id=task.condition_id,
                        acquisition=acquisition,
                    )
                )
            except ValueError as exc:
                raise GeneralExplorationValidation(str(exc)) from exc
        try:
            return prepare_general_measurement(
                case,
                GeneralMeasurementSubmission(
                    case_id=case.case_id,
                    task_id=task.task_id,
                    expected_revision=case.revision,
                    evidence=tuple(evidence),
                ),
            )
        except ValueError as exc:
            raise GeneralExplorationValidation(str(exc)) from exc

    def replay_committed_recording_submission(
        self,
        case_id: str,
        request: GeneralRecordingMeasurementSubmit,
    ) -> GeneralExperimentCase | None:
        """Return current state for an exact, already committed recording retry.

        This supports a client that lost the first HTTP response. It never accepts
        a different recording set, an unconfirmed request, or a revision that was
        not actually advanced.
        """

        if not request.controls_confirmed:
            return None
        case = self.get(case_id)
        if case.superseded_by_case_id is not None:
            return None
        if request.expected_revision >= case.revision:
            return None
        task = next(
            (item for item in case.completed_tasks if item.task_id == request.task_id),
            None,
        )
        if task is None:
            return None
        evidence_by_id = {item.evidence_id: item for item in case.evidence}
        committed_recordings = {
            evidence_by_id[evidence_id].lineage.recording_id
            for evidence_id in task.output_evidence_ids
        }
        if committed_recordings != set(request.recording_ids):
            return None
        return case

    def prepare_simulated_submission(
        self,
        case_id: str,
        request: GeneralSimulationMeasurementRequest,
    ) -> PreparedGeneralTransition:
        case = self.get(case_id)
        self._reject_superseded(case)
        try:
            evidence = build_general_simulated_evidence(case, request)
            task = case.current_task
            if task is None:  # pragma: no cover - builder already guards terminal state
                raise GeneralExplorationConflict("该模拟排练已经结束。")
            return prepare_general_measurement(
                case,
                GeneralMeasurementSubmission(
                    case_id=case.case_id,
                    task_id=task.task_id,
                    expected_revision=case.revision,
                    evidence=evidence,
                ),
            )
        except GeneralExplorationConflict:
            raise
        except ValueError as exc:
            message = str(exc)
            if "stale or foreign" in message:
                raise GeneralExplorationConflict("任务或 revision 已变化，请刷新后重试。") from exc
            raise GeneralExplorationValidation(message) from exc

    def validate_phyphox_capture_request(
        self,
        case_id: str,
        request: GeneralPhyphoxCaptureRequest,
    ) -> GeneralExperimentCase:
        case = self.get(case_id)
        self._reject_superseded(case)
        task = case.current_task
        if case.status != "collecting" or task is None:
            raise GeneralExplorationConflict("该通用探索已经结束，不能继续采集。")
        if request.expected_revision != case.revision or request.task_id != task.task_id:
            raise GeneralExplorationConflict("任务或 revision 已变化，请刷新后重试。")
        if not request.controls_confirmed:
            raise GeneralExplorationValidation("采集前必须确认本任务的控制条件。")
        if not request.privacy_acknowledged:
            raise GeneralExplorationValidation("请先确认可信局域网与传感器隐私提示。")
        if "phyphox_live" not in case.protocol.selected_sources:
            raise GeneralExplorationValidation("当前冻结协议没有授权 phyphox 实时来源。")
        if len(task.sensors) != 1:
            raise GeneralExplorationValidation(
                "当前实时桥只支持顺序单传感器任务；多传感器同步任务尚未开放。"
            )
        if task.sensors[0] == "bluetooth":
            raise GeneralExplorationValidation("Bluetooth 仍只支持能力识别。")
        return case

    def validate_phyphox_synchronized_capture_request(
        self,
        case_id: str,
        request: GeneralPhyphoxCaptureRequest,
    ) -> GeneralExperimentCase:
        case = self.get(case_id)
        self._reject_superseded(case)
        task = case.current_task
        if case.status != "collecting" or task is None:
            raise GeneralExplorationConflict("该通用探索已经结束，不能继续采集。")
        if request.expected_revision != case.revision or request.task_id != task.task_id:
            raise GeneralExplorationConflict("任务或 revision 已变化，请刷新后重试。")
        if not request.controls_confirmed:
            raise GeneralExplorationValidation("采集前必须确认本任务的控制条件。")
        if not request.privacy_acknowledged:
            raise GeneralExplorationValidation("请先确认可信局域网与传感器隐私提示。")
        if "phyphox_live" not in case.protocol.selected_sources:
            raise GeneralExplorationValidation("当前冻结协议没有授权 phyphox 实时来源。")
        if case.protocol.alignment != "simultaneous" or not 2 <= len(task.sensors) <= 3:
            raise GeneralExplorationValidation("该路由只接受 2 到 3 个传感器的同步任务。")
        if "bluetooth" in task.sensors:
            raise GeneralExplorationValidation("Bluetooth 仍只支持能力识别。")
        return case

    def save_committed(
        self,
        case: GeneralExperimentCase,
        *,
        expected_revision: int,
    ) -> None:
        case = GeneralExperimentCase.model_validate(case.model_dump(mode="python"))
        if case.revision != expected_revision + 1:
            raise GeneralExplorationValidation("一次提交必须且只能增加一个 revision。")
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE general_exploration_cases
                SET revision = ?, case_json = ?, updated_at = ?
                WHERE case_id = ? AND user_id = ? AND revision = ?
                """,
                (
                    case.revision,
                    case.model_dump_json(),
                    utc_now(),
                    case.case_id,
                    self._active_user_id,
                    expected_revision,
                ),
            )
        if cursor.rowcount != 1:
            raise GeneralExplorationConflict("探索已被其他请求更新，请刷新后重试。")

    def recording_is_referenced(self, recording_id: str) -> bool:
        return recording_id in self._referenced_recording_ids()

    def _referenced_recording_ids(self) -> set[str]:
        rows = self._database.fetch_all(
            "SELECT case_json FROM general_exploration_cases WHERE user_id = ?",
            (self._active_user_id,),
        )
        return {
            evidence.lineage.recording_id
            for row in rows
            for evidence in GeneralExperimentCase.model_validate_json(row["case_json"]).evidence
        }


general_exploration_store = GeneralExplorationStore(
    database,
    session_store,
    user_id=None,
)
