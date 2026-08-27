from __future__ import annotations

import argparse
import asyncio
import math
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pocketlab.active_experiment_design import (
    ActiveExperimentDesignSpec,
    list_active_experiment_design_specs,
)
from pocketlab.agent import (
    DiagnosticProposalUnavailable,
    get_active_model_name,
    run_diagnostic_finalization_agent,
    run_diagnostic_intake_agent,
    run_diagnostic_measurement_agent,
    run_evidence_workbench,
    run_experiment_agent,
)
from pocketlab.agent_runtime import AgentRuntimeError, agent_runtime_http_status
from pocketlab.analyzers import analyze_sensor_recording, sensor_capabilities
from pocketlab.auth import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    Account,
    InvalidCredentialsError,
    UsernameTakenError,
    auth_store,
    cookie_secure,
    user_context,
)
from pocketlab.capability_checks import (
    SensorCapabilityCheck,
    build_sensor_capability_check,
)
from pocketlab.diagnostic_evidence import get_diagnostic_recording
from pocketlab.diagnostics import build_diagnostic_retest_request, diagnostic_case_store
from pocketlab.evidence_workbench import (
    EvidenceWorkbenchHistoryItem,
    EvidenceWorkbenchNoteUpdate,
    EvidenceWorkbenchReport,
    build_evidence_audit,
    build_evidence_presentation,
    deterministic_workbench_answer,
    evidence_workbench_store,
    workbench_report_markdown,
)
from pocketlab.experiment_protocols import list_experiment_protocols
from pocketlab.exploration_history import (
    ExplorationHistoryItem,
    PublicExplorationHistoryDetail,
    exploration_history_store,
    general_exploration_history_item,
    investigation_history_item,
)
from pocketlab.explorations import list_explorations
from pocketlab.general_exploration_models import GeneralSensorCapabilityContract
from pocketlab.general_exploration_protocol import list_general_sensor_capabilities
from pocketlab.general_exploration_reasoner import GeneralReasonerUnavailable
from pocketlab.general_exploration_service import (
    advance_general_exploration,
    advance_general_simulation,
    decide_general_reasoning_checkpoint,
)
from pocketlab.general_exploration_state import (
    GeneralExperimentCase,
    GeneralReasoningCheckpointDecision,
)
from pocketlab.general_exploration_store import (
    GeneralAcquisitionPlan,
    GeneralExplorationCaseCreate,
    GeneralExplorationCaseHistoryItem,
    GeneralExplorationConflict,
    GeneralExplorationNotFound,
    GeneralExplorationValidation,
    GeneralPhyphoxCaptureMetadata,
    GeneralPhyphoxCaptureRequest,
    GeneralPhyphoxCaptureResponse,
    GeneralPhyphoxSynchronizedCaptureResponse,
    GeneralRecordingMeasurementSubmit,
    build_general_reality_evidence_reuse_audit,
    general_exploration_store,
)
from pocketlab.general_public_components import (
    GeneralPublicComponentCatalog,
    GeneralPublicComponentRunRequest,
    GeneralPublicComponentRunResult,
    GeneralPublicComponentValidation,
    build_general_public_component_catalog,
    run_general_public_component,
)
from pocketlab.general_question_compiler import (
    GENERAL_CLARIFICATION_CODES,
    GeneralQuestionCompileRequest,
    GeneralQuestionCompileResult,
    compile_general_question,
)
from pocketlab.general_readiness import (
    GeneralExplorationReadiness,
    get_general_exploration_readiness,
)
from pocketlab.general_simulation import (
    GeneralSimulationCaptureMetadata,
    GeneralSimulationMeasurementRequest,
    GeneralSimulationMeasurementResponse,
)
from pocketlab.investigation_models import (
    ExperimentProtocol,
    InvestigationCaptureMetadata,
    InvestigationCase,
    InvestigationCaseCreate,
    InvestigationCaseHistoryItem,
    InvestigationMeasurementSubmit,
    InvestigationPhyphoxCaptureRequest,
    InvestigationPhyphoxCaptureResponse,
    RecordingRef,
)
from pocketlab.investigation_router import (
    InvestigationRouteRecommendation,
    InvestigationRouteRequest,
    route_investigation_with_model,
)
from pocketlab.investigation_service import advance_investigation
from pocketlab.investigations import (
    InvestigationConflict,
    InvestigationNotFound,
    InvestigationValidation,
    investigation_store,
)
from pocketlab.model_profiles import (
    ENVIRONMENT_PROFILE_ID,
    ModelCapabilityProbe,
    ModelProfileCatalog,
    ModelProfileCreate,
    ModelProfileError,
    ModelProfileNotFound,
    ModelProfileSummary,
    ModelProfileUpdate,
    ModelSecretUnavailable,
    environment_model_configuration,
    model_profile_store,
    probe_model_compatibility,
)
from pocketlab.model_run_control import (
    ModelRunDecisionRequest,
    decide_model_run,
    get_model_run_status,
    model_run_context,
    validate_model_run_id,
)
from pocketlab.phyphox import (
    PhyphoxError,
    PhyphoxUrlError,
    capture_phyphox_acceleration,
    capture_phyphox_sensor,
    capture_phyphox_sensors,
    probe_phyphox,
)
from pocketlab.public_light_exploration import (
    PublicLightExplorationUnavailable,
    run_public_light_exploration,
)
from pocketlab.public_light_models import (
    PublicLightExploreRequest,
    PublicLightExploreResult,
)
from pocketlab.public_pressure_agent_models import (
    PublicPressureExploreRequest,
    PublicPressureExploreResult,
)
from pocketlab.public_pressure_exploration import (
    PublicPressureExplorationUnavailable,
    run_public_pressure_exploration,
)
from pocketlab.public_replay_dataset import (
    PublicReplayCatalogItem,
    evaluate_public_replay_dataset,
    get_public_replay_dataset,
    list_public_replay_catalog,
    read_public_replay_recording,
    verify_public_source_files,
)
from pocketlab.public_sensor_agent_models import (
    PublicSensorExploreRequest,
    PublicSensorExploreResult,
)
from pocketlab.public_sensor_exploration import (
    PublicSensorExplorationUnavailable,
    run_public_sensor_exploration,
)
from pocketlab.reality_feedback import RealityFeedbackRequest, revised_context
from pocketlab.runtime_audit import AgentRunAuditCatalog, agent_run_audit_store
from pocketlab.schemas import (
    AccelerationSample,
    AgentRunRequest,
    AgentRunResponse,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthSessionResponse,
    AuthStatusResponse,
    AuthUser,
    DiagnosticAgentResponse,
    DiagnosticCase,
    DiagnosticCaseCreate,
    DiagnosticCaseHistoryItem,
    DiagnosticCaseSnapshot,
    DiagnosticCheckpointDecision,
    DiagnosticMeasurementSubmit,
    DiagnosticMeasurementTask,
    DiagnosticPublicReplaySubmit,
    DiagnosticRecordingSubmit,
    DiagnosticSensorTaskResponse,
    DiagnosticTaskPhyphoxRequest,
    EvidenceWorkbenchRequest,
    ExplorationTemplate,
    LocalProfile,
    LocalProfileUpdate,
    MobileTaskResponse,
    PhyphoxCaptureMetadata,
    PhyphoxCaptureRequest,
    PhyphoxConnectionRequest,
    PhyphoxDeviceSaveRequest,
    PhyphoxDeviceSaveResponse,
    PhyphoxProbeResponse,
    PhyphoxTaskResponse,
    PocketLabSettings,
    SessionCreated,
    SessionHistoryItem,
    SessionRecord,
    SessionUpload,
    TaskSampleResponse,
    TaskSampleUpload,
    VibrationAnalysis,
)
from pocketlab.sensor_models import (
    PhyphoxSensorCaptureMetadata,
    PhyphoxSensorCaptureRequest,
    PhyphoxSensorCaptureResponse,
    SensorCapability,
    SensorChannelDefinition,
    SensorKind,
    SensorProvenance,
    SensorRecordingCreated,
    SensorRecordingHistoryItem,
    SensorRecordingRecord,
    SensorRecordingUpload,
    SensorSample,
)
from pocketlab.settings import settings_store
from pocketlab.store import session_store
from pocketlab.work_summaries import (
    WorkSummary,
    diagnostic_work_summary,
    general_work_summary,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
PUBLIC_REPLAY_DIR = Path(__file__).resolve().parent.parent / "datasets" / "public"

app = FastAPI(title="PocketLab Agent", version="0.9.0")


@app.exception_handler(GeneralReasonerUnavailable)
async def general_reasoner_unavailable_handler(
    _request: Request,
    exc: GeneralReasonerUnavailable,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "code": "general_reasoner_unavailable",
                "message": str(exc),
                "retryable": True,
            }
        },
    )


@app.exception_handler(AgentRuntimeError)
async def agent_runtime_error_handler(
    _request: Request,
    exc: AgentRuntimeError,
) -> JSONResponse:
    status_code = agent_runtime_http_status(exc.kind)
    headers = {"Retry-After": "2"} if exc.retryable else None
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "detail": {
                "code": exc.kind,
                "message": str(exc),
                "retryable": exc.retryable,
            }
        },
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.middleware("http")
async def bind_authenticated_user(request: Request, call_next):
    account = auth_store.resolve_session(request.cookies.get(SESSION_COOKIE_NAME))
    request.state.account = account
    path = request.url.path
    protected_api = path.startswith(("/api/v1/", "/api/v2/"))
    if protected_api and not path.startswith("/api/v1/auth/") and account is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "请先登录 PocketLab。"},
        )
    if (path == "/app" or path.startswith("/app/")) and account is None:
        return RedirectResponse("/login", status_code=303)
    if account is None:
        return await call_next(request)
    try:
        model_run_id = validate_model_run_id(request.headers.get("X-PocketLab-Model-Run"))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    with user_context(account.user_id), model_run_context(model_run_id, path):
        return await call_next(request)


def _auth_user(account: Account) -> AuthUser:
    return AuthUser(
        user_id=account.user_id,
        username=account.username,
        display_name=account.display_name,
        created_at=account.created_at,
        updated_at=account.updated_at,
    )


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
        secure=cookie_secure(),
        httponly=True,
        samesite="lax",
    )


@app.get("/", include_in_schema=False)
def web_root(request: Request) -> RedirectResponse:
    destination = "/app" if request.state.account is not None else "/login"
    return RedirectResponse(destination, status_code=303)


@app.get("/login", include_in_schema=False)
def login_page(request: Request) -> Response:
    if request.state.account is not None:
        return RedirectResponse("/app", status_code=303)
    return FileResponse(WEB_DIR / "login.html")


@app.get("/app", include_in_schema=False)
@app.get("/app/{page_path:path}", include_in_schema=False)
def web_app(page_path: str = "") -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/v1/auth/status", response_model=AuthStatusResponse)
def auth_status() -> AuthStatusResponse:
    return AuthStatusResponse(legacy_data_available=auth_store.local_data_available())


@app.post("/api/v1/auth/register", response_model=AuthSessionResponse)
def register_account(request: AuthRegisterRequest, response: Response) -> AuthSessionResponse:
    try:
        result = auth_store.register(
            username=request.username,
            password=request.password.get_secret_value(),
            display_name=request.display_name,
            claim_local_data=request.claim_local_data,
        )
    except UsernameTakenError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    token = auth_store.create_session(result.account.user_id)
    _set_session_cookie(response, token)
    return AuthSessionResponse(
        user=_auth_user(result.account),
        claimed_local_data=result.claimed_local_data,
    )


@app.post("/api/v1/auth/login", response_model=AuthSessionResponse)
def login_account(request: AuthLoginRequest, response: Response) -> AuthSessionResponse:
    try:
        account = auth_store.authenticate(
            request.username,
            request.password.get_secret_value(),
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    token = auth_store.create_session(account.user_id)
    _set_session_cookie(response, token)
    return AuthSessionResponse(user=_auth_user(account))


@app.get("/api/v1/auth/me", response_model=AuthUser)
def current_account(request: Request) -> AuthUser:
    account = request.state.account
    if account is None:
        raise HTTPException(status_code=401, detail="请先登录 PocketLab。")
    return _auth_user(account)


@app.post("/api/v1/auth/logout")
def logout_account(request: Request, response: Response) -> dict[str, bool]:
    auth_store.revoke_session(request.cookies.get(SESSION_COOKIE_NAME))
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        secure=cookie_secure(),
        httponly=True,
        samesite="lax",
    )
    return {"logged_out": True}


@app.get("/api/v1/model-runs/{run_id}")
async def model_run_status(run_id: str) -> dict[str, object]:
    try:
        return get_model_run_status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/model-runs/{run_id}/decision")
async def model_run_decision(
    run_id: str,
    request: ModelRunDecisionRequest,
) -> dict[str, object]:
    try:
        return decide_model_run(run_id, request.decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": get_active_model_name()}


@app.post("/api/v1/sessions", response_model=SessionCreated)
def create_session(upload: SessionUpload) -> SessionCreated:
    try:
        session = session_store.create(upload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SessionCreated(
        session_id=session.session_id,
        label=session.upload.label,
        analysis=session.analysis,
        created_at=session.created_at,
    )


@app.get("/api/v1/sessions", response_model=list[SessionHistoryItem])
def list_sessions() -> list[SessionHistoryItem]:
    return [
        SessionHistoryItem(
            session_id=session.session_id,
            label=session.upload.label,
            device=session.upload.device,
            notes=session.upload.notes,
            sample_count=len(session.upload.samples),
            analysis=session.analysis,
            created_at=session.created_at,
        )
        for session in session_store.list()
    ]


@app.get("/api/v1/sessions/{session_id}", response_model=SessionRecord)
def get_session(session_id: str) -> SessionRecord:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionRecord(
        session_id=session.session_id,
        upload=session.upload,
        analysis=session.analysis,
        created_at=session.created_at,
    )


@app.get("/api/v1/sessions/{session_id}/analysis", response_model=VibrationAnalysis)
def get_analysis(session_id: str) -> VibrationAnalysis:
    try:
        return session_store.get(session_id).analysis
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v2/sensors/capabilities", response_model=list[SensorCapability])
def get_sensor_capabilities() -> list[SensorCapability]:
    return sensor_capabilities()


@app.post("/api/v2/recordings", response_model=SensorRecordingCreated)
def create_sensor_recording(upload: SensorRecordingUpload) -> SensorRecordingCreated:
    if upload.provenance.source in {"public_replay", "phyphox_remote"}:
        raise HTTPException(
            status_code=422,
            detail=(
                "公开回放或 phyphox 实时来源只能由服务端创建，且只能通过服务端校验的"
                "采集链写入，不能由普通上传声明。"
            ),
        )
    try:
        recording = session_store.create_sensor_recording(upload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SensorRecordingCreated(
        session_id=recording.session_id,
        label=recording.upload.label,
        sensor=recording.upload.sensor,
        analysis=recording.analysis,
        created_at=recording.created_at,
    )


@app.get("/api/v2/recordings", response_model=list[SensorRecordingHistoryItem])
def list_sensor_recordings() -> list[SensorRecordingHistoryItem]:
    return [
        SensorRecordingHistoryItem(
            session_id=recording.session_id,
            label=recording.upload.label,
            device=recording.upload.device,
            sensor=recording.upload.sensor,
            sample_count=len(recording.upload.samples),
            provenance=recording.upload.provenance,
            analysis=recording.analysis,
            created_at=recording.created_at,
        )
        for recording in session_store.list_sensor_recordings()
    ]


@app.get("/api/v2/public-replays", response_model=list[PublicReplayCatalogItem])
def list_public_replays() -> list[PublicReplayCatalogItem]:
    try:
        return list_public_replay_catalog(PUBLIC_REPLAY_DIR)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _is_local_request(request: Request) -> bool:
    client_host = request.client.host if request.client is not None else ""
    return client_host in {"127.0.0.1", "::1", "localhost", "testclient"}


@app.post(
    "/api/v2/public-replays/light/explore",
    response_model=PublicLightExploreResult,
)
async def explore_public_light(
    http_request: Request,
    request: PublicLightExploreRequest,
) -> PublicLightExploreResult:
    if request.privacy_acknowledged and not _is_local_request(http_request):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "local_only_public_replay",
                "message": (
                    "该请求可能读取保留行为光照特征的公开序列，只允许从 PocketLab 本机界面运行。"
                ),
            },
        )
    try:
        result = await run_public_light_exploration(
            request,
            root=PUBLIC_REPLAY_DIR,
        )
        exploration_history_store.save_public(result)
        return result
    except PublicLightExplorationUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": exc.reason,
                "message": "公开 Light 来源或确定性工具未能通过安全校验。",
            },
        ) from exc


@app.post(
    "/api/v2/public-replays/pressure/explore",
    response_model=PublicPressureExploreResult,
)
async def explore_public_pressure(
    http_request: Request,
    request: PublicPressureExploreRequest,
) -> PublicPressureExploreResult:
    if request.privacy_acknowledged and not _is_local_request(http_request):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "local_only_public_replay",
                "message": (
                    "该请求将读取仅获准本地评测的公开 Pressure 序列，"
                    "只允许从 PocketLab 本机界面运行。"
                ),
            },
        )
    try:
        result = await run_public_pressure_exploration(
            request,
            root=PUBLIC_REPLAY_DIR,
        )
        exploration_history_store.save_public(result)
        return result
    except PublicPressureExplorationUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": exc.reason,
                "message": "公开 Pressure 来源或确定性工具未能通过安全校验。",
            },
        ) from exc


@app.post(
    "/api/v2/public-replays/sensors/{sensor}/explore",
    response_model=PublicSensorExploreResult,
)
async def explore_public_sensor(
    sensor: str,
    http_request: Request,
    request: PublicSensorExploreRequest,
) -> PublicSensorExploreResult:
    if sensor != request.sensor:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "sensor_path_body_mismatch",
                "message": "URL 中的传感器必须与请求体 sensor 完全一致。",
            },
        )
    if request.privacy_acknowledged and not _is_local_request(http_request):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "local_only_public_replay",
                "message": (
                    "该请求将读取只获准本地评测的公开传感器序列，只允许从 PocketLab 本机界面运行。"
                ),
            },
        )
    try:
        result = await run_public_sensor_exploration(
            request,
            root=PUBLIC_REPLAY_DIR,
        )
        exploration_history_store.save_public(result)
        return result
    except PublicSensorExplorationUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": exc.reason,
                "message": "公开传感器来源、专用协议或确定性工具未通过安全校验。",
            },
        ) from exc


@app.post(
    "/api/v2/public-replays/{dataset_id}/recordings/{recording_id}/import",
    response_model=SensorRecordingCreated,
)
def import_public_replay_recording(
    dataset_id: str,
    recording_id: str,
) -> SensorRecordingCreated:
    try:
        pack_dir, manifest = get_public_replay_dataset(PUBLIC_REPLAY_DIR, dataset_id)
        recording = next(
            (item for item in manifest.recordings if item.recording_id == recording_id),
            None,
        )
        if recording is None:
            raise KeyError(f"Unknown public replay recording: {recording_id}")
        if "account_import" not in manifest.privacy_review.allowed_operations:
            raise ValueError(
                "该公开数据含需显式确认的行为光照节律，只允许无持久化本地回放，不能导入账号历史。"
            )
        verify_public_source_files(pack_dir, manifest)
        verification = evaluate_public_replay_dataset(pack_dir, replay_repeat=1)
        if not verification["source_validated"]:
            raise ValueError("public replay pack failed full source validation")
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        if (
            upload.provenance.source != "public_replay"
            or upload.provenance.public_dataset_id != manifest.dataset_id
            or upload.provenance.public_recording_id != recording.recording_id
        ):
            raise ValueError("public replay provenance did not survive source validation")
        stored = session_store.create_sensor_recording(upload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SensorRecordingCreated(
        session_id=stored.session_id,
        label=stored.upload.label,
        sensor=stored.upload.sensor,
        analysis=stored.analysis,
        created_at=stored.created_at,
    )


@app.get("/api/v2/recordings/{session_id}", response_model=SensorRecordingRecord)
def get_sensor_recording(session_id: str) -> SensorRecordingRecord:
    try:
        recording = session_store.get_sensor_recording(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SensorRecordingRecord(
        session_id=recording.session_id,
        upload=recording.upload,
        analysis=recording.analysis,
        created_at=recording.created_at,
    )


@app.delete("/api/v2/recordings/{session_id}")
def delete_sensor_recording(session_id: str) -> dict[str, str]:
    try:
        session_store.get_sensor_recording(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if investigation_store.recording_is_referenced(session_id):
        raise HTTPException(
            status_code=409,
            detail="该记录已绑定到实验调查证据；请先删除对应调查，避免破坏审计链。",
        )
    if general_exploration_store.recording_is_referenced(session_id):
        raise HTTPException(
            status_code=409,
            detail="该记录已绑定到通用探索证据；请先删除对应探索，避免破坏审计链。",
        )
    session_store.delete(session_id)
    return {"deleted_session_id": session_id}


@app.get("/api/v2/experiment-protocols", response_model=list[ExperimentProtocol])
def get_experiment_protocols() -> list[ExperimentProtocol]:
    return list_experiment_protocols()


@app.get(
    "/api/v2/active-experiment-design-specs",
    response_model=list[ActiveExperimentDesignSpec],
)
def get_active_experiment_design_specs() -> list[ActiveExperimentDesignSpec]:
    """Expose frozen candidate-generation contracts; this does not claim execution readiness."""

    return list_active_experiment_design_specs()


@app.get(
    "/api/v2/general-exploration-capabilities",
    response_model=list[GeneralSensorCapabilityContract],
)
def get_general_exploration_capabilities() -> list[GeneralSensorCapabilityContract]:
    return list_general_sensor_capabilities()


@app.get(
    "/api/v2/general-exploration-readiness",
    response_model=GeneralExplorationReadiness,
)
def read_general_exploration_readiness() -> GeneralExplorationReadiness:
    return get_general_exploration_readiness()


@app.post(
    "/api/v2/general-explorations/compile",
    response_model=GeneralQuestionCompileResult,
)
async def compile_general_exploration_question(
    request: GeneralQuestionCompileRequest,
) -> GeneralQuestionCompileResult:
    """Compile untrusted text and issue only a hash-bound, user-scoped receipt."""

    has_clarification_resolution = bool(
        request.clarification_answers
        or request.condition_resolution is not None
        or request.mechanism_resolution is not None
    )
    if has_clarification_resolution != (request.clarification_receipt_id is not None):
        raise HTTPException(
            status_code=422,
            detail=(
                "补充澄清必须携带上一轮的一次性凭证；凭证也不能脱离结构化补充单独使用。"
            ),
        )
    reservation = None
    if has_clarification_resolution:
        try:
            reservation = general_exploration_store.reserve_clarification_receipt(request)
        except GeneralExplorationValidation as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except GeneralExplorationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        result = await compile_general_question(request)
    except BaseException:
        if reservation is not None:
            general_exploration_store.release_clarification_receipt(reservation)
        raise

    finite_result_codes = set(result.blocker_codes)
    unactionable_agent_clarification = (
        result.status == "needs_clarification"
        and result.source == "bounded_agent"
        and result.runtime.fallback_reason == "none"
        and (
            not finite_result_codes
            or not finite_result_codes <= set(GENERAL_CLARIFICATION_CODES)
        )
    )
    if unactionable_agent_clarification:
        result = GeneralQuestionCompileResult.model_validate(
            {
                **result.model_dump(mode="python"),
                "status": "rejected",
                "clarification_questions": (),
                "clarification_receipt": None,
            }
        )

    if reservation is not None and (
        result.source == "deterministic_fallback" or result.runtime.fallback_reason != "none"
    ):
        general_exploration_store.release_clarification_receipt(reservation)
        return result

    clarification_receipt = None
    finite_clarification_codes = set(result.blocker_codes)
    should_issue_clarification_receipt = (
        result.status == "needs_clarification"
        and result.source in {"bounded_agent", "server_policy"}
        and result.runtime.fallback_reason == "none"
        and bool(finite_clarification_codes)
        and finite_clarification_codes <= set(GENERAL_CLARIFICATION_CODES)
    )
    if should_issue_clarification_receipt:
        try:
            clarification_receipt = general_exploration_store.issue_clarification_receipt(
                request,
                result,
            )
        except GeneralExplorationValidation as exc:
            if reservation is not None:
                general_exploration_store.release_clarification_receipt(reservation)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except GeneralExplorationConflict as exc:
            if reservation is not None:
                general_exploration_store.release_clarification_receipt(reservation)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    if reservation is not None:
        try:
            general_exploration_store.consume_clarification_receipt(reservation, request)
        except GeneralExplorationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if clarification_receipt is not None:
        return GeneralQuestionCompileResult.model_validate(
            {**result.model_dump(mode="python"), "clarification_receipt": clarification_receipt}
        )
    if result.status == "draft_ready" and result.source == "bounded_agent":
        receipt = general_exploration_store.issue_compilation_receipt(result)
        return GeneralQuestionCompileResult.model_validate(
            {**result.model_dump(mode="python"), "receipt": receipt}
        )
    return result


@app.post("/api/v2/general-explorations", response_model=GeneralExperimentCase)
def create_general_exploration(
    request: GeneralExplorationCaseCreate,
) -> GeneralExperimentCase:
    try:
        return general_exploration_store.create(request)
    except GeneralExplorationValidation as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "blocker_codes": list(exc.blocker_codes),
                "user_messages": list(exc.user_messages),
            },
        ) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(
    "/api/v2/general-explorations",
    response_model=list[GeneralExplorationCaseHistoryItem],
)
def list_general_explorations() -> list[GeneralExplorationCaseHistoryItem]:
    return general_exploration_store.list()


@app.get("/api/v2/general-explorations/{case_id}", response_model=GeneralExperimentCase)
def get_general_exploration(case_id: str) -> GeneralExperimentCase:
    try:
        return general_exploration_store.get(case_id)
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v2/general-explorations/{case_id}/acquisition-plan",
    response_model=GeneralAcquisitionPlan,
)
def get_general_exploration_acquisition_plan(case_id: str) -> GeneralAcquisitionPlan:
    try:
        device_saved = settings_store.get().default_phyphox_device is not None
        return general_exploration_store.acquisition_plan(
            case_id,
            default_phyphox_device_saved=device_saved,
        )
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(
    "/api/v2/general-explorations/{case_id}/public-components",
    response_model=GeneralPublicComponentCatalog,
)
def get_general_public_components(case_id: str) -> GeneralPublicComponentCatalog:
    """List source-validated public analogues without treating them as case evidence."""

    try:
        case = general_exploration_store.get(case_id)
        return build_general_public_component_catalog(case, root=PUBLIC_REPLAY_DIR)
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (GeneralPublicComponentValidation, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "public_component_catalog_unavailable",
                "message": str(exc),
            },
        ) from exc


@app.post(
    "/api/v2/general-explorations/{case_id}/public-components/run",
    response_model=GeneralPublicComponentRunResult,
)
async def run_general_public_component_for_case(
    case_id: str,
    http_request: Request,
    request: GeneralPublicComponentRunRequest,
) -> GeneralPublicComponentRunResult:
    """Run one separate public Agent component; never mutate the general case."""

    if request.privacy_acknowledged and not _is_local_request(http_request):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "local_only_public_component",
                "message": "公开数据组件只允许从 PocketLab 本机界面运行。",
            },
        )
    try:
        case = general_exploration_store.get(case_id)
        if request.expected_revision != case.revision:
            raise GeneralExplorationConflict(
                f"revision 已变化：当前为 {case.revision}，请求为 {request.expected_revision}。"
            )
        execution = await run_general_public_component(
            case,
            request,
            root=PUBLIC_REPLAY_DIR,
        )
        exploration_history_store.save_public(execution.public_result)
        return execution.result
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeneralPublicComponentValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (
        PublicLightExplorationUnavailable,
        PublicPressureExplorationUnavailable,
        PublicSensorExplorationUnavailable,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "public_component_run_unavailable",
                "message": str(exc),
            },
        ) from exc


@app.delete("/api/v2/general-explorations/{case_id}")
def delete_general_exploration(case_id: str) -> dict[str, str]:
    try:
        general_exploration_store.delete(case_id)
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted_general_exploration_id": case_id}


@app.post(
    "/api/v2/general-explorations/{case_id}/measurements",
    response_model=GeneralExperimentCase,
)
async def submit_general_exploration_measurement(
    case_id: str,
    request: GeneralRecordingMeasurementSubmit,
) -> GeneralExperimentCase:
    try:
        replayed = general_exploration_store.replay_committed_recording_submission(
            case_id,
            request,
        )
        if replayed is not None:
            return replayed
        return await advance_general_exploration(
            general_exploration_store,
            case_id,
            request,
        )
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeneralExplorationValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/v2/general-explorations/{case_id}/simulate",
    response_model=GeneralSimulationMeasurementResponse,
)
async def simulate_general_exploration_measurement(
    case_id: str,
    request: GeneralSimulationMeasurementRequest,
) -> GeneralSimulationMeasurementResponse:
    """Run one labelled analyzer-contract rehearsal step through the normal Agent loop."""

    try:
        before = general_exploration_store.get(case_id)
        task = before.current_task
        if task is None:
            raise GeneralExplorationConflict("该模拟排练已经结束。")
        previous_evidence_ids = {item.evidence_id for item in before.evidence}
        updated = await advance_general_simulation(
            general_exploration_store,
            case_id,
            request,
        )
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeneralExplorationValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    evidence_ids = {
        item.evidence_id
        for item in updated.evidence
        if item.evidence_id not in previous_evidence_ids
        and item.lineage.source == "protocol_emulator"
    }
    evidence = tuple(item for item in updated.evidence if item.evidence_id in evidence_ids)
    return GeneralSimulationMeasurementResponse(
        case=updated,
        evidence=evidence,
        simulation=GeneralSimulationCaptureMetadata(
            profile=request.profile,
            sensors=task.sensors,
            recording_ids=tuple(item.lineage.recording_id for item in evidence),
        ),
    )


@app.post(
    "/api/v2/general-explorations/{case_id}/reality-feedback",
    response_model=GeneralExperimentCase,
)
async def revise_general_exploration_from_reality_feedback(
    case_id: str,
    request: RealityFeedbackRequest,
) -> GeneralExperimentCase:
    """Compile a replacement protocol from the user's corrected real-world context."""

    try:
        source = general_exploration_store.get(case_id)
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if source.superseded_by_case_id is not None:
        raise HTTPException(status_code=409, detail="该实验已经生成过现场修订版本。")
    if request.expected_revision != source.revision:
        raise HTTPException(status_code=409, detail="实验版本已经变化，请刷新后再提交反馈。")
    hypothesis_by_id = {
        item.hypothesis_id: item.statement_untrusted for item in source.protocol.hypotheses
    }
    if set(request.hypothesis_ids) - set(hypothesis_by_id):
        raise HTTPException(status_code=422, detail="反馈引用了不存在的候选解释。")
    evidence_reuse = build_general_reality_evidence_reuse_audit(source, request)
    preferred_sensors = tuple(item.sensor for item in source.protocol.sensors[:3])
    compile_request = GeneralQuestionCompileRequest(
        question=source.protocol.question,
        context=revised_context(
            original_context="",
            feedback=request,
            rejected_hypotheses=tuple(
                hypothesis_by_id[item] for item in request.hypothesis_ids
            ),
            task_title=source.current_task.title if source.current_task else None,
            limit=1200,
            evidence_reuse=evidence_reuse,
        ),
        preferred_sensors=preferred_sensors,
        privacy_acknowledged_sensors=(
            tuple(sensor for sensor in preferred_sensors if sensor in {"microphone", "location"})
            if request.confirm_sensitive_sensor_reuse
            else ()
        ),
        use_agent=True,
    )
    result = await compile_general_question(compile_request)
    if result.status != "draft_ready" or result.draft is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "现场反馈已经理解，但还需要一项具体补充才能安全重建实验。",
                "questions": list(result.clarification_questions),
                "blocker_codes": list(result.blocker_codes),
            },
        )
    sensitive_sensors = {
        item.sensor
        for item in result.draft.sensor_intents
        if item.sensor in {"microphone", "location"}
    }
    if sensitive_sensors and not request.confirm_sensitive_sensor_reuse:
        raise HTTPException(
            status_code=422,
            detail=(
                "新计划需要麦克风或位置传感器。请确认只保留协议所需派生结果后再提交。"
            ),
        )
    receipt_id = None
    if result.source == "bounded_agent":
        receipt_id = general_exploration_store.issue_compilation_receipt(result).receipt_id
    source_mode = (
        "protocol_emulator"
        if set(source.protocol.selected_sources) == {"protocol_emulator"}
        else "phone_upload"
    )
    revised = general_exploration_store.create(
        GeneralExplorationCaseCreate(
            draft=result.draft,
            compilation_receipt_id=receipt_id,
            source=source_mode,
            privacy_acknowledged_sensors=tuple(sensitive_sensors),
        )
    )
    try:
        return general_exploration_store.link_reality_feedback_revision(
            source.case_id,
            revised.case_id,
            request,
            evidence_reuse=evidence_reuse,
        )
    except (GeneralExplorationConflict, GeneralExplorationValidation) as exc:
        general_exploration_store.delete(revised.case_id)
        status_code = 409 if isinstance(exc, GeneralExplorationConflict) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.post(
    "/api/v2/general-explorations/{case_id}/reasoning-decision",
    response_model=GeneralExperimentCase,
)
def submit_general_reasoning_decision(
    case_id: str,
    request: GeneralReasoningCheckpointDecision,
) -> GeneralExperimentCase:
    """Continue an ambiguous loop or stop with its current calibrated evidence."""

    try:
        return decide_general_reasoning_checkpoint(
            general_exploration_store,
            case_id,
            request,
        )
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _acceleration_capture_upload(
    capture,
    *,
    label: str,
    notes: str,
    privacy_acknowledged: bool,
) -> SensorRecordingUpload:
    return SensorRecordingUpload(
        label=label,
        device=f"phyphox · {capture.probe.experiment_title}"[:120],
        sensor="accelerometer",
        notes=notes,
        channels={axis: SensorChannelDefinition(unit="m/s^2") for axis in ("x", "y", "z")},
        samples=[
            SensorSample(
                timestamp_ms=sample.timestamp_ms,
                values={"x": sample.x, "y": sample.y, "z": sample.z},
            )
            for sample in capture.samples
        ],
        provenance=SensorProvenance(
            source="phyphox_remote",
            experiment_title=capture.probe.experiment_title,
            remote_session=capture.probe.remote_session,
            config_sha256=capture.probe.config_sha256,
            channel_mapping={
                "timestamp": capture.probe.buffer_mapping.timestamp,
                "x": capture.probe.buffer_mapping.x,
                "y": capture.probe.buffer_mapping.y,
                "z": capture.probe.buffer_mapping.z,
            },
            privacy_acknowledged=privacy_acknowledged,
            phyphox_buffer_receipt=capture.buffer_receipt,
        ),
    )


def _attach_general_capture_lineage(
    upload: SensorRecordingUpload,
    *,
    case_id: str,
    task_id: str,
) -> SensorRecordingUpload:
    provenance = SensorProvenance.model_validate(
        {
            **upload.provenance.model_dump(mode="python"),
            "general_case_id": case_id,
            "general_task_id": task_id,
        }
    )
    return SensorRecordingUpload.model_validate(
        {**upload.model_dump(mode="python"), "provenance": provenance}
    )


@app.post(
    "/api/v2/general-explorations/{case_id}/phyphox",
    response_model=GeneralPhyphoxCaptureResponse,
)
async def capture_general_exploration_measurement(
    case_id: str,
    request: GeneralPhyphoxCaptureRequest,
) -> GeneralPhyphoxCaptureResponse:
    """Capture the current single-sensor task once, then bind it through the normal CAS path."""

    try:
        case = general_exploration_store.validate_phyphox_capture_request(
            case_id,
            request,
        )
        device = settings_store.get().default_phyphox_device
        if device is None:
            raise GeneralExplorationValidation(
                "尚未保存默认 phyphox 设备，请先到设备与设置完成连接。"
            )
        task = case.current_task
        if task is None:  # pragma: no cover - guarded by store validation
            raise GeneralExplorationConflict("实验当前没有可执行任务。")
        sensor = task.sensors[0]
        label = f"{case.protocol.title} · {task.title}"[:80]
        notes = f"General exploration {case.case_id}; task {task.task_id}"[:500]
        if sensor == "accelerometer":
            acceleration_capture = await capture_phyphox_acceleration(
                device.base_url,
                request.duration_s,
                device.buffer_mapping,
            )
            upload = _acceleration_capture_upload(
                acceleration_capture,
                label=label,
                notes=notes,
                privacy_acknowledged=request.privacy_acknowledged,
            )
            probe = acceleration_capture.probe
            requested_duration_s = acceleration_capture.requested_duration_s
            actual_duration_s = acceleration_capture.actual_duration_s
        else:
            sensor_capture = await capture_phyphox_sensor(
                device.base_url,
                sensor,
                request.duration_s,
                label=label,
                notes=notes,
                privacy_acknowledged=request.privacy_acknowledged,
            )
            upload = sensor_capture.recording
            probe = sensor_capture.probe
            requested_duration_s = sensor_capture.requested_duration_s
            actual_duration_s = sensor_capture.actual_duration_s
        upload = _attach_general_capture_lineage(
            upload,
            case_id=case.case_id,
            task_id=task.task_id,
        )
        stored = session_store.create_sensor_recording(upload)
        updated = await advance_general_exploration(
            general_exploration_store,
            case_id,
            GeneralRecordingMeasurementSubmit(
                expected_revision=request.expected_revision,
                task_id=request.task_id,
                recording_ids=(stored.session_id,),
                controls_confirmed=True,
            ),
        )
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeneralExplorationValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    evidence = next(
        item
        for item in reversed(updated.evidence)
        if item.lineage.recording_id == stored.session_id
    )
    return GeneralPhyphoxCaptureResponse(
        case=updated,
        evidence=evidence,
        capture=GeneralPhyphoxCaptureMetadata(
            recording_id=stored.session_id,
            experiment_title=probe.experiment_title,
            remote_session=stored.upload.provenance.remote_session,
            config_sha256=probe.config_sha256,
            requested_duration_s=requested_duration_s,
            actual_duration_s=actual_duration_s,
            sample_count=len(stored.upload.samples),
            sensor=stored.upload.sensor,
            analyzer_id=stored.analysis.analyzer_id,
        ),
    )


@app.post(
    "/api/v2/general-explorations/{case_id}/phyphox/synchronized",
    response_model=GeneralPhyphoxSynchronizedCaptureResponse,
)
async def capture_general_exploration_synchronized_measurement(
    case_id: str,
    request: GeneralPhyphoxCaptureRequest,
) -> GeneralPhyphoxSynchronizedCaptureResponse:
    """Capture one server-attested multi-sensor group, then commit one case revision."""

    try:
        case = general_exploration_store.validate_phyphox_synchronized_capture_request(
            case_id,
            request,
        )
        device = settings_store.get().default_phyphox_device
        if device is None:
            raise GeneralExplorationValidation(
                "尚未保存默认 phyphox 设备，请先到设备与设置完成连接。"
            )
        task = case.current_task
        if task is None:  # pragma: no cover - guarded by store validation
            raise GeneralExplorationConflict("实验当前没有可执行任务。")
        label = f"{case.protocol.title} · {task.title}"[:80]
        notes = f"General synchronized exploration {case.case_id}; task {task.task_id}"[:500]
        synchronized = await capture_phyphox_sensors(
            device.base_url,
            task.sensors,
            request.duration_s,
            label=label,
            notes=notes,
            privacy_acknowledged=request.privacy_acknowledged,
        )
        stored = session_store.create_sensor_recordings(
            tuple(
                _attach_general_capture_lineage(
                    synchronized.recordings[sensor],
                    case_id=case.case_id,
                    task_id=task.task_id,
                )
                for sensor in task.sensors
            )
        )
        updated = await advance_general_exploration(
            general_exploration_store,
            case_id,
            GeneralRecordingMeasurementSubmit(
                expected_revision=request.expected_revision,
                task_id=request.task_id,
                recording_ids=tuple(item.session_id for item in stored),
                controls_confirmed=True,
            ),
        )
    except GeneralExplorationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GeneralExplorationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GeneralExplorationValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    evidence_by_recording = {item.lineage.recording_id: item for item in updated.evidence}
    evidence = tuple(evidence_by_recording[item.session_id] for item in stored)
    alignment = evidence[0].lineage.alignment
    if alignment is None:  # pragma: no cover - provenance validator and binder guard
        raise HTTPException(status_code=500, detail="同步采集证明未进入证据链。")
    captures = tuple(
        GeneralPhyphoxCaptureMetadata(
            recording_id=item.session_id,
            experiment_title=synchronized.probe.experiment_title,
            remote_session=item.upload.provenance.remote_session,
            config_sha256=synchronized.probe.config_sha256,
            requested_duration_s=synchronized.requested_duration_s,
            actual_duration_s=synchronized.actual_duration_s[item.upload.sensor],
            sample_count=len(item.upload.samples),
            sensor=item.upload.sensor,
            analyzer_id=item.analysis.analyzer_id,
        )
        for item in stored
    )
    return GeneralPhyphoxSynchronizedCaptureResponse(
        case=updated,
        evidence=evidence,
        captures=captures,
        alignment=alignment,
    )


@app.post("/api/v2/investigations", response_model=InvestigationCase)
def create_investigation(request: InvestigationCaseCreate) -> InvestigationCase:
    try:
        return investigation_store.create(request)
    except InvestigationValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/api/v2/investigations",
    response_model=list[InvestigationCaseHistoryItem],
)
def list_investigations() -> list[InvestigationCaseHistoryItem]:
    return investigation_store.list()


@app.get(
    "/api/v2/exploration-history",
    response_model=list[ExplorationHistoryItem],
)
def list_exploration_history() -> list[ExplorationHistoryItem]:
    history = exploration_history_store.list_public(limit=200)
    for summary in general_exploration_store.list(limit=200):
        case = general_exploration_store.get(summary.case_id)
        history.append(
            general_exploration_history_item(
                case,
                created_at=summary.created_at,
                updated_at=summary.updated_at,
            )
        )
    for summary in investigation_store.list(limit=200):
        case = investigation_store.get(summary.case_id)
        history.append(
            investigation_history_item(
                case,
                created_at=summary.created_at,
                updated_at=summary.updated_at,
            )
        )
    return sorted(history, key=lambda item: item.updated_at, reverse=True)[:200]


@app.get("/api/v2/work-summaries", response_model=list[WorkSummary])
def list_work_summaries() -> list[WorkSummary]:
    """Return one recovery/report contract for the two primary Agent workflows."""

    summaries = [
        diagnostic_work_summary(diagnostic_case_store.get_snapshot(item.case_id))
        for item in diagnostic_case_store.list(limit=200)
    ]
    summaries.extend(
        general_work_summary(
            general_exploration_store.get(item.case_id),
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in general_exploration_store.list(limit=200)
    )
    return sorted(summaries, key=lambda item: item.updated_at, reverse=True)[:200]


@app.get(
    "/api/v2/exploration-history/public/{run_id}",
    response_model=PublicExplorationHistoryDetail,
)
def get_public_exploration_history(run_id: str) -> PublicExplorationHistoryDetail:
    try:
        return exploration_history_store.get_public(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v2/investigations/{case_id}", response_model=InvestigationCase)
def get_investigation(case_id: str) -> InvestigationCase:
    try:
        return investigation_store.get(case_id)
    except InvestigationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v2/investigations/{case_id}")
def delete_investigation(case_id: str) -> dict[str, str]:
    try:
        investigation_store.delete(case_id)
    except InvestigationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted_investigation_id": case_id}


async def _advance_investigation(
    case_id: str,
    request: InvestigationMeasurementSubmit,
) -> InvestigationCase:
    outcome = await advance_investigation(investigation_store, case_id, request)
    return outcome.case


@app.post(
    "/api/v2/investigations/{case_id}/measurements",
    response_model=InvestigationCase,
)
async def submit_investigation_measurement(
    case_id: str,
    request: InvestigationMeasurementSubmit,
) -> InvestigationCase:
    try:
        return await _advance_investigation(case_id, request)
    except InvestigationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvestigationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvestigationValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/v2/investigations/{case_id}/phyphox",
    response_model=InvestigationPhyphoxCaptureResponse,
)
async def capture_investigation_measurement(
    case_id: str,
    request: InvestigationPhyphoxCaptureRequest,
) -> InvestigationPhyphoxCaptureResponse:
    try:
        case = investigation_store.validate_capture_request(
            case_id,
            expected_revision=request.expected_revision,
            task_id=request.task_id,
            parameters=request.parameters,
            controls_confirmed=request.controls_confirmed,
        )
    except InvestigationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvestigationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvestigationValidation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not request.privacy_acknowledged:
        raise HTTPException(status_code=422, detail="请先确认可信局域网与远程采集隐私提示。")

    task = case.current_task
    if task is None:  # pragma: no cover - validated immediately above
        raise HTTPException(status_code=409, detail="实验当前没有可执行任务。")
    try:
        capture = await capture_phyphox_sensor(
            request.base_url,
            task.sensor,
            request.duration_s,
            label=f"{case.title} · {task.title}",
            notes=f"Investigation {case.case_id}; task {task.task_id}",
            privacy_acknowledged=request.privacy_acknowledged,
        )
        stored = session_store.create_sensor_recording(capture.recording)
        reference = RecordingRef(
            recording_type="sensor_v2",
            recording_id=stored.session_id,
            sensor=stored.upload.sensor,
            analyzer_id=stored.analysis.analyzer_id,
            analyzer_version=stored.analysis.analyzer_version,
            source=stored.upload.provenance.source,
            config_sha256=stored.upload.provenance.config_sha256,
            remote_session=stored.upload.provenance.remote_session or None,
        )
        updated = await _advance_investigation(
            case_id,
            InvestigationMeasurementSubmit(
                expected_revision=request.expected_revision,
                task_id=request.task_id,
                recording=reference,
                parameters=request.parameters,
                controls_confirmed=True,
                observation_notes=request.observation_notes,
            ),
        )
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except InvestigationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (InvestigationValidation, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    evidence = next(
        item
        for item in reversed(updated.evidence)
        if item.recording.recording_id == stored.session_id
    )
    return InvestigationPhyphoxCaptureResponse(
        case=updated,
        evidence=evidence,
        capture=InvestigationCaptureMetadata(
            experiment_title=capture.probe.experiment_title,
            remote_session=capture.recording.provenance.remote_session or None,
            config_sha256=capture.probe.config_sha256,
            requested_duration_s=capture.requested_duration_s,
            actual_duration_s=capture.actual_duration_s,
            sample_count=len(capture.recording.samples),
            sensor=stored.upload.sensor,
            analyzer_id=stored.analysis.analyzer_id,
        ),
    )


@app.post("/api/v1/agent/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    missing = []
    for session_id in request.session_ids:
        try:
            session_store.get(session_id)
        except KeyError:
            missing.append(session_id)
    if missing:
        raise HTTPException(status_code=404, detail={"missing_session_ids": missing})
    answer = await run_experiment_agent(request.question, request.session_ids)
    return AgentRunResponse(
        answer=answer,
        model=get_active_model_name(),
        session_ids=request.session_ids,
    )


@app.post(
    "/api/v2/investigations/route",
    response_model=InvestigationRouteRecommendation,
)
async def recommend_investigation_route(
    request: InvestigationRouteRequest,
) -> InvestigationRouteRecommendation:
    """Ask the active base model to route without creating or mutating an investigation."""

    return await route_investigation_with_model(request)


@app.post("/api/v2/evidence-workbench/analyze", response_model=EvidenceWorkbenchReport)
async def analyze_evidence_workbench(
    request: EvidenceWorkbenchRequest,
) -> EvidenceWorkbenchReport:
    """Explain selected v1/v2 evidence without mutating a diagnostic or exploration case."""

    sensors = []
    recordings = []
    missing = []
    for recording_id in request.recording_ids:
        try:
            recording = get_diagnostic_recording(session_store, recording_id)
        except KeyError:
            missing.append(recording_id)
            continue
        recordings.append(recording)
        sensors.append(recording.sensor)
    if missing:
        raise HTTPException(status_code=404, detail={"missing_recording_ids": missing})
    audits, comparability, contrasts, quality, boundaries = build_evidence_audit(
        recordings
    )
    citations, comparability_matrix, charts = build_evidence_presentation(
        audits,
        comparability,
    )
    model = get_active_model_name()
    try:
        answer = await run_evidence_workbench(request.question, request.recording_ids)
        analysis_status = "model_generated"
    except AgentRuntimeError as exc:
        if exc.kind == "concurrency_limit":
            raise
        answer = deterministic_workbench_answer(
            request.question,
            audits,
            comparability,
            quality,
        )
        analysis_status = "deterministic_only"
    except RuntimeError:
        answer = deterministic_workbench_answer(
            request.question,
            audits,
            comparability,
            quality,
        )
        analysis_status = "deterministic_only"
    return evidence_workbench_store.create_report(
        question=request.question,
        answer=answer,
        analysis_status=analysis_status,
        model=model,
        recording_ids=request.recording_ids,
        sensor_kinds=list(dict.fromkeys(sensors)),
        audits=audits,
        comparability=comparability,
        contrasts=contrasts,
        citations=citations,
        comparability_matrix=comparability_matrix,
        charts=charts,
        quality=quality,
        boundaries=boundaries,
    )


@app.get(
    "/api/v2/evidence-workbench/reports",
    response_model=list[EvidenceWorkbenchHistoryItem],
)
def list_evidence_workbench_reports() -> list[EvidenceWorkbenchHistoryItem]:
    return evidence_workbench_store.list()


@app.get(
    "/api/v2/evidence-workbench/reports/{report_id}",
    response_model=EvidenceWorkbenchReport,
)
def get_evidence_workbench_report(report_id: str) -> EvidenceWorkbenchReport:
    try:
        return evidence_workbench_store.get(report_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put(
    "/api/v2/evidence-workbench/reports/{report_id}/note",
    response_model=EvidenceWorkbenchReport,
)
def update_evidence_workbench_note(
    report_id: str,
    request: EvidenceWorkbenchNoteUpdate,
) -> EvidenceWorkbenchReport:
    try:
        return evidence_workbench_store.update_note(report_id, request.user_note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v2/evidence-workbench/reports/{report_id}/export")
def export_evidence_workbench_report(report_id: str) -> PlainTextResponse:
    try:
        report = evidence_workbench_store.get(report_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filename = f"pocketlab-{report_id}.md"
    return PlainTextResponse(
        workbench_report_markdown(report),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/v1/diagnostic-cases", response_model=DiagnosticAgentResponse)
async def create_diagnostic_case(request: DiagnosticCaseCreate) -> DiagnosticAgentResponse:
    case = diagnostic_case_store.create(request)
    try:
        agent_message = await run_diagnostic_intake_agent(case.case_id)
    except Exception as exc:
        current = diagnostic_case_store.get(case.case_id)
        if current.hypotheses and current.current_task is not None:
            agent_message = "诊断计划已保存，但 Agent 的最终说明生成失败。请按当前任务继续。"
        else:
            diagnostic_case_store.delete(case.case_id)
            raise HTTPException(
                status_code=502,
                detail=f"诊断计划生成失败：{exc}",
            ) from exc
    diagnostic_case_store.set_latest_agent_message(case.case_id, agent_message)
    return DiagnosticAgentResponse(
        case=diagnostic_case_store.get(case.case_id),
        agent_message=agent_message,
        model=get_active_model_name(),
    )


@app.post(
    "/api/v1/diagnostic-cases/{case_id}/reality-feedback",
    response_model=DiagnosticAgentResponse,
)
async def revise_diagnostic_case_from_reality_feedback(
    case_id: str,
    request: RealityFeedbackRequest,
) -> DiagnosticAgentResponse:
    """Replace invalid assumptions with a new plan while retaining the source case."""

    try:
        revised = diagnostic_case_store.create_reality_feedback_revision(case_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        agent_message = await run_diagnostic_intake_agent(revised.case_id)
    except Exception as exc:
        current = diagnostic_case_store.get(revised.case_id)
        if current.hypotheses and current.current_task is not None:
            agent_message = "现场事实已保存，新实验计划已建立；Agent 的补充说明暂时生成失败。"
        else:
            diagnostic_case_store.rollback_reality_feedback_revision(revised.case_id)
            raise HTTPException(
                status_code=502,
                detail=f"根据现场反馈重新规划失败，原案例未被替换：{exc}",
            ) from exc
    diagnostic_case_store.set_latest_agent_message(revised.case_id, agent_message)
    return DiagnosticAgentResponse(
        case=diagnostic_case_store.get(revised.case_id),
        agent_message=agent_message,
        model=get_active_model_name(),
    )


@app.get("/api/v1/diagnostic-cases", response_model=list[DiagnosticCaseHistoryItem])
def list_diagnostic_cases() -> list[DiagnosticCaseHistoryItem]:
    return diagnostic_case_store.list()


@app.get("/api/v1/diagnostic-cases/{case_id}", response_model=DiagnosticCase)
def get_diagnostic_case(case_id: str) -> DiagnosticCase:
    try:
        return diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/diagnostic-cases/{case_id}/snapshot",
    response_model=DiagnosticCaseSnapshot,
)
def get_diagnostic_case_snapshot(case_id: str) -> DiagnosticCaseSnapshot:
    try:
        return diagnostic_case_store.get_snapshot(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/diagnostic-cases/{case_id}")
def delete_diagnostic_case(case_id: str) -> dict[str, str]:
    try:
        diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    diagnostic_case_store.delete(case_id)
    return {"deleted_case_id": case_id}


@app.post(
    "/api/v1/diagnostic-cases/{case_id}/retest",
    response_model=DiagnosticAgentResponse,
)
async def create_diagnostic_retest(case_id: str) -> DiagnosticAgentResponse:
    """Create a separate executable case from a finished report's retest contract."""

    try:
        source = diagnostic_case_store.get(case_id)
        request = build_diagnostic_retest_request(source)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await create_diagnostic_case(request)


@app.post(
    "/api/v1/diagnostic-cases/{case_id}/final-report/retry",
    response_model=DiagnosticAgentResponse,
)
async def retry_diagnostic_final_report(case_id: str) -> DiagnosticAgentResponse:
    """Retry only the model-authored final explanation; preserve all evidence."""

    try:
        before = diagnostic_case_store.get(case_id)
        if before.final_report is None:
            raise ValueError("当前案例尚未结束，不能生成终局解决方案。")
        report = await run_diagnostic_finalization_agent(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    message = (
        "基模已重新生成完整终局解释与解决方案，服务端证据和安全审查已通过。"
        if report.finalization_source == "model_generated"
        else "基模本次仍未生成可安全采纳的完整终局方案；已保留明确标记的安全兜底，可稍后再试。"
    )
    diagnostic_case_store.set_latest_agent_message(case_id, message)
    return DiagnosticAgentResponse(
        case=diagnostic_case_store.get(case_id),
        agent_message=message,
        model=get_active_model_name(),
    )


@app.post(
    "/api/v1/diagnostic-cases/{case_id}/measurements",
    response_model=DiagnosticAgentResponse,
)
async def submit_diagnostic_measurement(
    case_id: str,
    request: DiagnosticMeasurementSubmit,
) -> DiagnosticAgentResponse:
    try:
        before = diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    already_committed = diagnostic_case_store.replay_committed_measurement(
        case_id,
        task_id=request.task_id,
        session_id=request.session_id,
    )
    if already_committed is not None:
        snapshot = diagnostic_case_store.get_snapshot(case_id)
        agent_message = (
            snapshot.latest_agent_message
            or (
                before.final_report.conclusion
                if before.final_report is not None
                else "这条测量已在之前的请求中保存；没有重复运行 Agent 或写入证据。"
            )
        )
        return DiagnosticAgentResponse(
            case=before,
            agent_message=agent_message,
            model=get_active_model_name(),
        )
    if before.current_task is None or before.current_task.task_id != request.task_id:
        expected = before.current_task.task_id if before.current_task else None
        raise HTTPException(
            status_code=409,
            detail={"expected_task_id": expected, "received_task_id": request.task_id},
        )
    _require_ready_task_analyzer(before.current_task)
    try:
        evidence_session = get_diagnostic_recording(session_store, request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if evidence_session.sensor != before.current_task.required_sensor:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "task_sensor_mismatch",
                "required_sensor": before.current_task.required_sensor,
                "received_sensor": evidence_session.sensor,
            },
        )

    try:
        agent_message = await run_diagnostic_measurement_agent(
            case_id=case_id,
            task_id=request.task_id,
            session_id=request.session_id,
            observation_notes=request.observation_notes,
        )
    except Exception as exc:
        current = diagnostic_case_store.get(case_id)
        measurement_was_saved = (
            len(current.evidence) == len(before.evidence) + 1
            and current.evidence[-1].session_id == request.session_id
        )
        if measurement_was_saved:
            if current.final_report:
                agent_message = current.final_report.conclusion
            else:
                agent_message = "证据与下一任务已保存，但 Agent 的最终说明生成失败。"
        else:
            proposal_unavailable = isinstance(
                exc,
                (TimeoutError, DiagnosticProposalUnavailable),
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "code": (
                        "diagnostic_agent_unavailable"
                        if proposal_unavailable
                        else "diagnostic_transition_failed"
                    ),
                    "message": (
                        "确定性分析已经完成并保留了这条测量，但模型本轮没有在时限内提交"
                        "诊断更新。当前任务没有推进；请直接重试这条已保存记录，无需重新采集。"
                        if proposal_unavailable
                        else "确定性分析已经完成并保留了这条测量，但本轮提案未通过服务端"
                        "状态或物理规则。当前任务没有推进；请重试这条已保存记录。"
                    ),
                    "recording_id": request.session_id,
                    "task_id": request.task_id,
                    "retryable": proposal_unavailable,
                },
            ) from exc
    diagnostic_case_store.set_latest_agent_message(case_id, agent_message)
    return DiagnosticAgentResponse(
        case=diagnostic_case_store.get(case_id),
        agent_message=agent_message,
        model=get_active_model_name(),
    )


def _diagnostic_preview(upload: SensorRecordingUpload) -> list[SensorSample]:
    if upload.sensor == "location":
        return []
    stride = max(1, math.ceil(len(upload.samples) / 1000))
    preview = upload.samples[::stride]
    if preview and preview[-1] is not upload.samples[-1]:
        preview.append(upload.samples[-1])
    return preview


async def _bind_diagnostic_recording(
    case_id: str,
    task_id: str,
    recording_id: str,
    observation_notes: str,
) -> DiagnosticSensorTaskResponse:
    try:
        stored = session_store.get_sensor_recording(recording_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    measurement = await submit_diagnostic_measurement(
        case_id,
        DiagnosticMeasurementSubmit(
            task_id=task_id,
            session_id=recording_id,
            observation_notes=observation_notes,
        ),
    )
    return DiagnosticSensorTaskResponse(
        session=SensorRecordingCreated(
            session_id=stored.session_id,
            label=stored.upload.label,
            sensor=stored.upload.sensor,
            analysis=stored.analysis,
            created_at=stored.created_at,
        ),
        case=measurement.case,
        agent_message=measurement.agent_message,
        model=measurement.model,
        preview_samples=_diagnostic_preview(stored.upload),
    )


@app.post(
    "/api/v2/diagnostic-cases/{case_id}/tasks/{task_id}/recordings",
    response_model=DiagnosticSensorTaskResponse,
)
async def submit_diagnostic_sensor_recording(
    case_id: str,
    task_id: str,
    request: DiagnosticRecordingSubmit,
) -> DiagnosticSensorTaskResponse:
    return await _bind_diagnostic_recording(
        case_id,
        task_id,
        request.recording_id,
        request.observation_notes,
    )


@app.post(
    "/api/v2/diagnostic-cases/{case_id}/tasks/{task_id}/phyphox",
    response_model=DiagnosticSensorTaskResponse,
)
async def capture_diagnostic_sensor_task(
    case_id: str,
    task_id: str,
    request: DiagnosticTaskPhyphoxRequest,
) -> DiagnosticSensorTaskResponse:
    try:
        case = diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    task = case.current_task
    if task is None or task.task_id != task_id:
        raise HTTPException(status_code=409, detail="diagnostic task is no longer current")
    _require_ready_task_analyzer(task)
    try:
        capture = await capture_phyphox_sensor(
            request.base_url,
            task.required_sensor,
            request.duration_s,
            label=request.label,
            notes=request.notes,
            privacy_acknowledged=request.privacy_acknowledged,
        )
        stored = session_store.create_sensor_recording(capture.recording)
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await _bind_diagnostic_recording(
        case_id,
        task_id,
        stored.session_id,
        request.observation_notes,
    )


@app.post(
    "/api/v2/diagnostic-cases/{case_id}/tasks/{task_id}/public-replay",
    response_model=DiagnosticSensorTaskResponse,
)
async def replay_public_diagnostic_task(
    case_id: str,
    task_id: str,
    request: DiagnosticPublicReplaySubmit,
) -> DiagnosticSensorTaskResponse:
    try:
        case = diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    task = case.current_task
    if task is None or task.task_id != task_id:
        raise HTTPException(status_code=409, detail="diagnostic task is no longer current")
    matching = [
        item for item in list_public_replay_catalog(PUBLIC_REPLAY_DIR)
        if item.sensor == task.required_sensor
    ]
    if request.dataset_id is not None:
        matching = [item for item in matching if item.dataset_id == request.dataset_id]
    if not matching:
        raise HTTPException(
            status_code=409,
            detail={"code": "no_reviewed_public_replay", "sensor": task.required_sensor},
        )
    used_recordings: set[tuple[str | None, str | None]] = set()
    for evidence in case.evidence:
        try:
            prior = session_store.get_sensor_recording(evidence.session_id)
        except KeyError:
            continue
        used_recordings.add(
            (
                prior.upload.provenance.public_dataset_id,
                prior.upload.provenance.public_recording_id,
            )
        )
    preferred_role = "baseline" if task.task_kind == "baseline" else "condition"
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    candidates = []
    acknowledgement_blocked = False
    explicit_recording_seen = request.recording_id is None
    for catalog_item in matching:
        if catalog_item.requires_user_acknowledgement and not request.privacy_acknowledged:
            acknowledgement_blocked = True
            continue
        try:
            pack_dir, manifest = get_public_replay_dataset(
                PUBLIC_REPLAY_DIR,
                catalog_item.dataset_id,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        available = [
            item
            for item in manifest.recordings
            if (manifest.dataset_id, item.recording_id) not in used_recordings
        ]
        if request.recording_id is not None:
            available = [
                item for item in available if item.recording_id == request.recording_id
            ]
            explicit_recording_seen = explicit_recording_seen or bool(available)
        if not available:
            continue
        available.sort(
            key=lambda item: (
                item.evidence_role == preferred_role,
                item.independent_measurement,
            ),
            reverse=True,
        )
        recording = available[0]
        try:
            upload = read_public_replay_recording(pack_dir, manifest, recording)
            analysis = analyze_sensor_recording(upload)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        candidates.append(
            (
                confidence_rank[analysis.confidence],
                recording.evidence_role == preferred_role,
                recording.independent_measurement,
                catalog_item.dataset_id,
                recording.recording_id,
                upload,
            )
        )
    if not explicit_recording_seen:
        raise HTTPException(status_code=404, detail="unknown public replay recording")
    if not candidates and acknowledgement_blocked:
        raise HTTPException(
            status_code=409,
            detail={"code": "public_replay_privacy_acknowledgement_required"},
        )
    if not candidates:
        raise HTTPException(
            status_code=409,
            detail={"code": "no_unused_public_replay", "sensor": task.required_sensor},
        )
    candidates.sort(reverse=True, key=lambda item: item[:5])
    upload = candidates[0][-1]
    try:
        stored = session_store.create_sensor_recording(upload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    note = (
        request.observation_notes.strip()
        + " 该记录来自公开数据集，不是当前用户手机或当前居家现场的物理证据。"
    ).strip()
    return await _bind_diagnostic_recording(
        case_id,
        task_id,
        stored.session_id,
        note,
    )


@app.post(
    "/api/v1/diagnostic-cases/{case_id}/checkpoint",
    response_model=DiagnosticAgentResponse,
)
async def decide_diagnostic_checkpoint(
    case_id: str,
    request: DiagnosticCheckpointDecision,
) -> DiagnosticAgentResponse:
    try:
        case = diagnostic_case_store.decide_checkpoint(
            case_id,
            decision=request.decision,
            expected_completed_task_count=request.expected_completed_task_count,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if case.final_report is not None:
        await run_diagnostic_finalization_agent(case_id)
        case = diagnostic_case_store.get(case_id)
    message = (
        "已按你的选择继续诊断；下一项是 Agent 在检查点前选出的最高信息价值任务。"
        if request.decision == "continue"
        else case.final_report.conclusion if case.final_report else "诊断已按用户选择结束。"
    )
    diagnostic_case_store.set_latest_agent_message(case_id, message)
    return DiagnosticAgentResponse(
        case=case,
        agent_message=message,
        model=get_active_model_name(),
    )


@app.get("/api/v1/mobile/cases/{case_id}/task", response_model=MobileTaskResponse)
def get_mobile_task(case_id: str) -> MobileTaskResponse:
    try:
        case = diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MobileTaskResponse(
        case_id=case.case_id,
        case_title=case.title,
        problem_statement=case.problem_statement,
        status=case.status,
        task=case.current_task,
        hypotheses=case.hypotheses,
        evidence_count=len(case.evidence),
        final_report=case.final_report,
    )


@app.post("/api/v1/phyphox/probe", response_model=PhyphoxProbeResponse)
async def probe_phyphox_connection(request: PhyphoxConnectionRequest) -> PhyphoxProbeResponse:
    try:
        probe = await probe_phyphox(request.base_url, request.buffer_mapping)
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PhyphoxProbeResponse(
        base_url=probe.base_url,
        experiment_title=probe.experiment_title,
        remote_session=probe.remote_session,
        measuring=probe.measuring,
        compatible=probe.compatible,
        buffer_mapping=probe.buffer_mapping,
        available_buffers=probe.available_buffers,
        missing_buffers=probe.missing_buffers,
        detected_sensors=probe.detected_sensors,
        export_buffers=probe.export_buffers,
        exploration_matches=probe.exploration_matches,
        config_sha256=probe.config_sha256,
        sensor_profiles=probe.sensor_profiles,
    )


@app.post(
    "/api/v2/phyphox/capability-checks/{sensor}",
    response_model=SensorCapabilityCheck,
)
async def check_phyphox_sensor_capability(
    sensor: SensorKind,
    request: PhyphoxConnectionRequest,
) -> SensorCapabilityCheck:
    """Inspect one current input without granting capture, analysis or Agent authority."""

    try:
        probe = await probe_phyphox(request.base_url, request.buffer_mapping)
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return build_sensor_capability_check(probe, sensor, sensor_capabilities())


@app.post("/api/v2/phyphox/capture", response_model=PhyphoxSensorCaptureResponse)
async def capture_phyphox_sensor_recording(
    request: PhyphoxSensorCaptureRequest,
) -> PhyphoxSensorCaptureResponse:
    """Capture and analyze a non-Agent v2 sensor recording.

    Passing this endpoint proves the deterministic capture/analysis path only. It
    does not make the sensor eligible as diagnostic Agent evidence.
    """

    try:
        capture = await capture_phyphox_sensor(
            request.base_url,
            request.sensor,
            request.duration_s,
            label=request.label,
            notes=request.notes,
            privacy_acknowledged=request.privacy_acknowledged,
        )
        stored = session_store.create_sensor_recording(capture.recording)
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    preview = []
    if capture.recording.sensor != "location":
        stride = max(1, math.ceil(len(capture.recording.samples) / 1000))
        preview = capture.recording.samples[::stride]
        if preview[-1] is not capture.recording.samples[-1]:
            preview.append(capture.recording.samples[-1])
    return PhyphoxSensorCaptureResponse(
        session=SensorRecordingCreated(
            session_id=stored.session_id,
            label=stored.upload.label,
            sensor=stored.upload.sensor,
            analysis=stored.analysis,
            created_at=stored.created_at,
        ),
        capture=PhyphoxSensorCaptureMetadata(
            experiment_title=capture.probe.experiment_title,
            remote_session=capture.recording.provenance.remote_session,
            config_sha256=capture.probe.config_sha256,
            requested_duration_s=capture.requested_duration_s,
            actual_duration_s=capture.actual_duration_s,
            sample_count=len(capture.recording.samples),
            profile=capture.profile,
        ),
        preview_samples=preview,
    )


@app.get("/api/v1/explorations", response_model=list[ExplorationTemplate])
def get_exploration_catalog() -> list[ExplorationTemplate]:
    return list_explorations()


@app.get("/api/v1/settings", response_model=PocketLabSettings)
def get_settings() -> PocketLabSettings:
    return settings_store.get()


def _raise_model_profile_http_error(exc: ModelProfileError) -> None:
    if isinstance(exc, ModelProfileNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ModelSecretUnavailable):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/settings/models", response_model=ModelProfileCatalog)
def get_model_profiles() -> ModelProfileCatalog:
    return model_profile_store.catalog()


@app.get("/api/v1/settings/agent-runs", response_model=AgentRunAuditCatalog)
def get_agent_run_audits(limit: int = 30) -> AgentRunAuditCatalog:
    return agent_run_audit_store.catalog(limit=min(max(limit, 1), 100))


@app.post("/api/v1/settings/models", response_model=ModelProfileSummary, status_code=201)
def create_model_profile(request: ModelProfileCreate) -> ModelProfileSummary:
    try:
        return model_profile_store.create(request)
    except ModelProfileError as exc:
        _raise_model_profile_http_error(exc)


@app.put(
    "/api/v1/settings/models/{profile_id}",
    response_model=ModelProfileSummary,
)
def update_model_profile(
    profile_id: str,
    request: ModelProfileUpdate,
) -> ModelProfileSummary:
    if profile_id == ENVIRONMENT_PROFILE_ID:
        raise HTTPException(status_code=409, detail="系统环境配置是只读的。")
    try:
        return model_profile_store.update(profile_id, request)
    except ModelProfileError as exc:
        _raise_model_profile_http_error(exc)


@app.post(
    "/api/v1/settings/models/{profile_id}/activate",
    response_model=ModelProfileCatalog,
)
def activate_model_profile(profile_id: str) -> ModelProfileCatalog:
    try:
        if profile_id == ENVIRONMENT_PROFILE_ID:
            if environment_model_configuration() is None:
                raise HTTPException(status_code=409, detail="系统环境模型尚未完整配置。")
            model_profile_store.activate_environment()
        else:
            model_profile_store.activate(profile_id)
        return model_profile_store.catalog()
    except ModelProfileError as exc:
        _raise_model_profile_http_error(exc)


@app.post(
    "/api/v1/settings/models/{profile_id}/probe",
    response_model=ModelCapabilityProbe,
)
async def probe_model_profile(profile_id: str) -> ModelCapabilityProbe:
    try:
        if profile_id == ENVIRONMENT_PROFILE_ID:
            config = environment_model_configuration()
            if config is None:
                raise HTTPException(status_code=409, detail="系统环境模型尚未完整配置。")
            return await probe_model_compatibility(config)
        config = model_profile_store.resolve(profile_id)
        result = await probe_model_compatibility(config)
        model_profile_store.save_probe(profile_id, result)
        return result
    except ModelProfileError as exc:
        _raise_model_profile_http_error(exc)


@app.delete(
    "/api/v1/settings/models/{profile_id}",
    response_model=ModelProfileCatalog,
)
def delete_model_profile(profile_id: str) -> ModelProfileCatalog:
    if profile_id == ENVIRONMENT_PROFILE_ID:
        raise HTTPException(status_code=409, detail="系统环境配置不能从网页删除。")
    try:
        model_profile_store.delete(profile_id)
        return model_profile_store.catalog()
    except ModelProfileError as exc:
        _raise_model_profile_http_error(exc)


@app.put("/api/v1/settings/profile", response_model=LocalProfile)
def update_profile(request: LocalProfileUpdate) -> LocalProfile:
    return settings_store.update_profile(request.display_name)


@app.put("/api/v1/settings/phyphox", response_model=PhyphoxDeviceSaveResponse)
async def save_default_phyphox_device(
    request: PhyphoxDeviceSaveRequest,
) -> PhyphoxDeviceSaveResponse:
    try:
        probe = await probe_phyphox(request.base_url, request.buffer_mapping)
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    device = settings_store.save_default_phyphox(request.name, probe)
    return PhyphoxDeviceSaveResponse(
        device=device,
        probe=PhyphoxProbeResponse(**probe.__dict__),
    )


@app.post("/api/v1/settings/phyphox/probe", response_model=PhyphoxDeviceSaveResponse)
async def probe_default_phyphox_device() -> PhyphoxDeviceSaveResponse:
    settings = settings_store.get()
    device = settings.default_phyphox_device
    if device is None:
        raise HTTPException(status_code=404, detail="尚未保存默认 phyphox 设备。")
    try:
        probe = await probe_phyphox(device.base_url, device.buffer_mapping)
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    refreshed = settings_store.refresh_default_phyphox(probe)
    return PhyphoxDeviceSaveResponse(
        device=refreshed,
        probe=PhyphoxProbeResponse(**probe.__dict__),
    )


@app.delete("/api/v1/settings/phyphox", response_model=PocketLabSettings)
def delete_default_phyphox_device() -> PocketLabSettings:
    settings_store.delete_default_phyphox()
    return settings_store.get()


def _require_ready_task_analyzer(task: DiagnosticMeasurementTask) -> None:
    if task.analyzer_status == "ready":
        return
    raise HTTPException(
        status_code=409,
        detail={
            "code": "task_analyzer_not_implemented",
            "required_sensor": task.required_sensor,
            "measurement_quantity": task.measurement_quantity,
            "recommended_phyphox_experiment": task.recommended_phyphox_experiment,
            "message": (
                f"当前 Task 需要{task.measurement_quantity}，但 PocketLab 尚未接入对应的"
                "确定性分析器；为避免把加速度数据误当成该物理量，已阻止提交正式证据。"
            ),
        },
    )


@app.post(
    "/api/v1/mobile/cases/{case_id}/tasks/{task_id}/samples",
    response_model=TaskSampleResponse,
)
async def upload_task_samples(
    case_id: str,
    task_id: str,
    request: TaskSampleUpload,
) -> TaskSampleResponse:
    try:
        before = diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if before.current_task is None or before.current_task.task_id != task_id:
        expected = before.current_task.task_id if before.current_task else None
        raise HTTPException(
            status_code=409,
            detail={"expected_task_id": expected, "received_task_id": task_id},
        )
    _require_ready_task_analyzer(before.current_task)
    if request.sensor != before.current_task.required_sensor:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "task_sensor_mismatch",
                "required_sensor": before.current_task.required_sensor,
                "received_sensor": request.sensor,
            },
        )

    upload = SessionUpload.model_validate(request.model_dump(exclude={"observation_notes"}))
    try:
        stored = session_store.create(upload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session = SessionCreated(
        session_id=stored.session_id,
        label=stored.upload.label,
        analysis=stored.analysis,
        created_at=stored.created_at,
    )

    try:
        measurement = await submit_diagnostic_measurement(
            case_id,
            DiagnosticMeasurementSubmit(
                task_id=task_id,
                session_id=stored.session_id,
                observation_notes=request.observation_notes,
            ),
        )
    except HTTPException:
        session_store.delete(stored.session_id)
        raise
    return TaskSampleResponse(
        session=session,
        case=measurement.case,
        agent_message=measurement.agent_message,
        model=measurement.model,
    )


@app.post(
    "/api/v1/mobile/cases/{case_id}/tasks/{task_id}/phyphox",
    response_model=PhyphoxTaskResponse,
)
async def capture_phyphox_task(
    case_id: str,
    task_id: str,
    request: PhyphoxCaptureRequest,
) -> PhyphoxTaskResponse:
    try:
        before = diagnostic_case_store.get(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if before.current_task is None or before.current_task.task_id != task_id:
        expected = before.current_task.task_id if before.current_task else None
        raise HTTPException(
            status_code=409,
            detail={"expected_task_id": expected, "received_task_id": task_id},
        )
    _require_ready_task_analyzer(before.current_task)

    try:
        capture = await capture_phyphox_acceleration(
            request.base_url,
            request.duration_s,
            request.buffer_mapping,
        )
    except PhyphoxUrlError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PhyphoxError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    buffer_receipt = capture.buffer_receipt
    provenance_parts = [
        "phyphox 远程采集",
        f"实验={capture.probe.experiment_title}",
        f"请求时长={capture.requested_duration_s:.2f}s",
        f"有效时长={capture.actual_duration_s:.2f}s",
    ]
    if buffer_receipt is not None:
        discarded = ",".join(
            f"{role}:{count}"
            for role, count in sorted(buffer_receipt.discarded_tail_samples.items())
        )
        alignment_note = f"对齐={buffer_receipt.alignment_method}"
        if discarded:
            alignment_note += f"({discarded})"
        provenance_parts.extend(
            (
                f"缓冲区读取={buffer_receipt.read_attempts}次",
                alignment_note,
            )
        )
    provenance = "；".join(provenance_parts) + "。"
    notes = f"{request.notes.strip()} {provenance}".strip()
    submitted = await upload_task_samples(
        case_id,
        task_id,
        TaskSampleUpload(
            label=request.label,
            device=f"phyphox · {capture.probe.experiment_title}"[:120],
            notes=notes[:500],
            observation_notes=request.observation_notes,
            samples=capture.samples,
        ),
    )
    preview_stride = max(1, math.ceil(len(capture.samples) / 1000))
    preview_samples = capture.samples[::preview_stride]
    if preview_samples[-1] is not capture.samples[-1]:
        preview_samples.append(capture.samples[-1])
    return PhyphoxTaskResponse(
        **submitted.model_dump(),
        capture=PhyphoxCaptureMetadata(
            experiment_title=capture.probe.experiment_title,
            remote_session=capture.probe.remote_session,
            requested_duration_s=capture.requested_duration_s,
            actual_duration_s=capture.actual_duration_s,
            sample_count=len(capture.samples),
            buffer_mapping=capture.probe.buffer_mapping,
        ),
        preview_samples=preview_samples,
    )


def synthetic_upload(frequency_hz: float = 12.0, amplitude: float = 0.35) -> SessionUpload:
    sampling_rate = 100.0
    duration = 5.0
    samples = []
    for index in range(int(sampling_rate * duration)):
        t = index / sampling_rate
        samples.append(
            AccelerationSample(
                timestamp_ms=t * 1000.0,
                x=amplitude * math.sin(2.0 * math.pi * frequency_hz * t),
                y=0.02 * math.sin(2.0 * math.pi * 3.0 * t),
                z=9.81,
            )
        )
    return SessionUpload(label=f"synthetic-{frequency_hz:g}Hz", samples=samples)


async def smoke_agent() -> None:
    session = session_store.create(synthetic_upload())
    answer = await run_experiment_agent(
        "桌面似乎在振动。请根据测量判断目前能得出什么，并设计下一次对照实验。",
        [session.session_id],
    )
    print(answer)


def cli() -> None:
    parser = argparse.ArgumentParser(description="PocketLab Agent server")
    parser.add_argument("--smoke-signal", action="store_true")
    parser.add_argument("--smoke-agent", action="store_true")
    args = parser.parse_args()

    if args.smoke_signal:
        analysis = session_store.create(synthetic_upload()).analysis
        print(analysis.model_dump_json(indent=2))
        return
    if args.smoke_agent:
        asyncio.run(smoke_agent())
        return

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("pocketlab.main:app", host="0.0.0.0", port=port, reload=False)
