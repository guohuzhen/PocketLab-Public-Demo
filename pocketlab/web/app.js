"use strict";

const state = {
  currentUser: null,
  diagnosticCase: null,
  measurementMode: "public",
  pendingFile: null,
  diagnosticRetryRecording: null,
  diagnosticFeedbackHypothesisIds: new Set(),
  sessions: [],
  selectedIds: new Set(),
  activeId: null,
  latestAgentMessage: "",
  phyphoxProbe: null,
  caseHistory: [],
  workSummaries: [],
  settings: null,
  savedDevice: null,
  modelCatalog: null,
  editingModelProfileId: null,
  workbenchReports: [],
  activeWorkbenchReport: null,
  investigationRoute: null,
  generalRoutedContext: "",
  agentRunCatalog: null,
  explorations: [],
  experimentProtocols: [],
  investigation: null,
  investigationError: "",
  investigationHistory: [],
  generalCapabilities: [],
  generalReadiness: null,
  generalHistory: [],
  generalCase: null,
  generalAcquisitionPlan: null,
  generalError: "",
  generalPublicComponents: null,
  generalPublicRun: null,
  generalPublicError: "",
  generalPublicRunning: false,
  generalCompiledDraft: null,
  generalCompileResult: null,
  generalFeedbackHypothesisIds: new Set(),
  explorationHistory: [],
  pendingExploration: null,
  sensorRecordings: [],
  sensorCapabilities: [],
  publicReplays: [],
  publicLightRun: null,
  publicLightError: "",
  publicPressureRun: null,
  publicPressureError: "",
  publicSensorRun: null,
  publicSensorError: "",
  publicSensorActive: null,
  publicSensorProtocol: null,
  capabilityCheck: null,
  explorationFilter: "all",
  busy: false,
  modelRunUi: null,
};

const nativeFetch = typeof window !== "undefined" && typeof window.fetch === "function"
  ? window.fetch.bind(window)
  : () => Promise.reject(new Error("fetch is unavailable outside the browser runtime"));
if (typeof window !== "undefined") window.fetch = modelAwareFetch;

function shouldTrackModelRequest(input, init = {}) {
  const method = String(init.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
  if (method !== "POST") return false;
  const rawUrl = typeof input === "string" ? input : input.url;
  const path = new URL(rawUrl, window.location.origin).pathname;
  return [
    /^\/api\/v1\/diagnostic-cases(?:$|\/[^/]+\/(?:measurements|reality-feedback|retest|checkpoint|final-report\/retry))$/,
    /^\/api\/v2\/diagnostic-cases\/[^/]+\/tasks\/[^/]+\/(?:recordings|phyphox|public-replay)$/,
    /^\/api\/v2\/general-explorations\/compile$/,
    /^\/api\/v2\/general-explorations\/[^/]+\/(?:measurements|simulate|reality-feedback|reasoning-decision|public-components\/run)$/,
    /^\/api\/v2\/public-replays\/(?:light|pressure|sensors\/[^/]+)\/explore$/,
    /^\/api\/v2\/investigations\/route$/,
    /^\/api\/v2\/investigations\/[^/]+\/(?:measurements|phyphox)$/,
    /^\/api\/v1\/agent\/run$/,
    /^\/api\/v2\/evidence-workbench\/analyze$/,
  ].some((pattern) => pattern.test(path));
}

async function modelAwareFetch(input, init = {}) {
  if (!shouldTrackModelRequest(input, init)) return nativeFetch(input, init);
  const runId = `web_${crypto.randomUUID().replaceAll("-", "")}`;
  const headers = new Headers(
    init.headers || (input instanceof Request ? input.headers : undefined),
  );
  headers.set("X-PocketLab-Model-Run", runId);
  const startedAt = Date.now();
  const activeProfile = state.modelCatalog?.profiles?.find(
    (profile) => profile.profile_id === state.modelCatalog?.active_profile_id,
  );
  const configuredMode = activeProfile?.reasoning_strategy || "high";
  state.modelRunUi = { runId, startedAt, decisionAvailable: false, finishing: false };
  renderModelRunPanel({
    phase: "connecting",
    detail: "正在建立模型连接；达到 2 分钟时只会询问你，不会自动截断或兜底。",
    elapsed_s: 0,
    decision_available: false,
    reasoning_mode: configuredMode,
  });
  pollModelRunStatus(runId);
  try {
    return await nativeFetch(input, { ...init, headers });
  } finally {
    if (state.modelRunUi?.runId === runId) {
      try {
        const finalStatusResponse = await nativeFetch(
          `/api/v1/model-runs/${encodeURIComponent(runId)}`,
        );
        if (finalStatusResponse.ok) renderModelRunPanel(await finalStatusResponse.json());
      } catch (_error) {
        // The business response remains authoritative if the optional status channel closes first.
      }
      state.modelRunUi.finishing = true;
      window.setTimeout(() => {
        if (state.modelRunUi?.runId === runId) {
          elements.modelRunPanel.hidden = true;
          state.modelRunUi = null;
        }
      }, 1400);
    }
  }
}

async function pollModelRunStatus(runId) {
  if (state.modelRunUi?.runId !== runId || state.modelRunUi.finishing) return;
  try {
    const response = await nativeFetch(`/api/v1/model-runs/${encodeURIComponent(runId)}`);
    if (response.ok) {
      const status = await response.json();
      renderModelRunPanel(status);
    } else if (response.status !== 404) {
      throw new Error(await readApiError(response));
    } else {
      const elapsed = (Date.now() - state.modelRunUi.startedAt) / 1000;
      renderModelRunPanel({
        phase: elapsed < 4 ? "connecting" : "thinking",
        detail: elapsed < 4 ? "正在建立模型连接" : "模型请求已经发出，正在等待基模响应",
        elapsed_s: elapsed,
        decision_available: false,
      });
    }
  } catch (_error) {
    const elapsed = (Date.now() - state.modelRunUi.startedAt) / 1000;
    renderModelRunPanel({
      phase: "connecting",
      detail: "业务请求仍在进行；运行状态通道暂时不可用",
      elapsed_s: elapsed,
      decision_available: false,
    });
  }
  window.setTimeout(() => pollModelRunStatus(runId), 600);
}

function renderModelRunPanel(status) {
  if (!elements.modelRunPanel) return;
  elements.modelRunPanel.hidden = false;
  const phase = status.phase || "connecting";
  const titles = {
    connecting: "正在连接基模",
    thinking: "基模正在处理（可能处于深度推理）",
    streaming: "基模正在流式生成",
    validating: "正在校验并整理模型结果",
    completed: "模型结果已完成",
    failed: "模型请求未成功",
    fallback_requested: "正在切换到安全兜底",
  };
  const eyebrows = {
    connecting: "MODEL CONNECTION",
    thinking: "MODEL PROCESSING",
    streaming: "LIVE MODEL STREAM",
    validating: "SERVER VALIDATION",
    completed: "MODEL COMPLETE",
    failed: "MODEL ERROR",
    fallback_requested: "MODEL FALLBACK",
  };
  elements.modelRunEyebrow.textContent = eyebrows[phase] || "MODEL RUN";
  elements.modelRunTitle.textContent = titles[phase] || "模型任务处理中";
  elements.modelRunDetail.textContent = status.detail || "正在等待模型服务响应。";
  elements.modelRunElapsed.textContent = `已等待 ${formatModelRunElapsed(status.elapsed_s || 0)}`;
  const reasoningMode = status.reasoning_mode || "provider_default";
  elements.modelRunMode.textContent = reasoningMode === "high"
    ? "MODE · HIGH"
    : reasoningMode === "fast"
      ? "MODE · FAST"
      : "MODE · PROVIDER DEFAULT";
  elements.modelRunPanel.dataset.phase = phase;
  elements.modelRunPanel.dataset.mode = reasoningMode;
  renderModelRunStream(status);
  elements.modelRunDecision.hidden = !status.decision_available;
  const allowedDecisions = new Set(status.allowed_decisions || []);
  const reachedMaxTurns = phase === "failed" && /max.?turns/i.test(status.error_kind || "");
  elements.modelRunDecisionText.textContent = reachedMaxTurns
    ? "本轮 Agent 已达到模型—工具往返次数上限；这不是基模思考时间到期。重试会提高本轮往返额度；只有你主动接受时，系统才会使用安全兜底。"
    : phase === "failed"
    ? "本轮基模调用或结果校验没有完成。你可以重试基模；只有你主动接受时，系统才会进入明确标记的安全兜底。"
    : reasoningMode === "high"
      ? "本次 High 运行已超过 2 分钟。你可以继续等待、停止当前请求并改用 Fast，或亲自选择明确标记的安全兜底。系统不会替你选择。"
      : "本次 Fast 运行已超过 2 分钟。你可以继续等待，或亲自选择明确标记的安全兜底；Fast 不会反向切换到 High。";
  elements.modelRunContinueButton.textContent = reachedMaxTurns
    ? "提高轮数并重试基模"
    : phase === "failed" ? "重试基模" : "继续等待基模";
  elements.modelRunContinueButton.disabled = false;
  elements.modelRunFastButton.hidden = !allowedDecisions.has("fast");
  elements.modelRunFastButton.disabled = false;
  elements.modelRunFallbackButton.disabled = false;
  if (state.modelRunUi) {
    state.modelRunUi.decisionAvailable = Boolean(status.decision_available);
    state.modelRunUi.phase = phase;
  }
}

function renderModelRunStream(status) {
  const outputCharacters = Number(status.stream_characters || 0);
  const reasoningChunks = Number(status.reasoning_stream_chunks || 0);
  const stages = Array.isArray(status.stage_events) ? status.stage_events : [];
  const hasActivity = outputCharacters > 0 || reasoningChunks > 0 || stages.length > 0;
  elements.modelRunStream.hidden = !hasActivity;
  if (!hasActivity) return;
  const firstChunk = status.first_stream_elapsed_s == null
    ? ""
    : ` · 首片段 ${formatModelRunElapsed(status.first_stream_elapsed_s)}`;
  elements.modelRunStreamMeta.textContent = outputCharacters > 0
    ? `${outputCharacters.toLocaleString("zh-CN")} 个可见字符${firstChunk}`
    : `High 推理流 ${reasoningChunks.toLocaleString("zh-CN")} 个片段 · 正文尚未开始`;
  elements.modelRunStreamPreview.textContent = status.stream_preview
    || (reasoningChunks > 0
      ? "High 模式仍在生成隐藏推理；PocketLab 只显示活动状态，不读取思维内容。"
      : "模型正在组织可校验的结构化结果…");
  elements.modelRunStages.innerHTML = stages.map((stage) => `
    <span><b>${escapeHtml(stage.label || "MODEL")}</b>${escapeHtml(stage.detail || "处理中")}<i>${escapeHtml(stage.at_s ?? 0)}s</i></span>`).join("");
}

function formatModelRunElapsed(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  if (value < 60) return `${value} 秒`;
  return `${Math.floor(value / 60)} 分 ${String(value % 60).padStart(2, "0")} 秒`;
}

async function decideActiveModelRun(decision) {
  const run = state.modelRunUi;
  if (!run?.runId || !run.decisionAvailable) return;
  const wasFailed = run.phase === "failed";
  elements.modelRunContinueButton.disabled = true;
  elements.modelRunFastButton.disabled = true;
  elements.modelRunFallbackButton.disabled = true;
  try {
    const response = await nativeFetch(
      `/api/v1/model-runs/${encodeURIComponent(run.runId)}/decision`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision }),
      },
    );
    if (!response.ok) throw new Error(await readApiError(response));
    renderModelRunPanel(await response.json());
    if (decision === "continue") {
      elements.modelRunDecision.hidden = true;
      showToast(wasFailed
        ? "已按你的选择重试基模"
        : "已继续等待基模；两分钟后仍未完成时可以再次选择");
    } else if (decision === "fast") {
      elements.modelRunDecision.hidden = true;
      showToast("已按你的选择停止 High，并以 Fast 模式重新请求基模");
    } else {
      showToast("已请求停止等待；结果会明确标记为安全兜底");
    }
  } catch (error) {
    showToast(error.message || "无法提交模型等待选择。", true);
    elements.modelRunContinueButton.disabled = false;
    elements.modelRunFastButton.disabled = false;
    elements.modelRunFallbackButton.disabled = false;
  }
}

const SIMULATION_PROFILES = {
  washing_unbalanced: {
    group: "洗衣机",
    label: "洗衣机脱水 · 衣物偏载",
    description: "8.2 Hz 旋转基频明显，并带有 16.4 Hz 二次谐波；振幅随启动阶段逐渐上升。",
    duration: 8, noise: 0.012, envelope: "ramp",
    components: [
      { axis: "x", frequency: 8.2, amplitude: 0.42 },
      { axis: "x", frequency: 16.4, amplitude: 0.13, phase: 0.5 },
      { axis: "y", frequency: 8.2, amplitude: 0.22, phase: 0.7 },
      { axis: "z", frequency: 4.1, amplitude: 0.05 },
    ],
  },
  washing_balanced: {
    group: "洗衣机",
    label: "洗衣机脱水 · 均匀摆放衣物",
    description: "保持 8.2 Hz 转速特征，但基频和谐波幅值明显下降，适合与偏载基线比较。",
    duration: 8, noise: 0.01, envelope: "ramp",
    components: [
      { axis: "x", frequency: 8.2, amplitude: 0.14 },
      { axis: "x", frequency: 16.4, amplitude: 0.035, phase: 0.5 },
      { axis: "y", frequency: 8.2, amplitude: 0.07, phase: 0.7 },
    ],
  },
  fan_direct: {
    group: "桌面风扇",
    label: "桌面风扇 · 直接接触桌面",
    description: "12 Hz 旋转基频和 24 Hz 谐波同时存在，模拟机械振动直接传入桌面。",
    duration: 6, noise: 0.009,
    components: [
      { axis: "x", frequency: 12, amplitude: 0.34 },
      { axis: "x", frequency: 24, amplitude: 0.07, phase: 0.3 },
      { axis: "y", frequency: 12, amplitude: 0.11, phase: 0.8 },
    ],
  },
  fan_isolated: {
    group: "桌面风扇",
    label: "桌面风扇 · 增加软垫隔振",
    description: "频率仍保持 12 Hz，但 RMS 明显下降，用于检验接触传振路径。",
    duration: 6, noise: 0.009,
    components: [
      { axis: "x", frequency: 12, amplitude: 0.09 },
      { axis: "x", frequency: 24, amplitude: 0.018, phase: 0.3 },
      { axis: "y", frequency: 12, amplitude: 0.03, phase: 0.8 },
    ],
  },
  speaker_resonance: {
    group: "扬声器共振",
    label: "音箱扫频 · 桌面 20 Hz 强响应",
    description: "20 Hz 响应显著且竖直方向最强，模拟桌面在特定激励频率附近被放大。",
    duration: 6, noise: 0.008,
    components: [
      { axis: "z", frequency: 20, amplitude: 0.29 },
      { axis: "x", frequency: 20, amplitude: 0.08, phase: 0.4 },
      { axis: "z", frequency: 40, amplitude: 0.035 },
    ],
  },
  speaker_off_resonance: {
    group: "扬声器共振",
    label: "音箱扫频 · 17 Hz 非共振对照",
    description: "只改变播放频率到 17 Hz，桌面响应明显降低，适合检验频率选择性放大。",
    duration: 6, noise: 0.008,
    components: [
      { axis: "z", frequency: 17, amplitude: 0.075 },
      { axis: "x", frequency: 17, amplitude: 0.025, phase: 0.4 },
    ],
  },
  refrigerator_on: {
    group: "冰箱压缩机",
    label: "冰箱压缩机 · 运行状态",
    description: "25 Hz 窄带振动叠加轻微低频摆动，模拟压缩机运行时的稳定机械激励。",
    duration: 7, noise: 0.012,
    components: [
      { axis: "x", frequency: 25, amplitude: 0.13 },
      { axis: "y", frequency: 25, amplitude: 0.06, phase: 0.6 },
      { axis: "x", frequency: 2, amplitude: 0.018 },
    ],
  },
  refrigerator_off: {
    group: "冰箱压缩机",
    label: "冰箱压缩机 · 停机对照",
    description: "同一位置停机记录，只保留较弱环境底噪和微小低频分量。",
    duration: 7, noise: 0.012,
    components: [
      { axis: "x", frequency: 25, amplitude: 0.016 },
      { axis: "x", frequency: 2, amplitude: 0.01 },
    ],
  },
  panel_loose: {
    group: "设备松动",
    label: "设备外壳 · 螺丝松动",
    description: "18 Hz 基频伴随较强 36 Hz 谐波，模拟松动面板的非线性碰撞特征。",
    duration: 6, noise: 0.011,
    components: [
      { axis: "x", frequency: 18, amplitude: 0.22 },
      { axis: "x", frequency: 36, amplitude: 0.17, phase: 0.2 },
      { axis: "y", frequency: 18, amplitude: 0.07 },
    ],
  },
  panel_tightened: {
    group: "设备松动",
    label: "设备外壳 · 紧固螺丝后",
    description: "18 Hz 工作频率仍存在，但 36 Hz 谐波和整体 RMS 显著降低。",
    duration: 6, noise: 0.011,
    components: [
      { axis: "x", frequency: 18, amplitude: 0.065 },
      { axis: "x", frequency: 36, amplitude: 0.018, phase: 0.2 },
      { axis: "y", frequency: 18, amplitude: 0.025 },
    ],
  },
  footsteps_near: {
    group: "楼板脚步",
    label: "楼板脚步 · 测点附近行走",
    description: "2 Hz 步频与 4/6 Hz 谐波形成周期冲击，模拟测点附近稳定步行。",
    duration: 8, noise: 0.014,
    components: [
      { axis: "z", frequency: 2, amplitude: 0.25 },
      { axis: "z", frequency: 4, amplitude: 0.11, phase: 0.2 },
      { axis: "z", frequency: 6, amplitude: 0.05, phase: 0.5 },
      { axis: "x", frequency: 2, amplitude: 0.06 },
    ],
  },
  footsteps_far: {
    group: "楼板脚步",
    label: "楼板脚步 · 远离测点行走",
    description: "步频保持 2 Hz，但各次谐波幅值降低，用于检验距离衰减。",
    duration: 8, noise: 0.014,
    components: [
      { axis: "z", frequency: 2, amplitude: 0.08 },
      { axis: "z", frequency: 4, amplitude: 0.033, phase: 0.2 },
      { axis: "z", frequency: 6, amplitude: 0.015, phase: 0.5 },
    ],
  },
  low_quality_short: {
    group: "质量门禁",
    label: "低质量样本 · 记录过短且接近噪声",
    description: "仅 1.2 秒、有效振动很弱，用于验证低置信度证据不会改变假设方向。",
    duration: 1.2, noise: 0.018,
    components: [{ axis: "x", frequency: 14, amplitude: 0.006 }],
  },
  alias_risk: {
    group: "质量门禁",
    label: "混叠风险 · 46 Hz 接近采样上限",
    description: "100 Hz 采样下的 46 Hz 信号接近奈奎斯特频率，用于触发混叠风险警告。",
    duration: 5, noise: 0.008,
    components: [{ axis: "x", frequency: 46, amplitude: 0.2 }],
  },
};

const elements = {};
let toastTimer = null;
let workSummaryRequest = null;

document.addEventListener("DOMContentLoaded", initializeApp);

async function initializeApp() {
  collectElements();
  bindEvents();
  if (!(await loadCurrentUser())) return;
  applyRoute(false);
  checkHealth();
  populateSimulationProfiles();
  drawEmptyChart();
  await loadWorkspaceState();
  renderDashboard();
  applyRoute(true);
}

function collectElements() {
  [
    "healthPill", "healthText", "profileChip", "profileInitial", "profileUsername",
    "modelRunPanel", "modelRunEyebrow", "modelRunTitle", "modelRunDetail", "modelRunElapsed", "modelRunMode",
    "modelRunStream", "modelRunStreamMeta", "modelRunStreamPreview", "modelRunStages",
    "modelRunDecision", "modelRunDecisionText", "modelRunContinueButton", "modelRunFastButton", "modelRunFallbackButton",
    "logoutButton", "mobileLogoutButton", "pageEyebrow", "pageTitle", "dashboardGreeting", "dashboardCaseCount",
    "dashboardSessionCount", "dashboardDeviceState", "dashboardDeviceDetail",
    "dashboardRecentEmpty", "dashboardRecentCases", "continueCaseButton",
    "showcaseDiagnosticStartButton", "showcaseExplorationStartButton",
    "investigationRouterQuestion", "investigationRouterContext", "routeInvestigationButton",
    "investigationRouteResult", "investigationRouteBadge", "investigationRouteTitle", "investigationRouteConfidence",
    "investigationRouteSummary", "investigationRouteSensors", "investigationRouteReasons", "investigationRoutePrivacy",
    "startRecommendedWorkflowButton", "startAlternativeWorkflowButton",
    "profileNameInput", "saveProfileButton",
    "modelProfileList", "modelProfileForm", "newModelProfileButton",
    "modelFormTitle", "modelFormMode", "modelProfileName", "modelBaseUrl", "modelNameInput",
    "modelReasoningStrategy", "modelApiKey", "modelApiKeyHint", "modelApiKeyToggle", "modelInputCost", "modelOutputCost",
    "modelSecretBackend", "modelSaveStatus", "saveModelProfileButton", "cancelModelEditButton",
    "refreshAgentRunsButton", "agentRuntimeSummary", "agentRuntimeList", "agentRuntimeBoundary",
    "deviceNameInput", "deviceUrlInput", "deviceSaveStatus", "deviceOverview",
    "saveDeviceButton", "checkSavedDeviceButton", "removeDeviceButton", "workflowDeviceStatus",
    "caseSetup", "activeWorkflow", "caseTitleInput",
    "problemInput", "caseContextInput", "createDiagnosticButton", "newCaseButton",
    "diagnosticCaseTitle", "diagnosticCaseId", "diagnosticRuntimeNotice", "hypothesisList", "currentTask", "terminationProgress",
    "diagnosticRealityFeedback", "diagnosticFeedbackType", "diagnosticFeedbackSelection", "diagnosticFeedbackMessage", "diagnosticFeedbackPrivacy", "diagnosticFeedbackSubmit", "diagnosticFeedbackStatus",
    "finalReportBlock", "finalOutcomeBadge", "finalConclusion", "finalUserTakeaway", "finalMechanismExplanation", "finalConfidenceExplanation", "finalTerminationReason", "finalEvidenceExplanation", "finalScopeBoundary",
    "terminationGrid", "finalUncertainties", "finalSolutionPlan", "solutionBasisBadge",
    "solutionProvenance", "solutionProvenanceBadge", "solutionProvenanceNote", "retryFinalReportButton",
    "solutionSummary", "solutionActions", "solutionEscalation", "optionalRetest",
    "optionalRetestTitle", "optionalRetestInstruction", "optionalRetestCriteria", "copyRetestButton", "measureBlock",
    "explorationHome", "explorationPresetWorkspace", "explorationBackLink", "explorationCatalogPanel",
    "generalExplorationWorkspace", "generalExplorationBackLink", "generalExplorationBuilder",
    "generalReadinessBoundary", "generalReadinessBadge", "generalReadinessSummary", "generalReadinessGates",
    "generalNaturalQuestion", "generalPreferredSensor", "generalCompilerMicrophonePrivacy", "generalCompilerLocationPrivacy", "generalCompileButton",
    "generalCompileResult", "generalCompilerClarification", "generalClarificationContracts", "generalConditionClarification", "generalClarificationVariable",
    "generalClarificationReference", "generalClarificationComparison", "generalMechanismClarification",
    "generalMechanismClarificationHint", "generalKeepMechanisms", "generalMechanismFields", "generalFirstMechanism",
    "generalSecondMechanism", "generalFreeformClarification", "generalClarificationAnswer", "generalClarificationRetry", "generalCompileStatus",
    "generalTitle", "generalQuestion", "generalIndependentVariable", "generalReferenceLabel",
    "generalComparisonLabel", "generalOptionalControl", "generalExecutionMode", "generalExecutionBoundary", "generalAlignment", "generalPrimarySensor", "generalSupportingSensor",
    "generalOptionalSensor", "generalOptionalSensor2", "generalSensorSummary", "generalPrivacyConfirm", "generalHypothesisConfirm", "generalCreateButton",
    "generalCreateStatus", "generalExplorationRun", "generalRunTitle", "generalRunQuestion",
    "generalRunRevision", "generalRunStatus", "generalTaskStep", "generalTaskTitle",
    "generalTaskInstruction", "generalTaskTags", "generalRecordingBind", "generalRecordingSelectors",
    "generalRefreshRecordings", "generalControlsConfirm", "generalControlsConfirmText", "generalSubmitMeasurement",
    "generalAcquisitionPlan", "generalMeasurementSources", "generalLiveStatus", "generalLiveDuration", "generalLivePrivacy",
    "generalLiveCapture", "generalLiveSourcePanel", "generalSavedSourcePanel", "generalSimulationSourcePanel",
    "generalSimulationProfile", "generalSimulateMeasurement",
    "generalRunMessage", "generalProgress", "generalSufficiency", "generalActivationRule", "generalBlockers",
    "generalReasoningCheckpoint", "generalCheckpointTitle", "generalCheckpointPrompt",
    "generalCheckpointEvidence", "generalCheckpointStop", "generalCheckpointContinue",
    "generalTrajectorySection", "generalTrajectoryStatus", "generalTrajectoryList",
    "generalHypothesisSection", "generalHypothesisList",
    "generalRealityFeedback", "generalFeedbackType", "generalFeedbackSelection", "generalFeedbackMessage", "generalFeedbackPrivacy", "generalFeedbackSubmit", "generalFeedbackStatus",
    "generalPublicComponents", "generalPublicBoundaries", "generalPublicPrivacy",
    "generalPublicComponentList", "generalPublicStatus", "generalPublicResult",
    "generalEvidenceTrace", "generalPlannerTrace", "generalFinalReport", "generalReportConfidence",
    "generalReportBasis", "generalReportAnswer", "generalReportNarrative", "generalReportReason", "generalSummaryGrid",
    "generalReasoningAnalysis", "generalReasoningScore", "generalReasoningMechanism", "generalReasoningExplanations",
    "generalVisualizationGrid", "generalContrastList", "generalAuxiliarySection",
    "generalAuxiliaryList", "generalReportHypothesisSection", "generalReportHypothesisConclusion", "generalReportHypotheses",
    "generalReportBoundaries",
    "explorationActiveRunsPanel", "explorationActiveRunsEmpty", "explorationActiveRuns",
    "explorationSetupPanel", "explorationSetupTitle", "explorationSetupQuestion",
    "explorationSetupPhoneTitle", "explorationSetupPhoneDescription", "explorationDistanceConstraint",
    "explorationSimulationQuestion", "explorationSimulationScope", "explorationAdvancedTools",
    "explorationSetupCancelButton", "explorationSetupPublicButton", "explorationSetupStartButton",
    "explorationCapabilityOverview", "explorationMaxDistance", "explorationGrid", "explorationEmpty",
    "capabilityCheckPanel", "capabilityCheckTitle", "capabilityCheckStatus", "capabilityCheckSummary",
    "capabilityCheckChannels", "capabilityCheckBlockers", "capabilityCheckNextSteps", "capabilityCheckPrivacy",
    "investigationWorkbench", "investigationTitle", "investigationQuestion", "investigationRevision",
    "investigationTaskStep", "investigationTaskTitle", "investigationTaskInstruction", "investigationControls",
    "investigationDistanceField", "investigationDistance", "investigationDuration", "investigationRecording",
    "investigationObservation", "investigationConfirm", "investigationCaptureButton", "investigationBindButton", "investigationStatus",
    "investigationDecision", "investigationDecisionSource", "investigationDecisionReason", "investigationDecisionBasis",
    "investigationError", "investigationErrorMessage", "investigationRefreshButton",
    "investigationProgress", "investigationDecisionState", "investigationBlockers",
    "investigationEvidenceTrace", "investigationToolTrace", "investigationPlannerTrace",
    "investigationReport", "investigationOutcome", "investigationConfidence", "investigationKeyMetrics",
    "investigationConclusion", "investigationStopReason", "investigationChart", "investigationArtifactWarnings",
    "investigationResultTable", "investigationUncertainties", "investigationMarketBoundary", "investigationBoundaries",
    "sensorCapabilityGrid", "sensorLabSensor", "sensorLabDuration", "sensorLabLabel",
    "sensorLabPrivacy", "sensorLabCaptureButton", "sensorLabStatus", "sensorLabResult",
    "sensorLabResultTitle", "sensorLabConfidence", "sensorLabMetrics", "sensorLabWarnings",
    "publicReplayLab", "publicLightRunner", "publicPressureRunner",
    "publicReplayDataset", "publicReplayRecording", "publicReplayImportButton",
    "publicReplayStatus", "publicReplayDetails",
    "publicLightQuestion", "publicLightQueryLux", "publicLightPrivacy", "publicLightRunButton",
    "publicLightStatus", "publicLightResult", "publicLightReportTitle", "publicLightExecutionStatus",
    "publicLightPlannerStatus", "publicLightReportSummary", "publicLightGates", "publicLightPlannerTrace",
    "publicLightToolTrace", "publicLightEvidence", "publicLightFindings", "publicLightSources",
    "publicLightUncertainties", "publicLightForbiddenClaims", "publicLightNextLive",
    "publicPressureQuestion", "publicPressurePrivacy", "publicPressureRunButton",
    "publicPressureStatus", "publicPressureResult", "publicPressureReportTitle",
    "publicPressureExecutionStatus", "publicPressurePlannerStatus", "publicPressureReportSummary",
    "publicPressureGates", "publicPressurePlannerTrace", "publicPressureToolTrace",
    "publicPressureEvidence", "publicPressureFindings", "publicPressureSources",
    "publicPressureUncertainties", "publicPressureForbiddenClaims", "publicPressureNextLive",
    "publicSensorRunner", "publicSensorName", "publicSensorIntro", "publicSensorQuestionLabel", "publicSensorQuestionHelp",
    "publicSensorQuestion", "publicSensorPrivacy", "publicSensorRunButton", "publicSensorStatus",
    "publicSensorResult", "publicSensorReportTitle", "publicSensorExecutionStatus",
    "publicSensorPlannerStatus", "publicSensorReportSummary", "publicSensorGates",
    "publicSensorPlannerTrace", "publicSensorToolTrace", "publicSensorEvidence",
    "publicSensorComparison", "publicSensorVisual", "publicSensorFindings", "publicSensorSources",
    "publicSensorUncertainties", "publicSensorForbiddenClaims", "publicSensorNextLive",
    "publicDiagnosticPane", "publicDiagnosticTitle", "publicDiagnosticBoundary", "publicDiagnosticPrivacy", "publicDiagnosticRunButton", "diagnosticRetryButton",
    "simulationPane", "simulationProfile", "simulationProfileTitle", "simulationProfileDescription",
    "simulationProfileTags", "filePane", "mobilePane", "taskFileInput", "taskFileButton",
    "taskFileTitle", "taskFileMeta", "taskAnalyzerNotice", "mobileSensorIntro",
    "mobileServerUrl", "mobileCaseCode", "mobileTaskCode",
    "phyphoxBaseUrl", "phyphoxDuration", "phyphoxLabel", "phyphoxObservation", "phyphoxStatus", "mobilePrivacyConfirm", "mobilePrivacyCheckbox",
    "probePhyphoxButton", "capturePhyphoxButton", "refreshTaskButton", "measurementForm", "measurementLabelInput", "observationInput",
    "submitTaskButton", "latestResult", "metricFrequency", "metricRms", "metricRate", "metricBand",
    "metricOneLabel", "metricOneUnit", "metricTwoLabel", "metricTwoUnit", "metricThreeLabel",
    "metricConfidence", "metricWarning", "signalChart", "chartEmpty",
    "diagnosticAgentMessage", "caseHistoryEmpty", "caseHistoryList", "caseHistoryCount",
    "explorationHistoryEmpty", "explorationHistoryList", "explorationHistoryCount",
    "sessionEmpty", "sessionList", "sessionHistoryCount", "selectedSessionChips", "evidenceWorkbenchLibrary",
    "questionInput", "runAgentButton", "modelBadge", "agentPlaceholder", "agentResponse",
    "workbenchReportHistory", "workbenchAnalysisStatus", "workbenchReportQuestion", "workbenchReportConfidence",
    "workbenchQuality", "workbenchAudits", "workbenchComparability", "workbenchMatrix", "workbenchContrasts", "workbenchCharts", "workbenchAnswer",
    "workbenchBoundaries", "workbenchUserNote", "saveWorkbenchNoteButton", "exportWorkbenchButton", "toast",
  ].forEach((id) => { elements[id] = document.getElementById(id); });
}

function bindEvents() {
  elements.logoutButton.addEventListener("click", logout);
  elements.mobileLogoutButton.addEventListener("click", logout);
  elements.continueCaseButton.addEventListener("click", () => {
    const item = state.workSummaries.find((entry) => entry.resumable) || state.workSummaries[0];
    if (item) navigateTo(item.resume_path);
    else if (state.caseHistory[0]) openCase(state.caseHistory[0].case_id);
  });
  document.querySelectorAll("[data-route-link]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigateTo(link.getAttribute("href"));
    });
  });
  window.addEventListener("popstate", () => applyRoute(true));
  elements.saveProfileButton.addEventListener("click", saveProfile);
  elements.modelProfileForm.addEventListener("submit", saveModelProfile);
  elements.newModelProfileButton.addEventListener("click", resetModelProfileForm);
  elements.cancelModelEditButton.addEventListener("click", resetModelProfileForm);
  elements.modelApiKeyToggle.addEventListener("click", toggleModelApiKey);
  elements.modelProfileList.addEventListener("click", handleModelProfileAction);
  elements.refreshAgentRunsButton.addEventListener("click", () => loadAgentRuns(false));
  elements.workbenchReportHistory.addEventListener("click", handleWorkbenchHistoryClick);
  elements.saveWorkbenchNoteButton.addEventListener("click", saveWorkbenchNote);
  elements.exportWorkbenchButton.addEventListener("click", exportWorkbenchReport);
  elements.saveDeviceButton.addEventListener("click", saveDefaultDevice);
  elements.checkSavedDeviceButton.addEventListener("click", () => checkSavedDevice(false));
  elements.removeDeviceButton.addEventListener("click", removeSavedDevice);
  elements.sensorLabSensor.addEventListener("change", updateSensorLabAvailability);
  elements.sensorLabPrivacy.addEventListener("change", updateSensorLabAvailability);
  elements.sensorLabCaptureButton.addEventListener("click", captureSensorLabRecording);
  elements.publicReplayDataset.addEventListener("change", renderPublicReplaySelection);
  elements.publicReplayRecording.addEventListener("change", updatePublicReplayAvailability);
  elements.publicReplayImportButton.addEventListener("click", importPublicReplayRecording);
  elements.publicLightQuestion.addEventListener("input", updatePublicLightAvailability);
  elements.publicLightQueryLux.addEventListener("input", updatePublicLightAvailability);
  elements.publicLightPrivacy.addEventListener("change", updatePublicLightAvailability);
  elements.publicLightRunButton.addEventListener("click", runPublicLightExploration);
  elements.publicPressureQuestion.addEventListener("input", updatePublicPressureAvailability);
  elements.publicPressurePrivacy.addEventListener("change", updatePublicPressureAvailability);
  elements.publicPressureRunButton.addEventListener("click", runPublicPressureExploration);
  elements.publicSensorQuestion.addEventListener("input", updatePublicSensorAvailability);
  elements.publicSensorPrivacy.addEventListener("change", updatePublicSensorAvailability);
  elements.publicSensorRunButton.addEventListener("click", runPublicSensorExploration);
  elements.explorationSetupCancelButton.addEventListener("click", closeExplorationSetup);
  elements.explorationSetupPublicButton.addEventListener("click", openPendingPublicExploration);
  elements.explorationSetupStartButton.addEventListener("click", startPendingPhoneInvestigation);
  elements.investigationCaptureButton.addEventListener("click", captureInvestigationMeasurement);
  elements.investigationBindButton.addEventListener("click", bindInvestigationRecording);
  elements.investigationRefreshButton.addEventListener("click", refreshCurrentInvestigation);
  elements.generalNaturalQuestion.addEventListener("input", () => {
    state.generalRoutedContext = "";
    resetGeneralCompilerClarification();
  });
  elements.showcaseDiagnosticStartButton.addEventListener("click", startDiagnosticShowcaseReplay);
  elements.showcaseExplorationStartButton.addEventListener("click", startExplorationShowcaseReplay);
  elements.generalPreferredSensor.addEventListener("change", updateGeneralCompilerAvailability);
  elements.generalCompilerMicrophonePrivacy.addEventListener("change", updateGeneralCompilerAvailability);
  elements.generalCompilerLocationPrivacy.addEventListener("change", updateGeneralCompilerAvailability);
  elements.generalCompileButton.addEventListener("click", compileGeneralQuestion);
  elements.generalClarificationVariable.addEventListener("input", updateGeneralCompilerAvailability);
  elements.generalClarificationReference.addEventListener("input", updateGeneralCompilerAvailability);
  elements.generalClarificationComparison.addEventListener("input", updateGeneralCompilerAvailability);
  elements.generalKeepMechanisms.addEventListener("change", () => {
    updateGeneralClarificationForm();
    updateGeneralCompilerAvailability();
  });
  elements.routeInvestigationButton.addEventListener("click", routeInvestigationQuestion);
  elements.investigationRouterQuestion.addEventListener("input", invalidateInvestigationRoute);
  elements.investigationRouterContext.addEventListener("input", invalidateInvestigationRoute);
  elements.startRecommendedWorkflowButton.addEventListener("click", () => startRoutedWorkflow(state.investigationRoute?.recommended_workflow));
  elements.startAlternativeWorkflowButton.addEventListener("click", () => startRoutedWorkflow(state.investigationRoute?.alternative_workflow));
  elements.generalFirstMechanism.addEventListener("input", updateGeneralCompilerAvailability);
  elements.generalSecondMechanism.addEventListener("input", updateGeneralCompilerAvailability);
  elements.generalClarificationAnswer.addEventListener("input", updateGeneralCompilerAvailability);
  elements.generalClarificationRetry.addEventListener("click", compileGeneralQuestion);
  elements.generalPrimarySensor.addEventListener("change", updateGeneralSensorForm);
  elements.generalTitle.addEventListener("input", updateGeneralSensorForm);
  elements.generalQuestion.addEventListener("input", updateGeneralSensorForm);
  elements.generalIndependentVariable.addEventListener("input", updateGeneralSensorForm);
  elements.generalReferenceLabel.addEventListener("input", updateGeneralSensorForm);
  elements.generalComparisonLabel.addEventListener("input", updateGeneralSensorForm);
  elements.generalSupportingSensor.addEventListener("change", updateGeneralSensorForm);
  elements.generalOptionalSensor.addEventListener("change", updateGeneralSensorForm);
  elements.generalOptionalSensor2.addEventListener("change", updateGeneralSensorForm);
  elements.generalOptionalControl.addEventListener("input", updateGeneralSensorForm);
  elements.generalExecutionMode.addEventListener("change", updateGeneralSensorForm);
  elements.generalAlignment.addEventListener("change", updateGeneralSensorForm);
  elements.generalPrivacyConfirm.addEventListener("change", updateGeneralSensorForm);
  elements.generalHypothesisConfirm.addEventListener("change", updateGeneralSensorForm);
  elements.generalCreateButton.addEventListener("click", createGeneralExploration);
  elements.generalRefreshRecordings.addEventListener("click", refreshGeneralRecordings);
  elements.generalRecordingSelectors.addEventListener("change", updateGeneralMeasurementButton);
  elements.generalControlsConfirm.addEventListener("change", updateGeneralMeasurementButton);
  elements.generalLiveDuration.addEventListener("input", updateGeneralMeasurementButton);
  elements.generalLivePrivacy.addEventListener("change", updateGeneralMeasurementButton);
  elements.generalLiveCapture.addEventListener("click", captureGeneralMeasurement);
  elements.generalSimulationProfile.addEventListener("change", updateGeneralMeasurementButton);
  elements.generalSimulateMeasurement.addEventListener("click", simulateGeneralMeasurement);
  elements.generalPublicPrivacy.addEventListener("change", renderGeneralPublicComponents);
  elements.generalPublicComponentList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-general-public-component]");
    if (button) runGeneralPublicComponent(button.dataset.generalPublicComponent);
  });
  elements.generalSubmitMeasurement.addEventListener("click", submitGeneralMeasurement);
  elements.generalCheckpointContinue.addEventListener("click", () => decideGeneralCheckpoint("continue"));
  elements.generalCheckpointStop.addEventListener("click", () => decideGeneralCheckpoint("stop"));
  elements.generalHypothesisList.addEventListener("click", handleGeneralFeedbackTarget);
  elements.generalTaskTags.addEventListener("click", handleGeneralTaskFeedback);
  elements.generalFeedbackType.addEventListener("change", renderGeneralFeedbackSelection);
  elements.generalFeedbackSubmit.addEventListener("click", submitGeneralRealityFeedback);
  elements.createDiagnosticButton.addEventListener("click", createDiagnosticCase);
  elements.newCaseButton.addEventListener("click", resetCaseView);
  elements.hypothesisList.addEventListener("click", handleDiagnosticFeedbackTarget);
  elements.currentTask.addEventListener("click", handleDiagnosticTaskFeedback);
  elements.diagnosticFeedbackType.addEventListener("change", renderDiagnosticFeedbackSelection);
  elements.diagnosticFeedbackSubmit.addEventListener("click", submitDiagnosticRealityFeedback);
  document.querySelectorAll("[data-exploration-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      state.explorationFilter = button.dataset.explorationFilter;
      document.querySelectorAll("[data-exploration-filter]").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      renderExplorations();
    });
  });
  elements.copyRetestButton.addEventListener("click", createOptionalRetestCase);
  elements.retryFinalReportButton.addEventListener("click", retryDiagnosticFinalReport);
  elements.modelRunContinueButton.addEventListener("click", () => decideActiveModelRun("continue"));
  elements.modelRunFastButton.addEventListener("click", () => decideActiveModelRun("fast"));
  elements.modelRunFallbackButton.addEventListener("click", () => decideActiveModelRun("fallback"));
  document.querySelectorAll(".source-choice").forEach((button) => {
    button.addEventListener("click", () => switchMeasurementMode(button.dataset.mode));
  });
  elements.taskFileButton.addEventListener("click", () => elements.taskFileInput.click());
  elements.simulationProfile.addEventListener("change", updateSimulationProfile);
  elements.taskFileInput.addEventListener("change", (event) => {
    if (event.target.files[0]) loadTaskFile(event.target.files[0]);
  });
  ["dragenter", "dragover"].forEach((name) => {
    elements.taskFileButton.addEventListener(name, (event) => event.preventDefault());
  });
  elements.taskFileButton.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer.files[0]) loadTaskFile(event.dataTransfer.files[0]);
  });
  elements.submitTaskButton.addEventListener("click", submitTaskMeasurement);
  elements.publicDiagnosticRunButton.addEventListener("click", runDiagnosticPublicReplay);
  elements.diagnosticRetryButton.addEventListener("click", retryDiagnosticRecording);
  elements.probePhyphoxButton.addEventListener("click", probePhyphoxConnection);
  elements.capturePhyphoxButton.addEventListener("click", capturePhyphoxMeasurement);
  elements.phyphoxBaseUrl.addEventListener("input", () => {
    if (state.phyphoxProbe) resetPhyphoxStatus("地址已改变，请重新检测手机连接。");
    updateSubmitButton();
  });
  elements.refreshTaskButton.addEventListener("click", refreshMobileTask);
  elements.runAgentButton.addEventListener("click", runAdvancedAgent);
  elements.questionInput.addEventListener("input", updateAdvancedButton);
  window.addEventListener("resize", debounce(() => {
    const active = getActiveSession();
    if (active) drawSignalChart(active.samples);
    else drawEmptyChart();
  }, 160));
}

async function loadCurrentUser() {
  try {
    const response = await fetch("/api/v1/auth/me");
    if (!response.ok) {
      window.location.replace("/login");
      return false;
    }
    state.currentUser = await response.json();
    renderAccountIdentity();
    return true;
  } catch (error) {
    window.location.replace("/login");
    return false;
  }
}

function renderAccountIdentity() {
  if (!state.currentUser) return;
  const name = state.currentUser.display_name || state.currentUser.username;
  elements.profileChip.textContent = name;
  elements.profileUsername.textContent = `@${state.currentUser.username}`;
  elements.profileInitial.textContent = name.trim().slice(0, 1).toUpperCase() || "P";
  elements.dashboardGreeting.textContent = `${name}，欢迎回来`;
}

async function logout() {
  if (state.busy) return;
  elements.logoutButton.disabled = true;
  try {
    await fetch("/api/v1/auth/logout", { method: "POST" });
  } finally {
    window.location.replace("/login");
  }
}

function routeState() {
  const path = window.location.pathname.replace(/\/+$/, "") || "/app";
  if (path === "/app") return { page: "dashboard", title: "工作台", eyebrow: "POCKETLAB WORKSPACE" };
  if (path === "/app/settings") return { page: "settings", title: "设备与设置", eyebrow: "ACCOUNT & PHONE BRIDGE" };
  if (path === "/app/explore") return { page: "explore", title: "探索实验", eyebrow: "REAL-WORLD EXPLORATION", exploreView: "home" };
  if (path === "/app/explore/presets") return { page: "explore", title: "预设实验", eyebrow: "VALIDATED PRESET LABS", exploreView: "presets" };
  if (path === "/app/explore/general") return { page: "explore", title: "自由探索", eyebrow: "BOUNDED GENERAL EXPLORATION", exploreView: "general" };
  const generalMatch = path.match(/^\/app\/explore\/general\/runs\/([^/]+)$/);
  if (generalMatch) return {
    page: "explore",
    title: "自由探索运行",
    eyebrow: "ACTIVE GENERAL EXPLORATION",
    exploreView: "general_run",
    generalCaseId: decodeURIComponent(generalMatch[1]),
  };
  const investigationMatch = path.match(/^\/app\/explore\/runs\/([^/]+)$/);
  if (investigationMatch) return {
    page: "explore",
    title: "实验运行",
    eyebrow: "ACTIVE EXPLORATION",
    exploreView: "run",
    investigationId: decodeURIComponent(investigationMatch[1]),
  };
  if (path === "/app/history") return { page: "history", title: "历史记录", eyebrow: "CASES & EVIDENCE" };
  if (path === "/app/advanced") return { page: "advanced", title: "证据工作台", eyebrow: "EVIDENCE REVIEW" };
  if (path === "/app/cases/new") return { page: "cases", title: "新建诊断", eyebrow: "GUIDED EXPERIMENT", newCase: true };
  const caseMatch = path.match(/^\/app\/cases\/([^/]+)$/);
  if (caseMatch) return { page: "cases", title: "当前案例", eyebrow: "ACTIVE DIAGNOSTIC", caseId: decodeURIComponent(caseMatch[1]) };
  return { page: "dashboard", title: "工作台", eyebrow: "POCKETLAB WORKSPACE" };
}

function navigateTo(path) {
  if (!path || path === window.location.pathname) {
    applyRoute(true);
    return;
  }
  window.history.pushState({}, "", path);
  applyRoute(true);
}

function applyRoute(loadCase = true) {
  const route = routeState();
  document.querySelectorAll("[data-page]").forEach((page) => {
    page.hidden = page.dataset.page !== route.page;
  });
  document.querySelectorAll("[data-nav]").forEach((link) => {
    link.classList.toggle("active", link.dataset.nav === route.page);
  });
  elements.pageTitle.textContent = route.title;
  elements.pageEyebrow.textContent = route.eyebrow;
  document.title = `${route.title} · PocketLab`;
  if (route.page === "cases") {
    if (route.newCase) {
      showNewCaseForm();
    } else if (route.caseId) {
      elements.caseSetup.hidden = true;
      elements.activeWorkflow.hidden = false;
      if (loadCase && state.diagnosticCase?.case_id !== route.caseId) {
        openCase(route.caseId, { updateRoute: false, scroll: false });
      }
    }
  }
  if (route.page === "explore") renderExploreRoute(route, loadCase);
  window.scrollTo({ top: 0, behavior: "auto" });
}

function renderExploreRoute(route, loadRun = true) {
  const view = route.exploreView || "home";
  elements.explorationHome.hidden = view !== "home";
  const presetView = view === "presets" || view === "run";
  const generalView = view === "general" || view === "general_run";
  elements.explorationPresetWorkspace.hidden = !presetView;
  elements.generalExplorationWorkspace.hidden = !generalView;
  elements.explorationPresetWorkspace.dataset.view = view;
  elements.explorationBackLink.href = view === "run" ? "/app/explore/presets" : "/app/explore";
  elements.explorationBackLink.textContent = view === "run" ? "← 返回预设实验" : "← 返回探索方式";
  if (view !== "run") {
    elements.investigationWorkbench.hidden = true;
  } else if (route.investigationId) {
    if (state.investigation?.case_id === route.investigationId) {
      renderInvestigation();
    } else if (loadRun) {
      openInvestigation(route.investigationId);
    }
  }
  elements.generalExplorationBuilder.hidden = view !== "general";
  elements.generalExplorationRun.hidden = view !== "general_run";
  elements.generalExplorationBackLink.href = view === "general_run" ? "/app/explore/general" : "/app/explore";
  elements.generalExplorationBackLink.textContent = view === "general_run" ? "← 返回自由探索设计" : "← 返回探索方式";
  if (view === "general_run" && route.generalCaseId) {
    if (state.generalCase?.case_id === route.generalCaseId) renderGeneralExploration();
    else if (loadRun) openGeneralExploration(route.generalCaseId);
  }
}

function showNewCaseForm() {
  state.diagnosticCase = null;
  state.diagnosticRetryRecording = null;
  state.pendingFile = null;
  state.phyphoxProbe = null;
  elements.activeWorkflow.hidden = true;
  elements.caseSetup.hidden = false;
  elements.latestResult.hidden = true;
  resetPhyphoxStatus();
}

function isDiagnosticShowcaseCase(item = state.diagnosticCase) {
  return Boolean(item?.context?.includes("showcase-replay:diagnostic-v1"));
}

function isGeneralShowcaseCase(item = state.generalCase) {
  return Boolean(
    item?.protocol?.title === "灯离远一倍，照度会怎样变化？· 零等待回放"
    && item.protocol.selected_sources?.length === 1
    && item.protocol.selected_sources[0] === "protocol_emulator"
  );
}

function setShowcaseLauncherBusy(busy, activeButton = null, activeLabel = "正在建立回放…") {
  state.busy = busy;
  const buttons = [elements.showcaseDiagnosticStartButton, elements.showcaseExplorationStartButton];
  buttons.forEach((button) => { button.disabled = busy; });
  if (activeButton) {
    const span = activeButton.querySelector("span");
    if (span) span.textContent = activeLabel;
  }
  updateSubmitButton();
  updateGeneralMeasurementButton();
}

function restoreShowcaseLauncherLabels() {
  elements.showcaseDiagnosticStartButton.querySelector("span").textContent = "洗衣机诊断 · 2 步";
  elements.showcaseExplorationStartButton.querySelector("span").textContent = "光学探索 · 4 步";
}

async function startDiagnosticShowcaseReplay() {
  if (state.busy) return;
  setShowcaseLauncherBusy(true, elements.showcaseDiagnosticStartButton, "正在建立洗衣机回放…");
  try {
    const response = await fetch("/api/v2/showcase-replays/diagnostic", { method: "POST" });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticCase = data.case;
    state.latestAgentMessage = data.agent_message;
    state.diagnosticRetryRecording = null;
    state.pendingFile = null;
    state.measurementMode = "public";
    elements.latestResult.hidden = true;
    navigateTo(`/app/cases/${encodeURIComponent(data.case.case_id)}`);
    renderDiagnosticCase(data.agent_message);
    switchMeasurementMode("public");
    await loadCaseHistory();
    showToast("洗衣机零等待诊断已就绪：点击两次即可看到报告");
  } catch (error) {
    showToast(error.message || "无法建立诊断回放。", true);
  } finally {
    setShowcaseLauncherBusy(false);
    restoreShowcaseLauncherLabels();
  }
}

async function startExplorationShowcaseReplay() {
  if (state.busy) return;
  setShowcaseLauncherBusy(true, elements.showcaseExplorationStartButton, "正在建立光学回放…");
  try {
    const response = await fetch("/api/v2/showcase-replays/exploration", { method: "POST" });
    if (!response.ok) throw new Error(await readApiError(response));
    state.generalCase = await response.json();
    state.generalAcquisitionPlan = null;
    state.generalPublicComponents = null;
    state.generalPublicRun = null;
    state.generalError = "";
    await Promise.all([
      refreshGeneralHistory(),
      loadExplorationHistory(),
      loadGeneralAcquisitionPlan(state.generalCase.case_id),
    ]);
    navigateTo(`/app/explore/general/runs/${encodeURIComponent(state.generalCase.case_id)}`);
    renderGeneralExploration();
    showToast("光学零等待探索已就绪：四次点击形成对比图与报告");
  } catch (error) {
    showToast(error.message || "无法建立探索回放。", true);
  } finally {
    setShowcaseLauncherBusy(false);
    restoreShowcaseLauncherLabels();
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("服务未就绪");
    const data = await response.json();
    elements.healthPill.classList.add("online");
    elements.healthText.textContent = "系统在线";
    elements.modelBadge.textContent = compactModelName(data.model);
  } catch (error) {
    elements.healthPill.classList.add("offline");
    elements.healthText.textContent = "连接失败";
    elements.modelBadge.textContent = "MODEL OFFLINE";
  }
}

async function loadWorkspaceState() {
  await Promise.allSettled([
    loadSettings(true),
    loadModelProfiles(),
    loadWorkbenchReports(),
    loadAgentRuns(true),
    loadCaseHistory(),
    loadExplorationHistory(),
    loadSessionHistory(),
    loadExplorations(),
    loadSensorCapabilities(),
    loadPublicReplays(),
    loadInvestigationWorkspace(),
    loadGeneralExplorationWorkspace(),
  ]);
  renderSessions();
  renderEvidenceWorkbenchLibrary();
  renderSelectedEvidence();
}

const SENSOR_LABELS = {
  accelerometer: "加速度计",
  gyroscope: "陀螺仪",
  magnetometer: "磁力计",
  light: "光线",
  pressure: "气压",
  proximity: "接近距离",
  microphone: "麦克风",
  location: "GPS / 位置",
  bluetooth: "Bluetooth",
};

const SENSOR_TASK_FALLBACKS = {
  accelerometer: { quantity: "三轴加速度、振动 RMS 与主频", experiment: "“加速度（不含重力）”或“加速度”实验" },
  gyroscope: { quantity: "三轴角速度", experiment: "输入为“陀螺仪 / gyroscope”的实验" },
  magnetometer: { quantity: "三轴磁场强度", experiment: "输入为“磁力计 / magnetic field”的实验" },
  light: { quantity: "相对照度", experiment: "输入为“光线 / light”的实验" },
  pressure: { quantity: "气压与相对高度变化", experiment: "输入为“气压 / pressure”的实验" },
  proximity: { quantity: "接近状态或距离", experiment: "输入为“接近传感器 / proximity”的实验" },
  microphone: { quantity: "相对声音幅值或频谱", experiment: "输入为“麦克风 / audio”的声音实验（例如“音频幅值 / Audio amplitude”）" },
  location: { quantity: "位置、轨迹或速度", experiment: "输入为“位置 / location (GPS)”的实验" },
  bluetooth: { quantity: "外部设备测量通道", experiment: "与外部 BLE 设备协议匹配的自定义实验" },
};

const GENERAL_TASK_ACTION_LABELS = {
  collect_condition: "采集条件证据",
  collect_supporting_sensor: "补充联合传感器",
  replicate_condition: "重复测量",
  correct_condition: "纠正低质量测量",
  probe_optional_sensor: "探查可选传感器",
  probe_optional_condition: "探查附加对照条件",
};

const GENERAL_TASK_REASON_LABELS = {
  initial_baseline: "先建立基准条件",
  missing_condition: "补齐尚未覆盖的条件",
  missing_supporting_sensor: "补齐联合证据",
  replication_required: "增加重复以判断波动",
  quality_correction: "上一条证据未通过质量门",
  optional_sensor_probe: "问题语义支持一次可选传感器探查",
  optional_condition_probe: "问题语义支持一次附加对照探查",
};

const GENERAL_PLANNER_RATIONALE_LABELS = {
  maximize_condition_coverage: "优先覆盖尚未测量的条件",
  balance_sensor_coverage: "平衡多传感器证据覆盖",
  replicate_highest_uncertainty: "优先复测当前波动最大的槽位",
  resolve_quality_failure: "先修复未通过质量门的测量",
  select_relevant_optional_sensor: "选择与问题最相关的可选传感器",
  select_relevant_control_condition: "选择能区分解释的附加对照",
  prefer_protocol_default: "模型不可用或无增益时采用冻结默认项",
};

const GENERAL_DECISION_SOURCE_LABELS = {
  server_initial: "服务端创建首个任务",
  deterministic_policy: "确定性强基线选择",
  deterministic_fallback: "安全回退选择",
  bounded_agent: "受限 Agent 选择",
  reasoning_agent: "证据推理 Agent 追测",
  user_checkpoint: "用户在歧义检查点继续",
};

const GENERAL_INFORMATION_GOAL_LABELS = {
  condition_coverage: "补齐条件覆盖",
  sensor_coverage: "补齐传感器覆盖",
  uncertainty_reduction: "降低重复不确定性",
  quality_recovery: "恢复证据质量",
  hypothesis_discrimination: "区分竞争解释",
  control_challenge: "检验附加对照",
};

const GENERAL_SERVER_FACT_LABELS = {
  "lowest-effort": "当前候选成本最低",
  "highest-observed-uncertainty": "当前观测不确定性最高",
  "no-observed-uncertainty": "尚无可比较的不确定性",
  "privacy-sensitive-derived-metric": "涉及隐私敏感派生量",
  "optional-observation": "一次可选辅助观察",
  "registered-repetition": "注册重复测量",
  "valid-high-quality-evidence": "已有高质量有效证据",
  "valid-medium-quality-evidence": "已有中等质量有效证据",
  "valid-low-quality-evidence": "已有低质量有效证据",
  "invalid-evidence": "存在未通过门禁的证据",
  "comparison-higher": "比较条件方向更高",
  "comparison-lower": "比较条件方向更低",
  "within-relative-deadband": "条件差异落在相对死区内",
};

const GENERAL_EXPECTED_RELATION_LABELS = {
  comparison_higher: "预期比较条件高于参考",
  comparison_lower: "预期比较条件低于参考",
  within_relative_deadband: "预期落在相对死区内",
  different_unspecified: "预期存在差异，方向未注册",
};

const GENERAL_HYPOTHESIS_MATCH_LABELS = {
  not_observed: "尚未观测",
  matches_expected: "方向与预测一致",
  conflicts_expected: "方向与预测冲突",
};

const GENERAL_HYPOTHESIS_ASSESSMENT_LABELS = {
  untested: "尚无成对证据",
  observed_prediction_matched: "观测方向与预测一致",
  observed_prediction_conflicted: "观测方向与预测冲突",
  mixed_observations: "不同观测给出混合结果",
};

function taskSensorDetails(task = state.diagnosticCase?.current_task) {
  const sensor = task?.required_sensor || "accelerometer";
  const fallback = SENSOR_TASK_FALLBACKS[sensor] || SENSOR_TASK_FALLBACKS.accelerometer;
  return {
    sensor,
    label: SENSOR_LABELS[sensor] || sensor,
    quantity: task?.measurement_quantity || fallback.quantity,
    experiment: task?.recommended_phyphox_experiment || fallback.experiment,
    analyzerReady: (task?.analyzer_status || "ready") === "ready",
  };
}

function probeSensorKinds(probe) {
  if (!probe) return [];
  const detected = Array.isArray(probe.detected_sensors) ? probe.detected_sensors : [];
  const profiled = probe.sensor_profiles && typeof probe.sensor_profiles === "object"
    ? Object.keys(probe.sensor_profiles)
    : [];
  return [...new Set([...detected, ...profiled].filter((sensor) => typeof sensor === "string" && sensor))];
}

function probeInputText(probe) {
  const sensors = probeSensorKinds(probe);
  return sensors.length
    ? sensors.map((sensor) => SENSOR_LABELS[sensor] || sensor).join(" · ")
    : "未识别输入类型";
}

function probeMatchesTask(probe, task = state.diagnosticCase?.current_task) {
  if (!probe || !task) return false;
  const required = task.required_sensor || "accelerometer";
  const detected = probe.detected_sensors || [];
  if (required === "accelerometer") return probe.compatible || detected.includes(required);
  return detected.includes(required);
}

async function loadExplorations() {
  try {
    const response = await fetch("/api/v1/explorations");
    if (!response.ok) throw new Error(await readApiError(response));
    state.explorations = await response.json();
    renderExplorations();
  } catch (error) {
    elements.explorationEmpty.hidden = false;
    elements.explorationEmpty.textContent = `探索目录读取失败：${error.message}`;
  }
}

const INVESTIGATION_SOURCE_LABELS = {
  protocol: "预注册协议",
  deterministic: "确定性规则",
  agent: "受限 Agent",
  fallback: "安全回退",
};

const INVESTIGATION_ROLE_LABELS = {
  background: "环境光对照",
  condition: "新距离条件",
  replication: "同距离重复",
  correction: "质量纠偏",
  exploration: "探索任务",
};

const INVESTIGATION_TOOL_LABELS = {
  "sensor_analysis.light.v2": "检查单次光照记录",
  aggregate_light_conditions: "聚合同距离重复",
  select_next_design_point: "选择下一测量距离",
  fit_light_distance_decay: "拟合距离衰减",
  sample_light_fit_series: "生成受控图表数据",
};

const PLANNER_RATIONALE_LABELS = {
  maximize_log_span: "扩大对数距离跨度",
  preserve_signal_to_background: "保持信号高于背景",
  reduce_saturation_risk: "降低传感器饱和风险",
  respect_user_constraint: "遵守用户现场约束",
  prefer_protocol_default: "使用预注册安全默认值",
};

const PLANNER_TRANSPORT_LABELS = {
  not_attempted: "未调用模型",
  function_tool: "Function Tool",
  validated_json_text: "严格 JSON 兼容模式",
};

async function loadInvestigationWorkspace() {
  try {
    const [protocolResponse, historyResponse, recordingsResponse] = await Promise.all([
      fetch("/api/v2/experiment-protocols"),
      fetch("/api/v2/investigations"),
      fetch("/api/v2/recordings"),
    ]);
    for (const response of [protocolResponse, historyResponse, recordingsResponse]) {
      if (!response.ok) throw new Error(await readApiError(response));
    }
    state.experimentProtocols = await protocolResponse.json();
    state.investigationHistory = await historyResponse.json();
    state.sensorRecordings = await recordingsResponse.json();
    renderActiveInvestigations();
    const route = routeState();
    if (route.exploreView === "run" && route.investigationId) {
      await openInvestigation(route.investigationId);
    } else {
      state.investigation = null;
      renderInvestigation();
    }
  } catch (error) {
    setInvestigationError(`可执行实验读取失败：${error.message}`);
  }
}

function renderActiveInvestigations() {
  const active = state.investigationHistory.filter((item) => !item.status.startsWith("completed"));
  elements.explorationActiveRunsEmpty.hidden = active.length > 0;
  elements.explorationActiveRuns.innerHTML = active.map((item) => `
    <article class="exploration-active-row">
      <div><span>${escapeHtml(SENSOR_LABELS[item.primary_sensor] || item.primary_sensor)}</span><b>${escapeHtml(item.title)}</b><small>${item.evidence_count} 项证据 · 更新于 ${escapeHtml(formatDateTime(item.updated_at))}</small></div>
      <button class="button button-secondary" type="button" data-continue-investigation="${escapeHtml(item.case_id)}">继续实验</button>
    </article>`).join("");
  elements.explorationActiveRuns.querySelectorAll("[data-continue-investigation]").forEach((button) => {
    button.addEventListener("click", () => {
      navigateTo(`/app/explore/runs/${encodeURIComponent(button.dataset.continueInvestigation)}`);
    });
  });
}

async function loadGeneralExplorationWorkspace() {
  try {
    const [capabilitiesResponse, readinessResponse, historyResponse] = await Promise.all([
      fetch("/api/v2/general-exploration-capabilities"),
      fetch("/api/v2/general-exploration-readiness"),
      fetch("/api/v2/general-explorations"),
    ]);
    for (const response of [capabilitiesResponse, readinessResponse, historyResponse]) {
      if (!response.ok) throw new Error(await readApiError(response));
    }
    state.generalCapabilities = await capabilitiesResponse.json();
    state.generalReadiness = await readinessResponse.json();
    state.generalHistory = await historyResponse.json();
    renderGeneralReadiness();
    populateGeneralSensorOptions();
    const route = routeState();
    if (route.exploreView === "general_run" && route.generalCaseId) {
      await openGeneralExploration(route.generalCaseId);
    }
  } catch (error) {
    state.generalError = `自由探索读取失败：${error.message}`;
    elements.generalCreateStatus.dataset.state = "error";
    elements.generalCreateStatus.textContent = state.generalError;
  }
}

function renderGeneralReadiness() {
  const readiness = state.generalReadiness;
  if (!readiness) return;
  const metrics = readiness.latest_live_metrics;
  elements.generalReadinessBoundary.dataset.maturity = readiness.maturity;
  elements.generalReadinessBadge.textContent = readiness.general_agent_beta
    ? "GENERAL AGENT BETA"
    : "BOUNDED AGENT PREVIEW";
  elements.generalReadinessSummary.textContent = readiness.phyphox_validated
    ? "已完成真实手机 Gate C；仍以单次报告中的证据边界为准。"
    : readiness.general_agent_beta
      ? "已通过模型闭环 Gate E/H；phyphox 接口兼容，但尚无真实手机 Gate C，不能称 Agent Ready。"
      : "最新产品 HTTP 新分布尚未通过 Gate E/H；受限 Agent 可继续试用，但当前不标为 Beta 或 Agent Ready。";
  elements.generalReadinessGates.textContent = metrics
    ? `Phase ${readiness.evaluation_phase} 真实模型：结构化 ${Math.round(metrics.structured_compiler_rate * 1000) / 10}% · 语义 ${Math.round(metrics.semantic_compiler_contract_rate * 1000) / 10}% · 澄清恢复 ${Math.round(metrics.clarification_recovery_rate * 1000) / 10}% · 闭环 ${Math.round(metrics.product_loop_contract_rate * 1000) / 10}% · 动态分支 ${Math.round(metrics.dynamic_counterfactual_pair_rate * 1000) / 10}% · 重复 ${Math.round(metrics.repeat_consistency_rate * 1000) / 10}% · Compiler 回退 ${Math.round(metrics.compiler_fallback_rate * 1000) / 10}% · 相对强工作流 +${Math.round(metrics.agent_capability_gain * 1000) / 10}pp · 安全失败 ${metrics.safety_failure_count} ｜ Gate C ${readiness.gate_c} · Gate E ${readiness.gate_e} · Gate H ${readiness.gate_h}`
    : `Gate C ${readiness.gate_c} · Gate E ${readiness.gate_e} · Gate H ${readiness.gate_h} · Agent Ready ${readiness.agent_ready}`;
}

function generalNumericCapabilities() {
  return state.generalCapabilities.filter((item) => (
    item.sensor !== "bluetooth"
    && item.supports_file_upload
    && item.supports_bounded_agent
    && item.metrics?.length
  ));
}

function populateGeneralSensorOptions() {
  const capabilities = generalNumericCapabilities();
  const previousPrimary = elements.generalPrimarySensor.value;
  const previousSupporting = elements.generalSupportingSensor.value;
  const previousOptional = elements.generalOptionalSensor.value;
  const previousOptional2 = elements.generalOptionalSensor2.value;
  const options = capabilities.map((item) => (
    `<option value="${escapeHtml(item.sensor)}">${escapeHtml(SENSOR_LABELS[item.sensor] || item.sensor)}</option>`
  )).join("");
  const previousPreferred = elements.generalPreferredSensor.value;
  elements.generalPreferredSensor.innerHTML = `<option value="">不限定；允许 Agent 组合受支持传感器</option>${options}`;
  if ([...elements.generalPreferredSensor.options].some((item) => item.value === previousPreferred)) {
    elements.generalPreferredSensor.value = previousPreferred;
  }
  elements.generalPrimarySensor.innerHTML = `<option value="">请选择主要传感器</option>${options}`;
  const defaultPrimary = capabilities.some((item) => item.sensor === previousPrimary)
    ? previousPrimary
    : (capabilities.find((item) => item.sensor === "light")?.sensor || "");
  elements.generalPrimarySensor.value = defaultPrimary;
  const secondary = () => `<option value="">不使用</option>${capabilities
    .map((item) => `<option value="${escapeHtml(item.sensor)}">${escapeHtml(SENSOR_LABELS[item.sensor] || item.sensor)}</option>`)
    .join("")}`;
  elements.generalSupportingSensor.innerHTML = secondary();
  if ([...elements.generalSupportingSensor.options].some((item) => item.value === previousSupporting)) {
    elements.generalSupportingSensor.value = previousSupporting;
  }
  elements.generalOptionalSensor.innerHTML = secondary();
  if ([...elements.generalOptionalSensor.options].some((item) => item.value === previousOptional)) {
    elements.generalOptionalSensor.value = previousOptional;
  }
  elements.generalOptionalSensor2.innerHTML = secondary();
  if ([...elements.generalOptionalSensor2.options].some((item) => item.value === previousOptional2)) {
    elements.generalOptionalSensor2.value = previousOptional2;
  }
  updateGeneralSensorForm();
  updateGeneralCompilerAvailability();
}

function generalCapability(sensor) {
  return state.generalCapabilities.find((item) => item.sensor === sensor) || null;
}

function generalMetricForSensor(sensor) {
  const capability = generalCapability(sensor);
  const compiled = state.generalCompiledDraft?.sensor_intents?.find((item) => item.sensor === sensor);
  return capability?.metrics?.find((item) => item.metric_key === compiled?.metric_key)
    || capability?.metrics?.[0]
    || null;
}

const GENERAL_CONDITION_CLARIFICATION_CODES = [
  "missing-single-variable",
  "missing-reference-or-comparison",
];
const GENERAL_FREEFORM_CLARIFICATION_CODES = [
  "ambiguous-primary-observable",
  "privacy-boundary-not-acknowledged",
  "unsupported-observable",
];

function generalClarificationRequirements() {
  const codes = [...new Set(state.generalCompileResult?.blocker_codes || [])];
  const conditionCodes = codes.filter((code) => GENERAL_CONDITION_CLARIFICATION_CODES.includes(code));
  const needsMechanism = codes.includes("ambiguous-competing-explanations");
  const freeformCodes = codes.filter((code) => GENERAL_FREEFORM_CLARIFICATION_CODES.includes(code));
  return {
    conditionCodes,
    needsCondition: conditionCodes.length > 0,
    needsMechanism,
    freeformCodes,
    actionableCount: Number(conditionCodes.length > 0) + Number(needsMechanism) + freeformCodes.length,
  };
}

function updateGeneralClarificationForm() {
  const requirements = generalClarificationRequirements();
  elements.generalConditionClarification.hidden = !requirements.needsCondition;
  elements.generalMechanismClarification.hidden = !requirements.needsMechanism;
  elements.generalFreeformClarification.hidden = requirements.freeformCodes.length === 0;
  elements.generalClarificationContracts.dataset.panelCount = String(
    Number(requirements.needsCondition)
    + Number(requirements.needsMechanism)
    + Number(requirements.freeformCodes.length > 0)
  );
  if (requirements.needsMechanism && !requirements.needsCondition) {
    elements.generalKeepMechanisms.checked = true;
    elements.generalKeepMechanisms.disabled = true;
    elements.generalMechanismClarificationHint.textContent = "必须明确填写两个不同的物理机制";
  } else {
    elements.generalKeepMechanisms.disabled = false;
    elements.generalMechanismClarificationHint.textContent = "可选；不勾选时，原问题里的其他操作会被丢弃";
  }
  const mechanismEnabled = requirements.needsMechanism && elements.generalKeepMechanisms.checked;
  elements.generalMechanismFields.hidden = !mechanismEnabled;
  elements.generalFirstMechanism.disabled = !mechanismEnabled;
  elements.generalSecondMechanism.disabled = !mechanismEnabled;
}

function updateGeneralCompilerAvailability() {
  const ready = elements.generalNaturalQuestion.value.trim().length >= 5;
  const preferred = elements.generalPreferredSensor.value;
  const microphoneAllowed = !preferred || preferred === "microphone";
  const locationAllowed = !preferred || preferred === "location";
  elements.generalCompilerMicrophonePrivacy.disabled = !microphoneAllowed;
  elements.generalCompilerLocationPrivacy.disabled = !locationAllowed;
  if (!microphoneAllowed) elements.generalCompilerMicrophonePrivacy.checked = false;
  if (!locationAllowed) elements.generalCompilerLocationPrivacy.checked = false;
  const missingRequiredPrivacy = (
    (preferred === "microphone" && !elements.generalCompilerMicrophonePrivacy.checked)
    || (preferred === "location" && !elements.generalCompilerLocationPrivacy.checked)
  );
  elements.generalCompileButton.disabled = state.busy || !ready || missingRequiredPrivacy;
  const requirements = generalClarificationRequirements();
  const conditionReady = !requirements.needsCondition || (
    elements.generalClarificationVariable.value.trim().length >= 2
    && elements.generalClarificationReference.value.trim().length >= 1
    && elements.generalClarificationComparison.value.trim().length >= 1
    && elements.generalClarificationReference.value.trim().toLocaleLowerCase()
      !== elements.generalClarificationComparison.value.trim().toLocaleLowerCase()
  );
  const mechanismRequired = requirements.needsMechanism && (
    !requirements.needsCondition || elements.generalKeepMechanisms.checked
  );
  const mechanismReady = !mechanismRequired || (
    elements.generalFirstMechanism.value.trim().length >= 3
    && elements.generalSecondMechanism.value.trim().length >= 3
    && elements.generalFirstMechanism.value.trim().toLocaleLowerCase()
      !== elements.generalSecondMechanism.value.trim().toLocaleLowerCase()
  );
  const freeformReady = requirements.freeformCodes.length === 0
    || elements.generalClarificationAnswer.value.trim().length >= 3;
  const canRetryClarification = state.generalCompileResult?.status === "needs_clarification"
    && requirements.actionableCount > 0
    && Boolean(state.generalCompileResult?.clarification_receipt?.receipt_id)
    && conditionReady && mechanismReady && freeformReady;
  elements.generalClarificationRetry.disabled = state.busy || !canRetryClarification;
}

function applyGeneralCompiledDraft(draft) {
  state.generalCompiledDraft = draft;
  elements.generalTitle.value = draft.title || "我的自由探索";
  elements.generalQuestion.value = draft.question || elements.generalNaturalQuestion.value.trim();
  elements.generalIndependentVariable.value = draft.independent_variable || "";
  const required = (draft.conditions || []).filter((item) => item.activation === "required");
  elements.generalReferenceLabel.value = required[0]?.label || "";
  elements.generalComparisonLabel.value = required[1]?.label || "";
  elements.generalOptionalControl.value = (draft.conditions || []).find(
    (item) => item.activation === "optional_control"
  )?.label || "";
  elements.generalAlignment.value = draft.alignment || "sequential";
  const primary = (draft.sensor_intents || []).find((item) => item.role === "primary");
  const supporting = (draft.sensor_intents || []).find(
    (item) => item.role === "supporting" && item.activation === "required"
  );
  const optional = (draft.sensor_intents || []).filter(
    (item) => item.role === "supporting" && item.activation === "optional_probe"
  );
  elements.generalPrimarySensor.value = primary?.sensor || "";
  elements.generalSupportingSensor.value = supporting?.sensor || "";
  elements.generalOptionalSensor.value = optional[0]?.sensor || "";
  elements.generalOptionalSensor2.value = optional[1]?.sensor || "";
  elements.generalPrivacyConfirm.checked = false;
  elements.generalHypothesisConfirm.checked = false;
  updateGeneralSensorForm();
}

function renderGeneralCompileResult(result) {
  const messages = [...(result.user_messages || []), ...(result.clarification_questions || [])];
  const runtime = result.runtime || {};
  const statusLabel = ({ draft_ready: "草案可审阅", needs_clarification: "需要补充确认", rejected: "已安全拒绝" })[result.status] || result.status;
  const receiptStatus = result.receipt
    ? "已签发一次性草案审计凭证"
    : result.clarification_receipt
      ? "已绑定本账号、当前题目的一次性澄清凭证"
      : "未签发 Agent 凭证";
  const hypotheses = result.draft?.hypotheses || [];
  const hypothesisSummary = hypotheses.length
    ? `<div class="general-compiler-hypotheses"><b>${hypotheses.length} 个未验证竞争假设</b>${hypotheses.map((hypothesis) => `<span>${escapeHtml(hypothesis.hypothesis_id)} · ${escapeHtml(hypothesis.statement_untrusted)}</span>`).join("")}<small>若你改写问题或单一变量，旧假设图会自动丢弃并等待重新生成。</small></div>`
    : "";
  elements.generalCompileResult.hidden = false;
  elements.generalCompileResult.dataset.status = result.status;
  const clarificationRequirements = generalClarificationRequirements();
  const needsClarification = result.status === "needs_clarification";
  const hasActionableClarification = needsClarification
    && clarificationRequirements.actionableCount > 0
    && Boolean(result.clarification_receipt?.receipt_id);
  elements.generalCompilerClarification.hidden = !hasActionableClarification;
  if (!needsClarification) {
    elements.generalClarificationVariable.value = "";
    elements.generalClarificationReference.value = "";
    elements.generalClarificationComparison.value = "";
    elements.generalKeepMechanisms.checked = false;
    elements.generalFirstMechanism.value = "";
    elements.generalSecondMechanism.value = "";
    elements.generalClarificationAnswer.value = "";
  }
  updateGeneralClarificationForm();
  elements.generalCompileResult.innerHTML = `
    <header><b>${escapeHtml(statusLabel)}</b><span>${escapeHtml(result.source === "bounded_agent" ? "BOUNDED AGENT" : "SAFE FALLBACK")}</span></header>
    ${messages.length ? `<ul>${messages.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : "<p>服务端严格校验已通过；草案已填入下方表单。</p>"}
    ${hypothesisSummary}
    <small>模型请求 ${escapeHtml(runtime.model_requests ?? 0)} 次 · 工具调用 ${escapeHtml(runtime.tool_calls ?? 0)} 次 · fallback ${escapeHtml(runtime.fallback_reason || "none")} · ${receiptStatus} · 未创建实验</small>`;
}

async function compileGeneralQuestion() {
  if (state.busy) return;
  const question = elements.generalNaturalQuestion.value.trim();
  if (question.length < 5) return;
  const preferred = elements.generalPreferredSensor.value;
  const privacyAcknowledged = [
    ...((!preferred || preferred === "microphone") && elements.generalCompilerMicrophonePrivacy.checked ? ["microphone"] : []),
    ...((!preferred || preferred === "location") && elements.generalCompilerLocationPrivacy.checked ? ["location"] : []),
  ];
  const clarificationAnswer = elements.generalClarificationAnswer.value.trim();
  const requirements = generalClarificationRequirements();
  const clarificationAnswers = clarificationAnswer
    ? requirements.freeformCodes.map((reasonCode) => ({
      reason_code: reasonCode,
      answer_untrusted: clarificationAnswer,
    }))
    : [];
  const conditionResolution = requirements.needsCondition ? {
    schema_version: "1.0",
    reason_codes: requirements.conditionCodes,
    independent_variable: elements.generalClarificationVariable.value.trim(),
    reference_label: elements.generalClarificationReference.value.trim(),
    comparison_label: elements.generalClarificationComparison.value.trim(),
    unselected_alternatives: "discard_as_experimental_conditions",
  } : null;
  const mechanismResolution = requirements.needsMechanism && elements.generalKeepMechanisms.checked ? {
    schema_version: "1.0",
    reason_code: "ambiguous-competing-explanations",
    first_mechanism_label_untrusted: elements.generalFirstMechanism.value.trim(),
    second_mechanism_label_untrusted: elements.generalSecondMechanism.value.trim(),
  } : null;
  const isClarificationRetry = Boolean(
    clarificationAnswers.length || conditionResolution || mechanismResolution
  );
  const clarificationReceiptId = isClarificationRetry
    ? state.generalCompileResult?.clarification_receipt?.receipt_id || null
    : null;
  const priorClarificationReceipt = state.generalCompileResult?.clarification_receipt || null;
  state.busy = true;
  updateGeneralCompilerAvailability();
  updateGeneralSensorForm();
  elements.generalCompileStatus.dataset.state = "loading";
  elements.generalCompileStatus.textContent = "正在生成严格 JSON 草案；此时不会创建实验或连接手机…";
  try {
    const response = await fetch("/api/v2/general-explorations/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        context: state.generalRoutedContext,
        clarification_answers: clarificationAnswers,
        condition_resolution: conditionResolution,
        mechanism_resolution: mechanismResolution,
        clarification_receipt_id: clarificationReceiptId,
        preferred_sensors: preferred ? [preferred] : [],
        privacy_acknowledged_sensors: privacyAcknowledged,
        use_agent: true,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.generalCompileResult = await response.json();
    if (
      state.generalCompileResult.status === "needs_clarification"
      && !state.generalCompileResult.clarification_receipt
      && state.generalCompileResult.runtime?.fallback_reason !== "none"
      && priorClarificationReceipt
    ) {
      state.generalCompileResult.clarification_receipt = priorClarificationReceipt;
    }
    if (state.generalCompileResult.draft) applyGeneralCompiledDraft(state.generalCompileResult.draft);
    renderGeneralCompileResult(state.generalCompileResult);
    const ready = state.generalCompileResult.status === "draft_ready";
    elements.generalCompileStatus.dataset.state = ready ? "ready" : (state.generalCompileResult.status === "rejected" ? "error" : "warning");
    elements.generalCompileStatus.textContent = ready
      ? "草案已填入下方编辑器。请检查条件、传感器与隐私设置，再单独创建协议。"
      : hasActionableClarification
        ? "系统没有自动创建实验；请按上方具体问题补充后重新生成。"
        : "Agent 草案尚未冻结；你仍可完整填写下方必填项，直接创建受限的相对比较协议。";
  } catch (error) {
    elements.generalCompileStatus.dataset.state = "error";
    elements.generalCompileStatus.textContent = `草案生成失败：${error.message}；没有创建实验。`;
  } finally {
    state.busy = false;
    updateGeneralCompilerAvailability();
    updateGeneralSensorForm();
  }
}

function resetGeneralCompilerClarification() {
  state.generalCompileResult = null;
  elements.generalCompileResult.hidden = true;
  elements.generalCompilerClarification.hidden = true;
  elements.generalConditionClarification.hidden = true;
  elements.generalClarificationVariable.value = "";
  elements.generalClarificationReference.value = "";
  elements.generalClarificationComparison.value = "";
  elements.generalMechanismClarification.hidden = true;
  elements.generalKeepMechanisms.checked = false;
  elements.generalFirstMechanism.value = "";
  elements.generalSecondMechanism.value = "";
  elements.generalFreeformClarification.hidden = true;
  elements.generalClarificationAnswer.value = "";
  updateGeneralCompilerAvailability();
}

function updateGeneralSensorForm() {
  const primary = elements.generalPrimarySensor.value;
  let supporting = elements.generalSupportingSensor.value;
  let optional = elements.generalOptionalSensor.value;
  let optional2 = elements.generalOptionalSensor2.value;
  if (supporting === primary) {
    elements.generalSupportingSensor.value = "";
    supporting = "";
  }
  if (optional === primary || optional === supporting) {
    elements.generalOptionalSensor.value = "";
    optional = "";
  }
  if (optional2 === primary || optional2 === supporting || optional2 === optional) {
    elements.generalOptionalSensor2.value = "";
    optional2 = "";
  }
  if (optional2) elements.generalOptionalControl.value = "";
  elements.generalOptionalControl.disabled = Boolean(optional2);
  elements.generalOptionalControl.title = optional2
    ? "两个候选探测不能共享一个数值激活对照；请先只比较两个可选传感器。"
    : "";
  for (const option of elements.generalPrimarySensor.options) {
    option.disabled = Boolean(option.value && [supporting, optional, optional2].includes(option.value));
  }
  for (const option of elements.generalSupportingSensor.options) {
    option.disabled = Boolean(option.value && [primary, optional, optional2].includes(option.value));
  }
  for (const option of elements.generalOptionalSensor.options) {
    option.disabled = Boolean(option.value && [primary, supporting, optional2].includes(option.value));
  }
  for (const option of elements.generalOptionalSensor2.options) {
    option.disabled = Boolean(option.value && [primary, supporting, optional].includes(option.value));
  }
  const selected = [primary, supporting, optional, optional2].filter(Boolean);
  const simulatedRehearsal = elements.generalExecutionMode.value === "protocol_emulator";
  elements.generalExecutionBoundary.textContent = simulatedRehearsal
    ? "模拟演练会运行生产分析器、多轮设计、动态终止与报告，但不读取手机，且永远不计现实证据或 Gate C。"
    : "现实实验可以形成当前现场证据；是否达到真机 Gate C 仍由独立评测决定。";
  const simultaneousAllowed = Boolean(
    supporting && !optional && !optional2 && !elements.generalOptionalControl.value.trim()
  );
  const simultaneousOption = [...elements.generalAlignment.options].find(
    (item) => item.value === "simultaneous"
  );
  if (simultaneousOption) simultaneousOption.disabled = !simultaneousAllowed;
  if (!simultaneousAllowed && elements.generalAlignment.value === "simultaneous") {
    elements.generalAlignment.value = "sequential";
  }
  const lines = selected.map((sensor) => {
    const capability = generalCapability(sensor);
    const metric = generalMetricForSensor(sensor);
    const mode = [optional, optional2].includes(sensor) ? "Agent 候选；最多二选一" : "必需重复";
    return `<div><b>${escapeHtml(SENSOR_LABELS[sensor] || sensor)}</b><span>${escapeHtml(metric?.label || "无可用分析量")} · ${escapeHtml(metric?.unit || "—")} · ${mode}</span></div>`;
  });
  elements.generalSensorSummary.innerHTML = lines.length
    ? lines.join("")
    : "选择主要传感器后，将显示服务端锁定的分析量。";
  const sensitive = selected.some((sensor) => generalCapability(sensor)?.privacy_ack_required);
  elements.generalPrivacyConfirm.parentElement.hidden = !sensitive;
  const hypothesisReviewRequired = updateGeneralHypothesisReview();
  const requiredFieldsReady = Boolean(
    elements.generalTitle.value.trim()
    && elements.generalQuestion.value.trim().length >= 5
    && elements.generalIndependentVariable.value.trim()
    && elements.generalReferenceLabel.value.trim()
    && elements.generalComparisonLabel.value.trim()
    && primary
  );
  elements.generalCreateButton.disabled = state.busy || !requiredFieldsReady
    || (sensitive && !elements.generalPrivacyConfirm.checked)
    || (hypothesisReviewRequired && !elements.generalHypothesisConfirm.checked);
}

function generalSensorIntent(sensor, role, activation = "required") {
  const capability = generalCapability(sensor);
  const metric = generalMetricForSensor(sensor);
  if (!capability || !metric) throw new Error(`${SENSOR_LABELS[sensor] || sensor} 没有可执行分析量。`);
  return {
    sensor,
    role,
    activation,
    metric_key: metric.metric_key,
    metric_unit: metric.unit,
    measurement_purpose: `${metric.label}用于回答当前条件比较；${activation === "optional_probe" ? "仅在能区分竞争解释时探测一次" : "作为必需证据重复测量"}。`,
  };
}

function generalCompiledDraftMatchesForm(sensorIntents) {
  const compiled = state.generalCompiledDraft;
  const compiledSensors = compiled?.sensor_intents || [];
  const requiredConditions = (compiled?.conditions || []).filter((item) => item.activation === "required");
  const optionalCondition = (compiled?.conditions || []).find((item) => item.activation === "optional_control");
  return Boolean(compiled)
    && elements.generalTitle.value.trim() === compiled.title
    && elements.generalQuestion.value.trim() === compiled.question
    && elements.generalIndependentVariable.value.trim() === compiled.independent_variable
    && elements.generalReferenceLabel.value.trim() === requiredConditions[0]?.label
    && elements.generalComparisonLabel.value.trim() === requiredConditions[1]?.label
    && elements.generalOptionalControl.value.trim() === (optionalCondition?.label || "")
    && elements.generalAlignment.value === compiled.alignment
    && compiledSensors.length === sensorIntents.length
    && compiledSensors.every((intent) => sensorIntents.some((selectedIntent) => (
      selectedIntent.sensor === intent.sensor
      && selectedIntent.role === intent.role
      && selectedIntent.activation === intent.activation
      && selectedIntent.metric_key === intent.metric_key
      && selectedIntent.metric_unit === intent.metric_unit
    )));
}

function updateGeneralHypothesisReview() {
  const primary = elements.generalPrimarySensor.value;
  const intents = primary ? [generalSensorIntent(primary, "primary")] : [];
  const supporting = elements.generalSupportingSensor.value;
  const optional = elements.generalOptionalSensor.value;
  const optional2 = elements.generalOptionalSensor2.value;
  if (supporting) intents.push(generalSensorIntent(supporting, "supporting"));
  if (optional) intents.push(generalSensorIntent(optional, "supporting", "optional_probe"));
  if (optional2) intents.push(generalSensorIntent(optional2, "supporting", "optional_probe"));
  const required = state.generalCompileResult?.status === "draft_ready"
    && Boolean(state.generalCompileResult?.receipt?.receipt_id)
    && Boolean((state.generalCompiledDraft?.hypotheses || []).length)
    && generalCompiledDraftMatchesForm(intents);
  elements.generalHypothesisConfirm.parentElement.hidden = !required;
  if (!required) elements.generalHypothesisConfirm.checked = false;
  return required;
}

function generalDraftFromForm() {
  const title = elements.generalTitle.value.trim();
  const question = elements.generalQuestion.value.trim();
  const independent = elements.generalIndependentVariable.value.trim();
  const reference = elements.generalReferenceLabel.value.trim();
  const comparison = elements.generalComparisonLabel.value.trim();
  const primary = elements.generalPrimarySensor.value;
  const supporting = elements.generalSupportingSensor.value;
  const optional = elements.generalOptionalSensor.value;
  const optional2 = elements.generalOptionalSensor2.value;
  const optionalControl = elements.generalOptionalControl.value.trim();
  const alignment = elements.generalAlignment.value;
  if (!title || question.length < 5 || !independent || !reference || !comparison || !primary) {
    throw new Error("请完整填写实验名称、问题、单一变量、两个条件和主要传感器。");
  }
  const selected = [primary, supporting, optional, optional2].filter(Boolean);
  if (new Set(selected).size !== selected.length) throw new Error("同一传感器不能同时承担多个角色。");
  const sensitive = selected.filter((sensor) => generalCapability(sensor)?.privacy_ack_required);
  if (sensitive.length && !elements.generalPrivacyConfirm.checked) {
    throw new Error("麦克风或位置实验需要先确认隐私边界。");
  }
  if (optional2 && optionalControl) {
    throw new Error("两个候选探测不能共享一个可选安全对照；请移除候选 B 或安全对照。" );
  }
  if (alignment === "simultaneous" && (!supporting || optional || optional2 || optionalControl)) {
    throw new Error("同步采集需要一个必需辅助传感器，且不能混入可选探测或可选对照。");
  }
  const conditions = [
    {
      condition_id: "reference",
      label: reference,
      factor_level: reference,
      instruction: `在“${reference}”条件下记录；除${independent}外保持其他条件不变。`,
      activation: "required",
    },
    {
      condition_id: "comparison",
      label: comparison,
      factor_level: comparison,
      instruction: `只把${independent}改变为“${comparison}”后记录。`,
      activation: "required",
    },
  ];
  if (optionalControl) conditions.push({
    condition_id: "optional-control",
    label: optionalControl,
    factor_level: "single-safe-control",
    instruction: `${optionalControl}；其他条件保持不变。`,
    activation: "optional_control",
  });
  const sensorIntents = [generalSensorIntent(primary, "primary")];
  if (supporting) sensorIntents.push(generalSensorIntent(supporting, "supporting"));
  if (optional) sensorIntents.push(generalSensorIntent(optional, "supporting", "optional_probe"));
  if (optional2) sensorIntents.push(generalSensorIntent(optional2, "supporting", "optional_probe"));
  const compiledMatches = generalCompiledDraftMatchesForm(sensorIntents);
  const agentDraftReady = state.generalCompileResult?.status === "draft_ready"
    && Boolean(state.generalCompileResult?.receipt?.receipt_id);
  if (
    compiledMatches && agentDraftReady
    && (state.generalCompiledDraft?.hypotheses || []).length
    && !elements.generalHypothesisConfirm.checked
  ) {
    throw new Error("请先检查并确认 Agent 起草的未验证假设图。");
  }
  const draft = compiledMatches && agentDraftReady ? state.generalCompiledDraft : {
      title,
      question,
      objective: "compare_conditions",
      requested_claim: "relative_comparison",
      independent_variable: independent,
      conditions,
      sensor_intents: sensorIntents,
      alignment,
      controls: ["保持同一设备和放置方式。", "保持每次记录时长及非目标环境条件一致。"],
      expected_pattern: "如果自变量确实影响观测结果，主要指标在两个自变量水平之间的变化应稳定高于重复波动。",
      safety_notes: ["只执行低风险、可随时停止且不损坏设备的操作。"],
      privacy_notes: sensitive.length ? ["敏感传感器仅保留所选分析量，不扩大用途。"] : [],
      claim_boundaries: ["只报告当前条件下的描述性相对变化。", "观察到的差异不自动证明因果或绝对校准。"],
  };
  return {
    draft,
    source: elements.generalExecutionMode.value,
    privacy_acknowledged_sensors: sensitive,
    ...(compiledMatches && agentDraftReady
      ? { compilation_receipt_id: state.generalCompileResult.receipt.receipt_id }
      : {}),
  };
}

async function createGeneralExploration() {
  if (state.busy) return;
  let payload;
  try {
    payload = generalDraftFromForm();
  } catch (error) {
    elements.generalCreateStatus.dataset.state = "error";
    elements.generalCreateStatus.textContent = error.message;
    return;
  }
  state.busy = true;
  elements.generalCreateButton.disabled = true;
  elements.generalCreateStatus.dataset.state = "loading";
  elements.generalCreateStatus.textContent = "正在让服务端冻结协议、分析器和安全预算…";
  try {
    const response = await fetch("/api/v2/general-explorations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.generalCase = await response.json();
    await Promise.all([
      refreshGeneralHistory(),
      loadExplorationHistory(),
      loadGeneralAcquisitionPlan(state.generalCase.case_id),
    ]);
    await loadGeneralPublicComponentsForCase(state.generalCase.case_id);
    elements.generalCreateStatus.dataset.state = "success";
    const rehearsal = state.generalCase.protocol.selected_sources?.length === 1
      && state.generalCase.protocol.selected_sources[0] === "protocol_emulator";
    elements.generalCreateStatus.textContent = rehearsal
      ? "模拟演练协议已冻结；下方可直接运行第一轮合成证据。"
      : "现实实验协议已冻结；下方已生成第一项测量任务。";
    navigateTo(`/app/explore/general/runs/${encodeURIComponent(state.generalCase.case_id)}`);
    showToast(rehearsal ? "模拟演练已就绪，可以运行第一轮" : "自由探索协议已冻结，可以绑定第一条记录");
  } catch (error) {
    elements.generalCreateStatus.dataset.state = "error";
    elements.generalCreateStatus.textContent = error.message;
  } finally {
    state.busy = false;
    updateGeneralSensorForm();
  }
}

async function refreshGeneralHistory() {
  const response = await fetch("/api/v2/general-explorations");
  if (!response.ok) throw new Error(await readApiError(response));
  state.generalHistory = await response.json();
}

async function openGeneralExploration(caseId) {
  try {
    const [caseResponse, recordingsResponse] = await Promise.all([
      fetch(`/api/v2/general-explorations/${encodeURIComponent(caseId)}`),
      fetch("/api/v2/recordings"),
    ]);
    if (!caseResponse.ok) throw new Error(await readApiError(caseResponse));
    if (!recordingsResponse.ok) throw new Error(await readApiError(recordingsResponse));
    state.generalCase = await caseResponse.json();
    state.sensorRecordings = await recordingsResponse.json();
    if (state.generalCase.superseded_by_case_id) {
      state.generalAcquisitionPlan = null;
      state.generalPublicComponents = null;
    } else {
      await loadGeneralAcquisitionPlan(caseId);
      if (isGeneralShowcaseCase()) {
        state.generalPublicComponents = null;
        state.generalPublicRun = null;
        state.generalPublicError = "";
      } else {
        await loadGeneralPublicComponentsForCase(caseId);
      }
    }
    state.generalError = "";
    renderGeneralExploration();
  } catch (error) {
    state.generalError = `自由探索读取失败：${error.message}`;
    elements.generalRunMessage.dataset.state = "error";
    elements.generalRunMessage.textContent = state.generalError;
  }
}

async function loadGeneralAcquisitionPlan(caseId) {
  const response = await fetch(`/api/v2/general-explorations/${encodeURIComponent(caseId)}/acquisition-plan`);
  if (!response.ok) throw new Error(await readApiError(response));
  state.generalAcquisitionPlan = await response.json();
  renderGeneralAcquisitionPlan();
  renderGeneralRecordingOptions();
}

async function loadGeneralPublicComponentsForCase(caseId) {
  state.generalPublicComponents = null;
  state.generalPublicRun = null;
  state.generalPublicError = "";
  elements.generalPublicPrivacy.checked = false;
  try {
    const response = await fetch(`/api/v2/general-explorations/${encodeURIComponent(caseId)}/public-components`);
    if (!response.ok) throw new Error(await readApiError(response));
    state.generalPublicComponents = await response.json();
  } catch (error) {
    state.generalPublicError = `公开组件读取失败：${error.message}`;
  }
}

function renderGeneralPublicComponents() {
  const catalog = state.generalPublicComponents;
  const result = state.generalPublicRun;
  const showcase = isGeneralShowcaseCase();
  elements.generalPublicComponents.hidden = !state.generalCase || showcase || Boolean(state.generalCase.superseded_by_case_id);
  if (!state.generalCase || showcase || state.generalCase.superseded_by_case_id) return;
  if (!catalog) {
    elements.generalPublicBoundaries.innerHTML = "";
    elements.generalPublicComponentList.innerHTML = "<p>当前无法读取公开组件。</p>";
    elements.generalPublicStatus.dataset.state = state.generalPublicError ? "error" : "loading";
    elements.generalPublicStatus.textContent = state.generalPublicError || "正在匹配当前协议的公开组件…";
    elements.generalPublicResult.hidden = true;
    return;
  }
  elements.generalPublicBoundaries.innerHTML = catalog.boundary_messages
    .map((message) => `<span>${escapeHtml(message)}</span>`).join("");
  const acknowledged = elements.generalPublicPrivacy.checked;
  elements.generalPublicComponentList.innerHTML = catalog.components.map((component) => `
    <article data-sensor="${escapeHtml(component.sensor)}">
      <header><div><span>${escapeHtml(SENSOR_LABELS[component.sensor] || component.sensor)}</span><b>${escapeHtml(component.title)}</b></div><strong>类比组件</strong></header>
      <p>${escapeHtml(component.supported_scope)}</p>
      <small>${escapeHtml(component.missing_scope)}</small>
      <small>来源：${component.dataset_ids.map(escapeHtml).join(" · ")}</small>
      <button class="button button-secondary" type="button" data-general-public-component="${escapeHtml(component.component_id)}" ${acknowledged && !state.busy && !state.generalPublicRunning ? "" : "disabled"}>运行这个公开 Agent 组件</button>
    </article>`).join("");
  elements.generalPublicStatus.dataset.state = state.generalPublicRunning
    ? "loading"
    : state.generalPublicError
    ? "error"
    : result
      ? "complete"
      : "ready";
  elements.generalPublicStatus.textContent = state.generalPublicRunning
    ? "正在运行独立公开 Agent 组件；不会修改当前案例…"
    : state.generalPublicError
    || (result
      ? `公开组件 ${result.run_id} 已完成；当前案例 revision ${result.case_revision} 未改变。`
      : "选择组件前需逐次确认本地回放边界；运行会单独进入公开历史。" );
  renderGeneralPublicResult(result);
}

function renderGeneralPublicResult(result) {
  elements.generalPublicResult.hidden = !result;
  if (!result) return;
  const findings = result.findings.length
    ? `<h5>公开数据发现</h5><ul>${result.findings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : "";
  const nextLive = result.next_live_measurement
    ? `<h5>若要回答当前现场问题</h5><p>${escapeHtml(result.next_live_measurement)}</p>`
    : "";
  elements.generalPublicResult.innerHTML = `
    <header><div><span>${escapeHtml(SENSOR_LABELS[result.sensor] || result.sensor)} · ${escapeHtml(result.execution_status)}</span><h4>${escapeHtml(result.title)}</h4></div><b>CASE EVIDENCE +0</b></header>
    <p>${escapeHtml(result.summary)}</p>
    <div class="general-public-result-metrics">
      <span><b>${escapeHtml(result.planner_status)}</b>Planner</span>
      <span><b>${result.planner_steps.length}</b>决策</span>
      <span><b>${result.tool_ids.length}</b>工具</span>
      <span><b>${result.evidence_count}</b>公开 evidence</span>
    </div>
    ${findings}
    <h5>仍然不能声称</h5><ul>${result.forbidden_claims.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <h5>不确定性</h5><ul>${result.uncertainties.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    ${nextLive}
    <footer>analogue_only · joint inference NO · Gate C +0 · Agent Ready NO</footer>`;
}

async function runGeneralPublicComponent(componentId) {
  const item = state.generalCase;
  if (!item || state.busy || !elements.generalPublicPrivacy.checked) return;
  state.busy = true;
  state.generalPublicRunning = true;
  state.generalPublicError = "";
  elements.generalPublicStatus.dataset.state = "loading";
  elements.generalPublicStatus.textContent = "正在运行独立公开 Agent 组件；不会修改当前案例…";
  renderGeneralPublicComponents();
  try {
    const response = await fetch(`/api/v2/general-explorations/${encodeURIComponent(item.case_id)}/public-components/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: item.revision,
        component_id: componentId,
        privacy_acknowledged: true,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.generalPublicRun = await response.json();
    await loadExplorationHistory();
    showToast("公开 Agent 组件已完成，当前实验未被改写");
  } catch (error) {
    state.generalPublicError = error.message;
  } finally {
    state.busy = false;
    state.generalPublicRunning = false;
    renderGeneralPublicComponents();
    updateGeneralMeasurementButton();
  }
}

async function refreshGeneralRecordings(options = {}) {
  const quiet = options?.quiet === true;
  elements.generalRefreshRecordings.disabled = true;
  try {
    const caseId = state.generalCase?.case_id;
    const [response, acquisitionResponse] = await Promise.all([
      fetch("/api/v2/recordings"),
      caseId
        ? fetch(`/api/v2/general-explorations/${encodeURIComponent(caseId)}/acquisition-plan`)
        : Promise.resolve(null),
    ]);
    if (!response.ok) throw new Error(await readApiError(response));
    if (acquisitionResponse && !acquisitionResponse.ok) {
      throw new Error(await readApiError(acquisitionResponse));
    }
    state.sensorRecordings = await response.json();
    if (acquisitionResponse) state.generalAcquisitionPlan = await acquisitionResponse.json();
    renderGeneralAcquisitionPlan();
    renderGeneralRecordingOptions();
    if (!quiet) showToast("账号测量记录与取证计划已刷新");
  } catch (error) {
    elements.generalRunMessage.dataset.state = "error";
    elements.generalRunMessage.textContent = error.message;
  } finally {
    elements.generalRefreshRecordings.disabled = false;
  }
}

function renderGeneralAcquisitionPlan() {
  const plan = state.generalAcquisitionPlan;
  if (state.generalCase?.superseded_by_case_id) {
    elements.generalAcquisitionPlan.innerHTML = "<p>这是反馈前的取证计划，已停止继续采集。已有测量仍保留在该版本中。</p>";
    return;
  }
  if (!plan || plan.case_id !== state.generalCase?.case_id) {
    elements.generalAcquisitionPlan.innerHTML = "<p>正在读取服务端取证计划…</p>";
    return;
  }
  const labels = {
    phyphox_live: "LIVE PHONE",
    account_recording: "SAVED RECORD",
    public_analogue: "PUBLIC ANALOGUE",
    protocol_simulator: "SIMULATED REHEARSAL",
  };
  const statusLabels = {
    available: "可执行",
    setup_required: "先连接手机",
    no_matching_recording: "暂无匹配记录",
    not_authorized: "协议未授权",
    analogue_only: "仅类比",
    terminal: "实验已结束",
  };
  elements.generalAcquisitionPlan.innerHTML = plan.sources.map((source) => {
    const recovery = source.recoverable_recording_ids?.length
      ? `<strong>${source.recoverable_recording_ids.length} 条采集后未绑定记录可恢复</strong>`
      : "";
    return `<article data-source="${escapeHtml(source.source)}" data-status="${escapeHtml(source.status)}"><header><span>${escapeHtml(labels[source.source] || source.source)}</span><b>${escapeHtml(statusLabels[source.status] || source.status)}</b></header><p>${escapeHtml(source.boundary_message)}</p>${recovery}<small>CASE EVIDENCE ${source.counts_as_case_evidence ? "允许候选" : "+0"} · Gate C credited ${source.gate_c_credited_records}</small></article>`;
  }).join("");
}

function renderGeneralRecordingOptions() {
  const task = state.generalCase?.current_task;
  const sensors = task?.sensors || [];
  const accountPlan = state.generalAcquisitionPlan?.sources?.find(
    (source) => source.source === "account_recording"
  );
  const candidateIds = new Set(accountPlan?.candidate_recording_ids || []);
  const recoveryIds = new Set(accountPlan?.recoverable_recording_ids || []);
  const current = new Map(
    [...elements.generalRecordingSelectors.querySelectorAll("select[data-sensor]")]
      .map((select) => [select.dataset.sensor, select.value])
  );
  const synchronized = sensors.length > 1;
  elements.generalRecordingSelectors.innerHTML = sensors.map((sensor) => {
    const recordings = state.sensorRecordings.filter((item) => (
      item.sensor === sensor
      && candidateIds.has(item.session_id)
      && (!synchronized || Boolean(item.provenance?.capture_group_id))
    ));
    const options = recordings.map((item) => {
      const group = item.provenance?.capture_group_id
        ? ` · GROUP ${item.provenance.capture_group_id.slice(-8)}`
        : "";
      const recovery = recoveryIds.has(item.session_id) ? "[可恢复] " : "";
      return `<option value="${escapeHtml(item.session_id)}">${escapeHtml(recovery)}${escapeHtml(item.label)} · ${escapeHtml(confidenceText(item.analysis?.confidence))}${escapeHtml(group)} · ${escapeHtml(formatDateTime(item.created_at))}</option>`;
    }).join("");
    return `<label><span>${escapeHtml(SENSOR_LABELS[sensor] || sensor)} 记录</span><select data-sensor="${escapeHtml(sensor)}"><option value="">请选择记录</option>${options}</select></label>`;
  }).join("") || "<p>当前没有可绑定任务。</p>";
  for (const select of elements.generalRecordingSelectors.querySelectorAll("select[data-sensor]")) {
    const previous = current.get(select.dataset.sensor);
    const recovery = state.sensorRecordings.find((item) => (
      recoveryIds.has(item.session_id) && item.sensor === select.dataset.sensor
    ));
    if ([...select.options].some((item) => item.value === previous)) select.value = previous;
    else if (recovery) select.value = recovery.session_id;
  }
  updateGeneralMeasurementButton();
}

function selectedGeneralRecordings() {
  return [...elements.generalRecordingSelectors.querySelectorAll("select[data-sensor]")]
    .map((select) => state.sensorRecordings.find((item) => item.session_id === select.value))
    .filter(Boolean);
}

function generalRecordingSelectionReady() {
  const task = state.generalCase?.current_task;
  const selected = selectedGeneralRecordings();
  if (!task || selected.length !== task.sensors.length) return false;
  if (task.sensors.length === 1) return true;
  const groups = new Set(selected.map((item) => item.provenance?.capture_group_id).filter(Boolean));
  const clocks = new Set(selected.map((item) => item.provenance?.clock_id).filter(Boolean));
  return groups.size === 1 && clocks.size === 1;
}

function updateGeneralMeasurementButton() {
  const collecting = state.generalCase?.status === "collecting" && !state.generalCase?.superseded_by_case_id;
  const task = state.generalCase?.current_task;
  const controlsConfirmed = isGeneralShowcaseCase() || elements.generalControlsConfirm.checked;
  const duration = Number(elements.generalLiveDuration.value);
  const liveAllowed = Boolean(task
    && ((task.sensors?.length === 1 && state.generalCase.protocol.alignment === "sequential")
      || (task.sensors?.length >= 2 && task.sensors?.length <= 3 && state.generalCase.protocol.alignment === "simultaneous"))
    && state.generalCase.protocol.selected_sources?.includes("phyphox_live"));
  const simulationAllowed = Boolean(task
    && state.generalCase.protocol.selected_sources?.length === 1
    && state.generalCase.protocol.selected_sources[0] === "protocol_emulator");
  elements.generalSubmitMeasurement.disabled = !collecting
    || !generalRecordingSelectionReady()
    || !controlsConfirmed
    || state.busy;
  elements.generalLiveCapture.disabled = !collecting
    || !liveAllowed
    || !state.savedDevice
    || !controlsConfirmed
    || !elements.generalLivePrivacy.checked
    || !Number.isFinite(duration)
    || duration < 1
    || duration > 300
    || state.busy;
  elements.generalSimulateMeasurement.disabled = !collecting
    || !simulationAllowed
    || !controlsConfirmed
    || state.busy;
}

function renderGeneralLiveSource() {
  const item = state.generalCase;
  const task = item?.current_task;
  elements.generalMeasurementSources.hidden = !task;
  if (!task) return;
  const sensors = task.sensors || [];
  const sensor = sensors[0];
  const sensorLabels = sensors.map((value) => SENSOR_LABELS[value] || value).join(" + ");
  const simulatedRehearsal = item.protocol.selected_sources?.length === 1
    && item.protocol.selected_sources[0] === "protocol_emulator";
  const showcase = isGeneralShowcaseCase(item);
  elements.generalLiveSourcePanel.hidden = simulatedRehearsal;
  elements.generalSavedSourcePanel.hidden = simulatedRehearsal;
  elements.generalSimulationSourcePanel.hidden = !simulatedRehearsal;
  const simulationHeader = elements.generalSimulationSourcePanel.querySelector("header b");
  const simulationDescription = elements.generalSimulationSourcePanel.querySelector("p");
  const simulationFootnote = elements.generalSimulationSourcePanel.querySelector("small");
  simulationHeader.textContent = showcase
    ? "回放后台冻结证据，立即进入下一实验状态"
    : "运行同一分析器、Planner 与终止链";
  simulationDescription.innerHTML = showcase
    ? "本案例只读取服务器预置的照度序列，仍写入标准 <code>Evidence</code>、决策轨迹和终止报告；不会请求基模。"
    : "只生成带 <code>protocol_emulator</code> lineage 的确定性 analyzer-contract 序列。它能验收软件闭环与多传感器决策，但不是现实、公开或手机证据。";
  simulationFootnote.textContent = showcase
    ? "服务器冻结演示数据 · physical false · Gate C +0 · 0 次模型请求"
    : "Gate C credited 0 · user phone evidence false · 结果会写入当前账号的“模拟演练”历史。";
  elements.generalSimulateMeasurement.textContent = showcase
    ? "回放本步并立即推进"
    : "生成本轮模拟证据并继续";
  const liveAllowed = ((sensors.length === 1 && item.protocol.alignment === "sequential")
    || (sensors.length >= 2 && sensors.length <= 3 && item.protocol.alignment === "simultaneous"))
    && item.protocol.selected_sources?.includes("phyphox_live");
  elements.generalLiveCapture.textContent = `从手机采集${sensorLabels || "当前传感器"}并继续`;
  if (!liveAllowed) {
    elements.generalLiveStatus.textContent = "当前任务不符合已开放的顺序单传感器或同步 2–3 传感器合同；请绑定已有记录。";
  } else if (!state.savedDevice) {
    elements.generalLiveStatus.textContent = "尚未保存默认手机；请先到“设备与设置”连接 phyphox。";
  } else {
    const guidance = sensors.length > 1
      ? `一个同时暴露 ${sensorLabels} profile 的自定义实验`
      : SENSOR_TASK_FALLBACKS[sensor]?.experiment || `输入为 ${sensorLabels} 的实验`;
    elements.generalLiveStatus.textContent = `${state.savedDevice.name} · 请在 phyphox 打开${guidance}并启用远程访问。`;
  }
  updateGeneralMeasurementButton();
}

function generalCompletionBasis(value) {
  return {
    "registered-three-repeats": "注册的三次重复",
    "adaptive-two-repeat-sufficiency": "服务端认证的两次充分证据",
    none: "尚未结束",
  }[value] || value;
}

function generalHypothesisTerminationSummary(audit) {
  if (!audit || audit.disposition === "not-applicable") {
    return { title: "未注册竞争假设", detail: "本实验没有需要额外关闭的竞争假设图。" };
  }
  if (audit.disposition === "pending-discriminator-evidence") {
    return {
      title: "竞争假设终止门尚未通过",
      detail: `仍需 ${audit.unresolved_discriminator_ids.length} 项预注册判别观察；服务端不会静默结束。`,
    };
  }
  if (audit.disposition === "remaining-discriminators-exempted") {
    return {
      title: "竞争假设终止门已通过 · 有审计豁免",
      detail: `共享判别量已区分全部假设，豁免 ${audit.waived_discriminator_ids.length} 项；依据 ${audit.source_evidence_ids.length} 条有效证据。`,
    };
  }
  return {
    title: "竞争假设终止门已通过",
    detail: `${audit.observed_discriminator_ids.length} 项预注册判别观察均已完成，没有豁免。`,
  };
}

function generalHypothesisConclusionSummary(audit) {
  if (!audit) {
    return { state: "legacy", title: "旧记录没有结构化竞争结论", detail: "仍可查看逐假设证据卡，但不能把旧模板当作当前终止政策的结论收据。" };
  }
  if (audit.conclusion_code === "not-applicable") {
    return { state: "not-applicable", title: "无需竞争结论", detail: "本实验没有注册竞争假设。" };
  }
  if (!audit.conclusion_available || audit.conclusion_code === "pending-discriminator-evidence") {
    return { state: "pending", title: "竞争结论尚未形成", detail: `判别观察仍未关闭；当前有 ${audit.untested_hypothesis_ids.length} 个假设尚未完成证据判断。` };
  }
  if (audit.conclusion_code === "one-hypothesis-favored") {
    return { state: "favored", title: `当前证据更符合 ${audit.favored_hypothesis_id}`, detail: `相对削弱 ${audit.weakened_hypothesis_ids.length} 个竞争假设；依据 ${audit.source_evidence_ids.length} 条判别证据，仅作非因果比较。` };
  }
  const compatibleCount = audit.compatible_hypothesis_ids.length;
  return { state: "non-discriminating", title: "本次判别没有产生唯一倾向", detail: `${compatibleCount} 个假设仍与观测相容；这是明确的非判别结论，不会被包装成原因证明。` };
}

function renderHouseholdInstruction(value) {
  const instruction = String(value || "").trim();
  if (!instruction) return "<p>当前没有可执行说明，请使用下方反馈入口让 Agent 重新规划。</p>";
  const matches = [...instruction.matchAll(/(准备|操作|记录|单一变量|本次复现|本次校正|本次观察|基线状态|保持不变|停止条件)：/g)];
  if (matches.length < 2) {
    return `<div class="household-instruction"><article><b>照着做</b><p>${escapeHtml(instruction)}</p></article></div>`;
  }
  const labels = {
    准备: ["1", "先准备好"],
    操作: ["2", "照着做"],
    记录: ["3", "怎样记录"],
    单一变量: ["4", "自变量（本次主动改变）"],
    本次复现: ["4", "本次怎样复现"],
    本次校正: ["4", "本次只校正"],
    本次观察: ["4", "本次只观察"],
    基线状态: ["4", "本次保持的状态"],
    保持不变: ["=", "其余保持不变"],
    停止条件: ["!", "遇到这些情况就停"],
  };
  const sections = matches.map((match, index) => {
    const start = match.index + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index : instruction.length;
    return { key: match[1], text: instruction.slice(start, end).trim().replace(/^[。；\s]+|[。；\s]+$/g, "") };
  }).filter((section) => section.text);
  return `<div class="household-instruction">${sections.map((section) => {
    const [step, label] = labels[section.key];
    return `<article data-kind="${section.key === "停止条件" ? "stop" : "step"}"><span>${step}</span><div><b>${label}</b><p>${escapeHtml(section.text)}</p></div></article>`;
  }).join("")}</div>`;
}

function renderGeneralHypotheses(item) {
  const hypotheses = item.protocol.hypotheses || [];
  const feedbackAllowed = !item.superseded_by_case_id && !item.report;
  elements.generalHypothesisSection.hidden = hypotheses.length === 0;
  if (!hypotheses.length) {
    elements.generalHypothesisList.innerHTML = "";
    return;
  }
  const latestAudit = item.planner_trace.at(-1);
  const reportObservationStates = (item.report?.hypothesis_assessments || []).flatMap(
    (assessment) => assessment.observations.map((observation) => [
      `${assessment.hypothesis_id}:${observation.observation_id}`,
      observation,
    ]),
  );
  const observationStates = new Map(
    [
      ...(latestAudit?.hypothesis_observation_states || []).map((observation) => [
        `${observation.hypothesis_id}:${observation.observation_id}`,
        observation,
      ]),
      ...reportObservationStates,
    ],
  );
  const selectedHypotheses = new Set(latestAudit?.selected_discriminates_hypothesis_ids || []);
  elements.generalHypothesisSection.querySelector("header > b").textContent = `${hypotheses.length} HYPOTHESES`;
  elements.generalHypothesisList.innerHTML = hypotheses.map((hypothesis) => {
    const observations = hypothesis.observations.map((observation) => {
      const stateSnapshot = observationStates.get(`${hypothesis.hypothesis_id}:${observation.observation_id}`);
      const matchCode = stateSnapshot?.match_code || "not_observed";
      const observed = stateSnapshot?.observed_relation
        ? GENERAL_SERVER_FACT_LABELS[stateSnapshot.observed_relation] || stateSnapshot.observed_relation
        : "等待有效条件对比";
      return `<li data-match="${escapeHtml(matchCode)}"><b>${escapeHtml(SENSOR_LABELS[observation.sensor] || observation.sensor)}</b><span>${escapeHtml(GENERAL_EXPECTED_RELATION_LABELS[observation.expected_relation] || observation.expected_relation)}</span><small>${escapeHtml(GENERAL_HYPOTHESIS_MATCH_LABELS[matchCode] || matchCode)} · ${escapeHtml(observed)}</small></li>`;
    }).join("");
    const feedbackButton = feedbackAllowed
      ? `<button type="button" class="hypothesis-feedback-button" data-general-feedback-hypothesis="${escapeHtml(hypothesis.hypothesis_id)}">这条解释不符合我家实际</button>`
      : "";
    return `<article data-selected="${selectedHypotheses.has(hypothesis.hypothesis_id)}"><header><div><b>${escapeHtml(hypothesis.hypothesis_id)}</b><span>待核对的解释</span></div>${selectedHypotheses.has(hypothesis.hypothesis_id) ? "<strong>最近候选用于区分</strong>" : ""}</header><p>${escapeHtml(hypothesis.statement_untrusted)}</p><ul>${observations}</ul>${feedbackButton}</article>`;
  }).join("");
}

function renderGeneralFeedbackSelection() {
  const item = state.generalCase;
  const feedbackType = elements.generalFeedbackType.value;
  const targetsHypothesis = ["hypothesis_not_applicable", "hypothesis_needs_correction"].includes(feedbackType);
  if (!targetsHypothesis) {
    state.generalFeedbackHypothesisIds.clear();
    elements.generalFeedbackSelection.textContent = feedbackType === "task_not_feasible"
      ? "将重新设计当前这一步；请说明家中什么条件使它做不了。"
      : feedbackType === "instruction_unclear"
        ? "请指出哪句话看不懂，Agent 会改成可照做的步骤。"
        : "请补充 Agent 不知道的现场事实，例如设备结构、可接近位置或安全限制。";
    return;
  }
  const selected = [...state.generalFeedbackHypothesisIds];
  if (!selected.length) {
    elements.generalFeedbackSelection.textContent = "请先点击上方某条解释中的“不符合我家实际”。";
    return;
  }
  const labels = selected.map((id) => {
    const hypothesis = item?.protocol?.hypotheses?.find((candidate) => candidate.hypothesis_id === id);
    return hypothesis ? `${id}：${hypothesis.statement_untrusted}` : id;
  });
  elements.generalFeedbackSelection.textContent = feedbackType === "hypothesis_not_applicable"
    ? `将整条排除：${labels.join("；")}`
    : `将按你的说明修正：${labels.join("；")}`;
}

function realityFeedbackReuseSummary(feedback, noun) {
  const reuse = feedback?.evidence_reuse;
  const planning = reuse?.planning_context_evidence_ids?.length || 0;
  const archived = reuse?.archived_only_evidence_ids?.length || 0;
  if (!planning && !archived) return "没有旧测量进入新计划。";
  const parts = [];
  if (planning) parts.push(`${planning} 条旧测量的数值事实只用于重新规划`);
  if (archived) parts.push(`${archived} 条旧测量仅保留在旧版本`);
  return `${parts.join("；")}；它们都不会直接计入新${noun}的证据或结论。`;
}

function handleGeneralFeedbackTarget(event) {
  const button = event.target.closest("[data-general-feedback-hypothesis]");
  if (!button || !state.generalCase) return;
  state.generalFeedbackHypothesisIds.clear();
  state.generalFeedbackHypothesisIds.add(button.dataset.generalFeedbackHypothesis);
  elements.generalFeedbackType.value = "hypothesis_needs_correction";
  elements.generalRealityFeedback.open = true;
  renderGeneralFeedbackSelection();
  elements.generalFeedbackMessage.focus();
}

function handleGeneralTaskFeedback(event) {
  const successor = event.target.closest("[data-open-general-successor]");
  if (successor) {
    navigateTo(`/app/explore/general/runs/${encodeURIComponent(successor.dataset.openGeneralSuccessor)}`);
    return;
  }
  const button = event.target.closest("[data-general-task-feedback]");
  if (!button || !state.generalCase) return;
  state.generalFeedbackHypothesisIds.clear();
  elements.generalFeedbackType.value = button.dataset.generalTaskFeedback;
  elements.generalRealityFeedback.open = true;
  renderGeneralFeedbackSelection();
  elements.generalFeedbackMessage.focus();
}

async function submitGeneralRealityFeedback() {
  const item = state.generalCase;
  if (!item || state.busy || item.superseded_by_case_id) return;
  const feedbackType = elements.generalFeedbackType.value;
  const message = elements.generalFeedbackMessage.value.trim();
  const targetsHypothesis = ["hypothesis_not_applicable", "hypothesis_needs_correction"].includes(feedbackType);
  const hypothesisIds = targetsHypothesis
    ? [...state.generalFeedbackHypothesisIds]
    : [];
  if (message.length < 3) {
    elements.generalFeedbackStatus.dataset.state = "error";
    elements.generalFeedbackStatus.textContent = "请用一句日常语言说明实际情况。";
    return;
  }
  if (targetsHypothesis && !hypothesisIds.length) {
    elements.generalFeedbackStatus.dataset.state = "error";
    elements.generalFeedbackStatus.textContent = "请先选择哪条解释不符合实际。";
    return;
  }
  state.busy = true;
  elements.generalFeedbackSubmit.disabled = true;
  elements.generalFeedbackStatus.dataset.state = "loading";
  elements.generalFeedbackStatus.textContent = "正在保留旧记录，并按你家的实际情况重做实验计划…";
  try {
    const response = await fetch(`/api/v2/general-explorations/${encodeURIComponent(item.case_id)}/reality-feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feedback_type: feedbackType,
        message,
        hypothesis_ids: hypothesisIds,
        expected_task_id: item.current_task?.task_id || null,
        expected_revision: item.revision,
        confirm_sensitive_sensor_reuse: elements.generalFeedbackPrivacy.checked,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const revised = await response.json();
    state.generalCase = revised;
    state.generalAcquisitionPlan = null;
    state.generalFeedbackHypothesisIds.clear();
    elements.generalFeedbackMessage.value = "";
    elements.generalFeedbackPrivacy.checked = false;
    window.history.replaceState({}, "", `/app/explore/general/runs/${encodeURIComponent(revised.case_id)}`);
    await Promise.all([
      refreshGeneralHistory(),
      loadExplorationHistory(),
      loadGeneralAcquisitionPlan(revised.case_id),
    ]);
    await loadGeneralPublicComponentsForCase(revised.case_id);
    applyRoute(false);
    renderGeneralExploration();
    elements.generalFeedbackStatus.dataset.state = "success";
    elements.generalFeedbackStatus.textContent = `新实验已按现场事实生成。${realityFeedbackReuseSummary(revised.revision_feedback, "实验")}`;
    showToast("已按你家的实际情况生成新实验");
  } catch (error) {
    elements.generalFeedbackStatus.dataset.state = "error";
    elements.generalFeedbackStatus.textContent = error.message;
  } finally {
    state.busy = false;
    elements.generalFeedbackSubmit.disabled = false;
    updateGeneralMeasurementButton();
  }
}

function renderGeneralTrajectory(item) {
  const evidenceById = new Map(item.evidence.map((evidence) => [evidence.evidence_id, evidence]));
  const decisionsByRevision = new Map(
    item.decision_trace.map((decision) => [decision.revision, decision]),
  );
  const trajectory = [
    ...item.completed_tasks.map((task) => ({ task, state: "completed" })),
    ...(item.current_task ? [{ task: item.current_task, state: "current" }] : []),
  ];
  elements.generalTrajectoryStatus.textContent = `${item.completed_tasks.length} DONE${item.current_task ? " · 1 NEXT" : " · STOPPED"}`;
  if (!trajectory.length) {
    elements.generalTrajectoryList.innerHTML = "<p>当前案例没有可显示的任务路线。</p>";
    return;
  }
  elements.generalTrajectoryList.innerHTML = trajectory.map(({ task, state: taskState }) => {
    const decision = decisionsByRevision.get(task.sequence);
    const sourceLabel = decision
      ? GENERAL_DECISION_SOURCE_LABELS[decision.source] || decision.source
      : "服务端历史任务";
    const reasonLabel = GENERAL_TASK_REASON_LABELS[task.reason_code] || task.reason_code;
    const actionLabel = GENERAL_TASK_ACTION_LABELS[task.action] || task.action;
    const sensorLabels = task.sensors.map((sensor) => SENSOR_LABELS[sensor] || sensor).join(" + ");
    const taskEvidence = (task.output_evidence_ids || [])
      .map((evidenceId) => evidenceById.get(evidenceId))
      .filter(Boolean);
    const evidenceSummary = taskEvidence.length
      ? taskEvidence.map((evidence) => {
        const source = evidence.lineage.simulated ? "模拟演练" : evidence.lineage.source;
        return `<li data-valid="${escapeHtml(String(evidence.valid))}"><b>${escapeHtml(SENSOR_LABELS[evidence.sensor] || evidence.sensor)}</b><span>${escapeHtml(evidence.metric.label)} ${escapeHtml(formatMetricValue(evidence.metric.value))} ${escapeHtml(evidence.metric.unit)}</span><small>${escapeHtml(source)} · ${escapeHtml(confidenceText(evidence.quality))}${evidence.valid ? " · 已纳入" : " · 已排除"}</small></li>`;
      }).join("")
      : `<li class="general-trajectory-waiting"><span>${taskState === "current" ? "等待你完成本轮测量" : "本任务没有绑定证据"}</span></li>`;
    const stateLabel = taskState === "current"
      ? "当前下一步"
      : task.measurement_valid
        ? "质量门通过"
        : "质量门未通过";
    return `<article data-state="${escapeHtml(taskState)}" data-valid="${escapeHtml(String(task.measurement_valid))}">
      <div class="general-trajectory-index"><span>${escapeHtml(String(task.sequence))}</span><i></i></div>
      <div class="general-trajectory-card">
        <header><div><span>${escapeHtml(actionLabel)} · ${escapeHtml(sensorLabels)}</span><b>${escapeHtml(task.title)}</b></div><strong>${escapeHtml(stateLabel)}</strong></header>
        <p>${escapeHtml(task.instruction)}</p>
        <ul>${evidenceSummary}</ul>
        <footer><span>进入本步：${escapeHtml(reasonLabel)}</span><span>决策来源：${escapeHtml(sourceLabel)}</span></footer>
      </div>
    </article>`;
  }).join("");
}

function renderGeneralReasoningCheckpoint(item) {
  const checkpoint = item?.reasoning_checkpoint;
  elements.generalReasoningCheckpoint.hidden = !checkpoint;
  if (!checkpoint) return;
  const reasoning = checkpoint.reasoning;
  const recommended = (checkpoint.continuation_candidates || []).find(
    (candidate) => candidate.candidate_id === checkpoint.recommended_candidate_id,
  );
  elements.generalCheckpointTitle.textContent = checkpoint.continue_allowed
    ? `已测 ${checkpoint.triggered_at_task_count} 轮：继续探索，还是依据当前证据收手？`
    : `已测 ${checkpoint.triggered_at_task_count} 轮：已达到安全任务上限`;
  elements.generalCheckpointPrompt.textContent = checkpoint.prompt;
  elements.generalCheckpointEvidence.innerHTML = `
    <article><span>Agent 当前判断</span><b>${escapeHtml(reasoning.answer_headline)}</b><p>${escapeHtml(reasoning.mechanism_explanation)}</p></article>
    <article><span>当前证据强度</span><b>${escapeHtml(confidenceText(reasoning.confidence))} · ${Math.round(reasoning.confidence_score * 100)}%</b><p>${escapeHtml(reasoning.next_measurement_reason || "继续测量的边际信息增益有限。")}</p></article>
    ${recommended ? `<article><span>若继续，下一项冻结测量</span><b>${escapeHtml(recommended.title)}</b><p>${escapeHtml(recommended.instruction)}</p></article>` : ""}`;
  elements.generalCheckpointContinue.hidden = !checkpoint.continue_allowed;
  elements.generalCheckpointContinue.disabled = state.busy || !checkpoint.continue_allowed;
  elements.generalCheckpointStop.disabled = state.busy;
}

async function decideGeneralCheckpoint(action) {
  const item = state.generalCase;
  if (!item?.reasoning_checkpoint || state.busy) return;
  state.busy = true;
  renderGeneralReasoningCheckpoint(item);
  elements.generalRunMessage.dataset.state = "loading";
  elements.generalRunMessage.textContent = action === "continue"
    ? "正在恢复 Agent 推荐的下一项判别测量…"
    : "正在依据当前证据生成较低确定性的收手报告…";
  try {
    const response = await fetch(`/api/v2/general-explorations/${encodeURIComponent(item.case_id)}/reasoning-decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_revision: item.revision, action }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.generalCase = await response.json();
    await Promise.all([
      refreshGeneralHistory(),
      loadExplorationHistory(),
      loadGeneralAcquisitionPlan(state.generalCase.case_id),
    ]);
    renderGeneralExploration();
    showToast(action === "continue" ? "已恢复下一项判别测量" : "已按当前证据生成报告");
  } catch (error) {
    elements.generalRunMessage.dataset.state = "error";
    elements.generalRunMessage.textContent = error.message;
  } finally {
    state.busy = false;
    renderGeneralReasoningCheckpoint(state.generalCase);
    updateGeneralMeasurementButton();
  }
}

function renderGeneralExploration() {
  const item = state.generalCase;
  if (!item) return;
  const showcase = isGeneralShowcaseCase(item);
  elements.generalExplorationRun.dataset.showcase = String(showcase);
  elements.generalExplorationRun.hidden = routeState().exploreView !== "general_run";
  elements.generalRealityFeedback.hidden = Boolean(showcase || item.superseded_by_case_id || item.report);
  elements.generalRunTitle.textContent = item.protocol.title;
  elements.generalRunQuestion.textContent = item.protocol.question;
  elements.generalRunRevision.textContent = String(item.revision);
  const compilerLabel = showcase
    ? "服务器编排回放"
    : item.compiler_provenance?.source === "bounded_agent_compiler"
    ? "Agent 编译凭证"
    : "手工审阅协议";
  const simulatedRehearsal = item.protocol.selected_sources?.length === 1
    && item.protocol.selected_sources[0] === "protocol_emulator";
  const executionLabel = showcase
    ? "零等待回放 · 非现实证据"
    : simulatedRehearsal ? "模拟演练 · 非现实证据" : "现实实验";
  const runStateLabel = item.superseded_by_case_id
    ? "已根据现场反馈重规划"
    : item.status === "collecting"
      ? "进行中"
      : item.status === "awaiting_user_decision"
        ? "等待继续/收手选择"
        : "已结束";
  elements.generalRunStatus.textContent = `${runStateLabel} · ${executionLabel} · ${compilerLabel}`;
  renderGeneralAcquisitionPlan();
  const task = item.current_task;
  const taskCondition = task
    ? item.protocol.conditions.find((condition) => condition.condition_id === task.condition_id)
    : null;
  const showcaseTaskTitle = task
    ? `${taskCondition?.label || task.condition_id} · 光线 · 第 ${task.repeat_index} 轮回放`
    : "";
  const showcaseInstruction = task
    ? `后台已冻结“${taskCondition?.label || task.condition_id}”的第 ${task.repeat_index} 轮照度序列。无需连接手机或选择信号；点击下方按钮后，系统会立即计算照度中位数与质量，并把证据写入实验路线。`
    : "";
  elements.generalTaskStep.textContent = task ? `TASK ${task.sequence} · REPEAT ${task.repeat_index}` : item.reasoning_checkpoint ? "AGENT CHECKPOINT" : "PROTOCOL COMPLETE";
  elements.generalTaskTitle.textContent = (showcase && task ? showcaseTaskTitle : task?.title) || (item.reasoning_checkpoint ? "等待你选择继续探索或收手" : "实验已经结束");
  elements.generalTaskInstruction.innerHTML = renderHouseholdInstruction((showcase && task ? showcaseInstruction : task?.instruction) || (item.reasoning_checkpoint ? item.reasoning_checkpoint.prompt : "服务端已停止生成新任务，请查看报告。"));
  const taskContext = task
    ? `<div class="general-task-context"><span>本轮场景：${escapeHtml(taskCondition?.label || task.condition_id)}</span><span>${showcase ? "后台证据" : "打开"}：${task.sensors.map((sensor) => escapeHtml(SENSOR_LABELS[sensor] || sensor)).join(" + ")}</span></div>`
    : "";
  const taskFeedback = item.superseded_by_case_id
    ? `<div class="general-task-feedback"><button type="button" class="task-feedback-button" data-open-general-successor="${escapeHtml(item.superseded_by_case_id)}">打开按现场事实修订后的实验</button></div>`
    : task && !showcase
      ? `<div class="general-task-feedback"><button type="button" class="task-feedback-button" data-general-task-feedback="task_not_feasible">这一步在我家做不了</button><button type="button" class="task-feedback-button" data-general-task-feedback="instruction_unclear">我没看懂怎样操作</button></div>`
      : "";
  elements.generalTaskTags.innerHTML = `${taskContext}${taskFeedback}`;
  elements.generalControlsConfirmText.textContent = showcase
    ? "服务器冻结演示数据会被标记为非现实证据，并直接走标准状态机"
    : simulatedRehearsal
    ? "我理解本轮只生成确定性模拟序列，用于排练软件闭环，不代表任何现实测量"
    : "我已按任务说明只改变自变量，并确认记录来自当前实验现场";
  elements.generalRecordingBind.hidden = !task || Boolean(item.superseded_by_case_id);
  elements.generalControlsConfirm.parentElement.hidden = showcase || !task || Boolean(item.superseded_by_case_id);
  elements.generalSubmitMeasurement.hidden = !task || Boolean(item.superseded_by_case_id);
  if (task && !item.superseded_by_case_id) {
    renderGeneralRecordingOptions();
    renderGeneralLiveSource();
  } else {
    elements.generalMeasurementSources.hidden = true;
  }
  renderGeneralPublicComponents();

  const conditionLabels = new Map(item.protocol.conditions.map((condition) => [condition.condition_id, condition.label]));
  const termination = item.termination;
  const optionalProbeTaskCount = item.completed_tasks.filter(
    (completedTask) => completedTask.action === "probe_optional_sensor"
  ).length;
  elements.generalProgress.innerHTML = [
    ["有效证据", showcase ? `${termination.valid_evidence_count} / 4 回放目标` : `${termination.valid_evidence_count} / ${termination.required_evidence_count}`],
    ["条件覆盖", `${Math.round(termination.condition_coverage_ratio * 100)}%`],
    ["传感器覆盖", `${Math.round(termination.sensor_coverage_ratio * 100)}%`],
    ["重复覆盖", `${Math.round(termination.repeat_coverage_ratio * 100)}%`],
    ["证据预算", `${item.evidence.length} / ${item.protocol.evidence_policy.max_measurements}`],
    ["测量任务", showcase ? `${item.completed_tasks.length} / 4` : `${item.completed_tasks.length} / ${item.protocol.evidence_policy.hard_task_count || 32}`],
    ["纠偏预算", `${termination.correction_count} / ${item.protocol.evidence_policy.max_corrections}`],
    ["可选探测", `${optionalProbeTaskCount} / ${item.protocol.evidence_policy.max_optional_probe_count * (item.protocol.evidence_policy.optional_probe_evidence_mode === "paired_condition_contrast" ? 2 : 1)}`],
    ["终止权", "服务端独占"],
  ].map(([label, value]) => `<div><span>${label}</span><b>${value}</b></div>`).join("");
  const sufficiency = termination.adaptive_sufficiency;
  const hypothesisTermination = termination.hypothesis_termination;
  const hypothesisTerminationSummary = generalHypothesisTerminationSummary(hypothesisTermination);
  const hypothesisConclusionSummary = generalHypothesisConclusionSummary(termination.hypothesis_conclusion);
  elements.generalSufficiency.dataset.eligible = String(sufficiency.eligible);
  elements.generalSufficiency.dataset.hypothesisGate = String(hypothesisTermination?.gate_satisfied ?? true);
  elements.generalSufficiency.innerHTML = `<b>${sufficiency.eligible ? "已通过动态充分度" : "动态充分度未通过"}</b><span>${escapeHtml(generalCompletionBasis(termination.completion_basis))}</span><small>高质量 ${sufficiency.all_evidence_high_quality ? "是" : "否"} · 无纠偏 ${sufficiency.correction_free ? "是" : "否"}</small><div class="general-hypothesis-gate"><b>${escapeHtml(hypothesisTerminationSummary.title)}</b><small>${escapeHtml(hypothesisTerminationSummary.detail)}</small></div><div class="general-hypothesis-conclusion" data-state="${escapeHtml(hypothesisConclusionSummary.state)}"><b>${escapeHtml(hypothesisConclusionSummary.title)}</b><small>${escapeHtml(hypothesisConclusionSummary.detail)}</small></div>`;
  const activationRules = item.protocol.optional_activation_rules || [];
  elements.generalActivationRule.hidden = activationRules.length === 0;
  elements.generalActivationRule.innerHTML = activationRules.map((rule) => {
    const matchingEvidence = item.evidence.slice().reverse().find((evidence) => evidence.valid
      && evidence.sensor === rule.probe_sensor
      && evidence.metric.key === rule.metric_key
      && evidence.metric.unit === rule.metric_unit);
    const triggered = matchingEvidence ? matchingEvidence.metric.value > rule.threshold : null;
    const stateLabel = triggered === null ? "等待辅助证据" : triggered ? "已触发附加对照" : "未触发，返回注册重复";
    const conditionLabel = conditionLabels.get(rule.target_condition_id) || rule.target_condition_id;
    return `<article data-triggered="${escapeHtml(String(triggered))}"><b>服务端证据门槛 · ${escapeHtml(stateLabel)}</b><span>${escapeHtml(SENSOR_LABELS[rule.probe_sensor] || rule.probe_sensor)} / ${escapeHtml(rule.metric_key)} &gt; ${escapeHtml(formatMetricValue(rule.threshold))} ${escapeHtml(rule.metric_unit)}</span><small>目标：${escapeHtml(conditionLabel)} · ${escapeHtml(rule.policy_source)}</small></article>`;
  }).join("");
  renderGeneralTrajectory(item);
  renderGeneralHypotheses(item);
  renderGeneralReasoningCheckpoint(item);
  const showcaseBlockers = termination.valid_evidence_count < 2
    ? ["还需完成近距离与距离加倍的首轮回放。"]
    : termination.valid_evidence_count < 4
      ? ["还需各回放一次，确认条件差异明显超过同条件重复波动。"]
      : [];
  const visibleBlockers = showcase ? showcaseBlockers : termination.blocker_codes;
  elements.generalBlockers.innerHTML = visibleBlockers.length
    ? visibleBlockers.map((code) => `<li>${escapeHtml(code)}</li>`).join("")
    : "<li>没有阻塞项</li>";
  elements.generalEvidenceTrace.innerHTML = item.evidence.length ? item.evidence.slice().reverse().map((evidence) => `
    <article data-valid="${evidence.valid}" data-simulated="${Boolean(evidence.lineage.simulated)}"><header><b>${escapeHtml(evidence.condition_id)} · ${escapeHtml(SENSOR_LABELS[evidence.sensor] || evidence.sensor)}</b><span>${evidence.lineage.simulated ? "模拟 · " : ""}${escapeHtml(confidenceText(evidence.quality))}</span></header><p>${escapeHtml(evidence.metric.label)} ${escapeHtml(formatMetricValue(evidence.metric.value))} ${escapeHtml(evidence.metric.unit)}</p><small>${escapeHtml(evidence.lineage.source)} · physical ${evidence.lineage.physical_evidence ? "true" : "false"} · Gate C ${evidence.lineage.gate_c_passed ? "credited" : "+0"}</small><small>${escapeHtml(evidence.analysis.analyzer_id)} @ ${escapeHtml(evidence.analysis.analyzer_version)}</small></article>`).join("") : "<p>尚无证据。</p>";
  const tasks = new Map(
    [...item.completed_tasks, item.current_task]
      .filter(Boolean)
      .map((candidateTask) => [candidateTask.sequence, candidateTask]),
  );
  const audits = new Map(item.planner_trace.map((audit) => [audit.commit_revision, audit]));
  elements.generalPlannerTrace.innerHTML = item.decision_trace.slice().reverse().map((decision) => {
    const audit = audits.get(decision.revision);
    const selectedTask = tasks.get(decision.revision);
    const sensorLabels = (selectedTask?.sensors || []).map((sensor) => SENSOR_LABELS[sensor] || sensor).join(" + ");
    const sourceLabel = GENERAL_DECISION_SOURCE_LABELS[decision.source] || decision.source;
    const reasonLabel = GENERAL_TASK_REASON_LABELS[decision.reason_code] || decision.reason_code;
    const rationaleLabel = audit
      ? GENERAL_PLANNER_RATIONALE_LABELS[audit.rationale_code] || audit.rationale_code
      : reasonLabel;
    const taskSummary = selectedTask
      ? `${GENERAL_TASK_ACTION_LABELS[selectedTask.action] || selectedTask.action} · ${conditionLabels.get(selectedTask.condition_id) || selectedTask.condition_id} · ${sensorLabels}`
      : "历史任务摘要不可用";
    const runtimeSummary = audit?.runtime
      ? `${audit.outcome === "accepted" ? "模型决策已通过服务端校验" : "模型结果未采用，已安全回退"} · ${audit.runtime.transport}`
      : "该步由服务端确定性合同完成";
    const informationGoal = audit?.selected_information_goal
      ? GENERAL_INFORMATION_GOAL_LABELS[audit.selected_information_goal] || audit.selected_information_goal
      : "旧历史未保存信息目标";
    const effortSummary = audit?.selected_effort_points
      ? `相对现场成本 ${audit.selected_effort_points} 点`
      : "旧历史未保存成本快照";
    const factCodes = [
      ...(audit?.selected_candidate_fact_codes || []),
      ...(audit?.evidence_fact_codes || []),
      ...(audit?.contrast_fact_codes || []),
    ];
    const factSummary = factCodes.length
      ? factCodes.map((code) => GENERAL_SERVER_FACT_LABELS[code] || code).join(" · ")
      : "该历史步骤没有持久化服务端 facts";
    const hypothesisSummary = audit?.selected_discriminates_hypothesis_ids?.length
      ? `区分假设：${audit.selected_discriminates_hypothesis_ids.join(" · ")}`
      : "该候选没有注册的假设区分目标";
    const matchSummary = audit?.hypothesis_match_codes?.length
      ? audit.hypothesis_match_codes.map((code) => GENERAL_HYPOTHESIS_MATCH_LABELS[code] || code).join(" · ")
      : "旧历史未保存假设匹配状态";
    return `<article data-source="${escapeHtml(decision.source)}"><header><b>${escapeHtml(sourceLabel)}</b><span>第 ${decision.revision} 步</span></header><p>${escapeHtml(selectedTask?.title || taskSummary)}</p><small>${escapeHtml(taskSummary)}</small><strong>${escapeHtml(rationaleLabel)}</strong><small>${escapeHtml(runtimeSummary)}</small><small>${escapeHtml(informationGoal)} · ${escapeHtml(effortSummary)}</small><details><summary>查看冻结候选审计</summary><small>${escapeHtml(factSummary)}</small><small>${escapeHtml(hypothesisSummary)} · ${escapeHtml(matchSummary)}</small><code>selected=${escapeHtml(decision.selected_candidate_id)} · candidates=${decision.candidate_ids.map((value) => escapeHtml(value)).join(", ")}</code></details></article>`;
  }).join("");
  renderGeneralReport(item.report, item.protocol);
  elements.generalRunMessage.dataset.state = item.superseded_by_case_id ? "warning" : item.status === "collecting" ? "ready" : item.status === "awaiting_user_decision" ? "warning" : "complete";
  elements.generalRunMessage.textContent = item.superseded_by_case_id
    ? "这是反馈前的旧版本。原任务与测量记录均已保留，但不能继续提交；请打开修订后的实验。"
    : item.status === "collecting"
      ? showcase
        ? "点击“回放本步并立即推进”；后台冻结照度证据会立即进入标准分析、决策与终止页面。"
        : simulatedRehearsal
        ? "确认模拟边界后运行本轮；Agent 会依据已提交的模拟分析结果决定下一任务。"
        : "选择与当前任务传感器一致的账号记录；记录不会自动冒充真机 Gate C。"
      : item.status === "awaiting_user_decision"
        ? "当前解释仍有歧义：请选择继续 Agent 推荐测量，或依据现有证据收手。"
        : showcase
          ? "零等待回放已结束；条件对比图、重复性审计与有边界报告均由标准探索状态机生成。"
          : simulatedRehearsal
          ? "模拟排练已结束；报告仅证明软件闭环可执行。"
          : "实验已经结束，不会再生成新任务。";
  elements.generalControlsConfirm.checked = false;
  elements.generalLivePrivacy.checked = false;
  updateGeneralMeasurementButton();
}

function generalDirectionLabel(value) {
  return {
    increase: "明显升高",
    decrease: "明显降低",
    within_observed_repeatability: "未超出重复波动",
  }[value] || value;
}

function generalAuxiliaryInterpretation(value) {
  return {
    single_optional_probe_not_a_condition_comparison: "单次可选传感器探查，不构成条件比较",
    single_optional_condition_probe_not_registered_comparison: "单次附加对照探查，不属于预注册比较",
    paired_optional_probe_descriptive_contrast: "竞争假设的成对辅助对照，仅作描述性判断",
  }[value] || value;
}

function compactGeneralChartLabel(value, maximum = 16) {
  const text = String(value);
  return text.length > maximum ? `${text.slice(0, maximum - 1)}…` : text;
}

function renderGeneralSeriesChart(series, independentVariable) {
  const points = (series.points || []).filter((point) => Number.isFinite(point.median) && Number.isFinite(point.median_absolute_deviation));
  if (points.length < 2) {
    return `<article class="general-chart-card"><header><b>${escapeHtml(SENSOR_LABELS[series.sensor] || series.sensor)}</b><span>证据不足</span></header><p>至少需要两个条件的重复摘要才能绘图。</p></article>`;
  }
  const width = 560;
  const height = 286;
  const left = 72;
  const right = 28;
  const top = 38;
  const bottom = 72;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const lowerValues = points.map((point) => point.median - point.median_absolute_deviation);
  const upperValues = points.map((point) => point.median + point.median_absolute_deviation);
  let minimum = Math.min(...lowerValues);
  let maximum = Math.max(...upperValues);
  const observedSpan = maximum - minimum;
  const padding = observedSpan > 0
    ? observedSpan * 0.2
    : Math.max(Math.abs(maximum) * 0.08, 1e-6);
  minimum -= padding;
  maximum += padding;
  const y = (value) => top + ((maximum - value) / (maximum - minimum)) * plotHeight;
  const x = (index) => left + (index / (points.length - 1)) * plotWidth;
  const ticks = Array.from({ length: 5 }, (_, index) => maximum - (index / 4) * (maximum - minimum));
  const polyline = points.map((point, index) => `${x(index).toFixed(2)},${y(point.median).toFixed(2)}`).join(" ");
  const plotBottom = height - bottom;
  const area = `${left},${plotBottom} ${polyline} ${x(points.length - 1).toFixed(2)},${plotBottom}`;
  const grid = ticks.map((tick) => {
    const yValue = y(tick).toFixed(2);
    return `<line class="general-chart-grid-line" x1="${left}" y1="${yValue}" x2="${width - right}" y2="${yValue}"></line><text class="general-chart-y-label" x="${left - 8}" y="${Number(yValue) + 3}" text-anchor="end">${escapeHtml(formatMetricValue(tick))}</text>`;
  }).join("");
  const conditionGuides = points.map((point, index) => {
    const xValue = x(index).toFixed(2);
    return `<line class="general-chart-condition-guide" x1="${xValue}" y1="${top}" x2="${xValue}" y2="${plotBottom}"></line>`;
  }).join("");
  const marks = points.map((point, index) => {
    const xValue = x(index).toFixed(2);
    const yValue = y(point.median).toFixed(2);
    const upper = y(point.median + point.median_absolute_deviation).toFixed(2);
    const lower = y(point.median - point.median_absolute_deviation).toFixed(2);
    const hitboxWidth = Math.min(120, Math.max(72, plotWidth / points.length));
    const hitboxX = Math.max(left, Math.min(width - right - hitboxWidth, Number(xValue) - (hitboxWidth / 2)));
    const tooltip = `完整条件：${point.condition_label}；中位数 ${formatMetricValue(point.median)} ${series.unit}；MAD ${formatMetricValue(point.median_absolute_deviation)}`;
    return `<g><line class="general-chart-error" x1="${xValue}" y1="${upper}" x2="${xValue}" y2="${lower}"></line><line class="general-chart-error-cap" x1="${Number(xValue) - 7}" y1="${upper}" x2="${Number(xValue) + 7}" y2="${upper}"></line><line class="general-chart-error-cap" x1="${Number(xValue) - 7}" y1="${lower}" x2="${Number(xValue) + 7}" y2="${lower}"></line><circle class="general-chart-point-halo" cx="${xValue}" cy="${yValue}" r="11"></circle><circle class="general-chart-point" cx="${xValue}" cy="${yValue}" r="5"></circle><text class="general-chart-value" x="${xValue}" y="${Math.max(top + 12, Number(upper) - 11)}" text-anchor="middle">${escapeHtml(formatMetricValue(point.median))}</text><g class="general-chart-x-target" tabindex="0" role="img" aria-label="${escapeHtml(tooltip)}"><title>${escapeHtml(tooltip)}</title><rect class="general-chart-label-hitbox" x="${hitboxX.toFixed(2)}" y="${height - 56}" width="${hitboxWidth.toFixed(2)}" height="42" rx="7"></rect><text class="general-chart-x-label" x="${xValue}" y="${height - 39}" text-anchor="middle">${escapeHtml(compactGeneralChartLabel(point.condition_label))}</text><text class="general-chart-repeat-label" x="${xValue}" y="${height - 22}" text-anchor="middle">${point.repeat_count} 次重复</text></g></g>`;
  }).join("");
  const sensorLabel = SENSOR_LABELS[series.sensor] || series.sensor;
  const delta = points.at(-1).median - points[0].median;
  const deltaPrefix = delta > 0 ? "+" : "";
  return `<article class="general-chart-card"><header><div><span class="general-chart-kicker">CONDITION CONTRAST</span><b>${escapeHtml(sensorLabel)}</b><span>${escapeHtml(series.metric_label)}</span></div><strong>${escapeHtml(series.unit)}</strong></header><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(`${sensorLabel}按${independentVariable}的条件对比图`)}"><rect class="general-chart-plot" x="${left}" y="${top}" width="${plotWidth}" height="${plotHeight}" rx="12"></rect>${grid}${conditionGuides}<polygon class="general-chart-area" points="${area}"></polygon><line class="general-chart-axis" x1="${left}" y1="${top}" x2="${left}" y2="${plotBottom}"></line><line class="general-chart-axis" x1="${left}" y1="${plotBottom}" x2="${width - right}" y2="${plotBottom}"></line><polyline class="general-chart-line" points="${polyline}"></polyline>${marks}</svg><footer><div><span>横轴 · ${escapeHtml(independentVariable)}</span><span>纵轴 · ${escapeHtml(series.metric_label)}</span><span class="general-chart-hover-hint">悬停横轴标签查看完整条件</span></div><strong class="general-chart-delta">Δ ${escapeHtml(`${deltaPrefix}${formatMetricValue(delta)} ${series.unit}`)}</strong></footer></article>`;
}

function renderGeneralReport(report, protocol) {
  elements.generalFinalReport.hidden = !report;
  if (!report) return;
  const conditionLabels = new Map((protocol?.conditions || []).map((condition) => [condition.condition_id, condition.label]));
  elements.generalReportConfidence.textContent = confidenceText(report.confidence);
  elements.generalReportBasis.textContent = report.evidence_scope === "simulated_rehearsal"
    ? `${generalCompletionBasis(report.completion_basis)} · 模拟演练`
    : `${generalCompletionBasis(report.completion_basis)} · 现实记录`;
  elements.generalReportAnswer.textContent = report.answer_headline || report.answer;
  elements.generalReportNarrative.hidden = !report.answer_headline;
  elements.generalReportNarrative.textContent = report.answer_headline ? report.answer : "";
  const reasoning = report.reasoning;
  elements.generalReasoningAnalysis.hidden = !reasoning;
  if (reasoning) {
    const roleLabels = {
      target_mechanism: "自变量的直接解释",
      alternative_mechanism: "其他可能解释",
      confound: "可能的干扰因素",
      measurement_artifact: "测量方式造成的假象",
    };
    const verdictLabels = {
      favored: "最受支持",
      plausible: "仍可能",
      weakened: "已削弱",
      unsupported: "缺少支持",
      untested: "尚未检验",
    };
    const scopeLabel = reasoning.claim_scope === "local_intervention_supported"
      ? "本次受控干预支持"
      : reasoning.claim_scope === "ranked_explanation"
        ? "解释排序"
        : "描述性判断";
    elements.generalReasoningScore.textContent = `${scopeLabel} · ${confidenceText(reasoning.confidence)} · ${Math.round(reasoning.confidence_score * 100)}%`;
    elements.generalReasoningMechanism.textContent = reasoning.mechanism_explanation;
    elements.generalReasoningExplanations.innerHTML = reasoning.explanations.map((explanation) => `
      <article data-role="${escapeHtml(explanation.role)}" data-verdict="${escapeHtml(explanation.verdict)}">
        <header><span>${escapeHtml(roleLabels[explanation.role] || explanation.role)}</span><b>${escapeHtml(verdictLabels[explanation.verdict] || explanation.verdict)}</b></header>
        <h6>${escapeHtml(explanation.label)}</h6><p>${escapeHtml(explanation.reasoning)}</p>
        <small>证据事实：${escapeHtml((explanation.supporting_fact_ids || []).join(" · ") || "无直接支持事实")}</small>
      </article>`).join("");
  }
  const reportHypothesisTermination = generalHypothesisTerminationSummary(report.hypothesis_termination);
  elements.generalReportReason.textContent = `${report.termination_reason} ${reportHypothesisTermination.title}：${reportHypothesisTermination.detail}`;
  const visualizations = report.visualizations || [];
  elements.generalVisualizationGrid.innerHTML = visualizations.length
    ? visualizations.flatMap((artifact) => artifact.series.map((series) => renderGeneralSeriesChart(series, artifact.independent_variable))).join("")
    : "<p class=\"general-visualization-empty\">这条旧记录没有服务端可视化产物；数值摘要仍可继续验收。</p>";
  elements.generalSummaryGrid.innerHTML = report.summaries.map((summary) => `
    <article><span>${escapeHtml(conditionLabels.get(summary.condition_id) || summary.condition_id)} · ${escapeHtml(SENSOR_LABELS[summary.sensor] || summary.sensor)}</span><b>${escapeHtml(formatMetricValue(summary.median))} ${escapeHtml(summary.unit)}</b><small>${summary.values.length} 次重复 · MAD ${escapeHtml(formatMetricValue(summary.median_absolute_deviation))}</small></article>`).join("");
  elements.generalContrastList.innerHTML = report.contrasts.map((contrast) => `
    <article><b>${escapeHtml(SENSOR_LABELS[contrast.sensor] || contrast.sensor)} · ${escapeHtml(generalDirectionLabel(contrast.direction))}</b><span>${escapeHtml(conditionLabels.get(contrast.reference_condition_id) || contrast.reference_condition_id)} → ${escapeHtml(conditionLabels.get(contrast.comparison_condition_id) || contrast.comparison_condition_id)}</span><small>Δ ${escapeHtml(formatMetricValue(contrast.absolute_delta))} ${escapeHtml(contrast.unit)} · 阈值 ${escapeHtml(formatMetricValue(contrast.descriptive_threshold))}</small></article>`).join("");
  const hypothesisAssessments = report.hypothesis_assessments || [];
  elements.generalReportHypothesisSection.hidden = hypothesisAssessments.length === 0;
  const hypothesisConclusionSummary = generalHypothesisConclusionSummary(report.hypothesis_conclusion);
  elements.generalReportHypothesisConclusion.dataset.state = hypothesisConclusionSummary.state;
  elements.generalReportHypothesisConclusion.innerHTML = `<b>${escapeHtml(hypothesisConclusionSummary.title)}</b><span>${escapeHtml(hypothesisConclusionSummary.detail)}</span>`;
  elements.generalReportHypotheses.innerHTML = hypothesisAssessments.map((assessment) => {
    const observations = assessment.observations.map((observation) => {
      const observed = observation.observed_relation
        ? GENERAL_SERVER_FACT_LABELS[observation.observed_relation] || observation.observed_relation
        : "等待参考与比较条件的有效证据";
      const evidenceSummary = observation.source_evidence_ids.length
        ? `证据 ${observation.source_evidence_ids.length} 条 · ${observation.source_evidence_ids.join(" · ")}`
        : "尚未绑定成对证据";
      return `<li data-match="${escapeHtml(observation.match_code)}"><header><b>${escapeHtml(SENSOR_LABELS[observation.sensor] || observation.sensor)} · ${escapeHtml(observation.metric_key)}</b><strong>${escapeHtml(GENERAL_HYPOTHESIS_MATCH_LABELS[observation.match_code] || observation.match_code)}</strong></header><span>${escapeHtml(GENERAL_EXPECTED_RELATION_LABELS[observation.expected_relation] || observation.expected_relation)} · 实际：${escapeHtml(observed)}</span><small>${escapeHtml(evidenceSummary)}</small></li>`;
    }).join("");
    return `<article data-assessment="${escapeHtml(assessment.assessment_code)}"><header><div><b>${escapeHtml(assessment.hypothesis_id)}</b><span>未验证假设 · 非因果</span></div><strong>${escapeHtml(GENERAL_HYPOTHESIS_ASSESSMENT_LABELS[assessment.assessment_code] || assessment.assessment_code)}</strong></header><p>${escapeHtml(assessment.statement_untrusted)}</p><ul>${observations}</ul></article>`;
  }).join("");
  const auxiliary = report.auxiliary_observations || [];
  elements.generalAuxiliarySection.hidden = auxiliary.length === 0;
  elements.generalAuxiliaryList.innerHTML = auxiliary.map((observation) => `
    <article><header><b>${escapeHtml(SENSOR_LABELS[observation.sensor] || observation.sensor)}</b><span>${escapeHtml(confidenceText(observation.quality))}</span></header><p>${escapeHtml(conditionLabels.get(observation.condition_id) || observation.condition_id)} · ${escapeHtml(observation.metric_key)}</p><strong>${escapeHtml(formatMetricValue(observation.value))} ${escapeHtml(observation.unit)}</strong><small>${escapeHtml(generalAuxiliaryInterpretation(observation.interpretation))}</small></article>`).join("");
  elements.generalReportBoundaries.innerHTML = report.claim_boundaries.map((value) => `<li>${escapeHtml(value)}</li>`).join("");
}

async function submitGeneralMeasurement() {
  const item = state.generalCase;
  const task = item?.current_task;
  if (!item || !task || state.busy) return;
  state.busy = true;
  updateGeneralMeasurementButton();
  elements.generalRunMessage.dataset.state = "loading";
  elements.generalRunMessage.textContent = "正在绑定证据、运行确定性门禁并选择下一步…";
  try {
    const response = await fetch(`/api/v2/general-explorations/${encodeURIComponent(item.case_id)}/measurements`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: item.revision,
        task_id: task.task_id,
        recording_ids: selectedGeneralRecordings().map((recording) => recording.session_id),
        controls_confirmed: elements.generalControlsConfirm.checked,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.generalCase = await response.json();
    await Promise.all([
      refreshGeneralHistory(),
      loadExplorationHistory(),
      loadGeneralAcquisitionPlan(state.generalCase.case_id),
    ]);
    renderGeneralExploration();
    showToast(state.generalCase.status === "collecting" ? "证据已绑定，下一任务已生成" : state.generalCase.status === "awaiting_user_decision" ? "证据仍有歧义，请选择继续或收手" : "自由探索已形成机制化报告");
  } catch (error) {
    elements.generalRunMessage.dataset.state = "error";
    elements.generalRunMessage.textContent = error.message;
  } finally {
    state.busy = false;
    updateGeneralMeasurementButton();
  }
}

async function captureGeneralMeasurement() {
  const item = state.generalCase;
  const task = item?.current_task;
  const duration = Number(elements.generalLiveDuration.value);
  if (!item || !task || state.busy || elements.generalLiveCapture.disabled) return;
  state.busy = true;
  updateGeneralMeasurementButton();
  elements.generalRunMessage.dataset.state = "loading";
  elements.generalRunMessage.textContent = "正在从手机采集一次、运行确定性分析并绑定当前任务…";
  try {
    const synchronized = task.sensors.length > 1;
    const endpoint = synchronized ? "phyphox/synchronized" : "phyphox";
    const response = await fetch(`/api/v2/general-explorations/${encodeURIComponent(item.case_id)}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: item.revision,
        task_id: task.task_id,
        duration_s: duration,
        controls_confirmed: elements.generalControlsConfirm.checked,
        privacy_acknowledged: elements.generalLivePrivacy.checked,
      }),
    });
    if (!response.ok) {
      const message = await readApiError(response);
      await refreshGeneralRecordings({ quiet: true });
      const recoveryCount = state.generalAcquisitionPlan?.sources?.find(
        (source) => source.source === "account_recording"
      )?.recoverable_recording_ids?.length || 0;
      throw new Error(recoveryCount
        ? `${message} 已找到 ${recoveryCount} 条本任务未绑定记录，并在“绑定已有记录”中自动选中。`
        : message);
    }
    const result = await response.json();
    state.generalCase = result.case;
    await Promise.all([
      refreshGeneralHistory(),
      loadExplorationHistory(),
      loadSessionHistory(),
      loadGeneralAcquisitionPlan(state.generalCase.case_id),
    ]);
    renderGeneralExploration();
    elements.generalRunMessage.dataset.state = state.generalCase.status === "collecting" ? "ready" : state.generalCase.status === "awaiting_user_decision" ? "warning" : "complete";
    const captures = result.captures || [result.capture];
    elements.generalRunMessage.textContent = `${captures[0].experiment_title} · 已保存并绑定 ${captures.map((capture) => `${SENSOR_LABELS[capture.sensor] || capture.sensor} ${capture.sample_count} 点`).join("、")}${result.alignment ? `；同步误差 ≤ ${formatMetricValue(result.alignment.maximum_alignment_error_ms)} ms` : ""}。`;
    showToast(state.generalCase.status === "collecting" ? "手机证据已绑定，下一任务已生成" : state.generalCase.status === "awaiting_user_decision" ? "证据仍有歧义，请选择继续或收手" : "自由探索已形成机制化报告");
  } catch (error) {
    elements.generalRunMessage.dataset.state = "error";
    elements.generalRunMessage.textContent = error.message;
  } finally {
    state.busy = false;
    updateGeneralMeasurementButton();
  }
}

async function simulateGeneralMeasurement() {
  const item = state.generalCase;
  const task = item?.current_task;
  if (!item || !task || state.busy || elements.generalSimulateMeasurement.disabled) return;
  const showcase = isGeneralShowcaseCase(item);
  state.busy = true;
  updateGeneralMeasurementButton();
  elements.generalRunMessage.dataset.state = "loading";
  elements.generalRunMessage.textContent = showcase
    ? "正在提交后台冻结证据，并立即推进标准探索状态机…"
    : "正在生成带模拟 lineage 的分析器合同序列，并运行同一多轮决策与终止链…";
  try {
    const endpoint = showcase
      ? `/api/v2/showcase-replays/exploration/${encodeURIComponent(item.case_id)}/tasks/${encodeURIComponent(task.task_id)}`
      : `/api/v2/general-explorations/${encodeURIComponent(item.case_id)}/simulate`;
    const body = showcase
      ? { expected_revision: item.revision }
      : {
        expected_revision: item.revision,
        task_id: task.task_id,
        profile: elements.generalSimulationProfile.value,
        controls_confirmed: elements.generalControlsConfirm.checked,
      };
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const result = await response.json();
    state.generalCase = result.case;
    await Promise.all([
      refreshGeneralHistory(),
      loadExplorationHistory(),
      loadGeneralAcquisitionPlan(state.generalCase.case_id),
    ]);
    renderGeneralExploration();
    const sensors = result.simulation.sensors.map((sensor) => SENSOR_LABELS[sensor] || sensor).join(" + ");
    elements.generalRunMessage.dataset.state = state.generalCase.status === "collecting" ? "ready" : state.generalCase.status === "awaiting_user_decision" ? "warning" : "complete";
    elements.generalRunMessage.textContent = `${sensors} · 已提交 ${result.evidence.length} 条${showcase ? "冻结回放" : "模拟分析"}证据；physical=false，Gate C +0。${state.generalCase.status === "collecting" ? "下一任务已由服务端生成。" : state.generalCase.status === "awaiting_user_decision" ? "证据仍有歧义，请选择继续或收手。" : "软件排练已形成有边界报告。"}`;
    showToast(state.generalCase.status === "collecting" ? (showcase ? "本步已回放，下一实验状态已就绪" : "模拟证据已提交，下一任务已生成") : state.generalCase.status === "awaiting_user_decision" ? "证据仍有歧义，请选择继续或收手" : showcase ? "光学零等待探索回放完成" : "模拟演练闭环完成");
  } catch (error) {
    elements.generalRunMessage.dataset.state = "error";
    elements.generalRunMessage.textContent = error.message;
  } finally {
    state.busy = false;
    updateGeneralMeasurementButton();
  }
}

async function refreshInvestigationHistory() {
  const response = await fetch("/api/v2/investigations");
  if (!response.ok) throw new Error(await readApiError(response));
  state.investigationHistory = await response.json();
  renderActiveInvestigations();
}

async function openInvestigation(caseId) {
  const response = await fetch(`/api/v2/investigations/${encodeURIComponent(caseId)}`);
  if (!response.ok) throw new Error(await readApiError(response));
  state.investigation = await response.json();
  clearInvestigationError();
  renderInvestigation();
}

async function refreshCurrentInvestigation() {
  if (!state.investigation?.case_id || state.busy) return;
  elements.investigationRefreshButton.disabled = true;
  try {
    const caseId = state.investigation.case_id;
    const response = await fetch(`/api/v2/investigations/${encodeURIComponent(caseId)}`);
    if (!response.ok) throw new Error(await readApiError(response));
    state.investigation = await response.json();
    clearInvestigationError();
    renderInvestigation();
    showToast(`实验已刷新到 revision ${state.investigation.revision}`);
  } catch (error) {
    setInvestigationError(`刷新失败：${error.message}`);
  } finally {
    elements.investigationRefreshButton.disabled = false;
  }
}

function setInvestigationError(message) {
  state.investigationError = String(message || "未知错误");
  renderInvestigationError();
}

function clearInvestigationError() {
  state.investigationError = "";
  renderInvestigationError();
}

function renderInvestigationError() {
  if (!elements.investigationError) return;
  elements.investigationError.hidden = !state.investigationError;
  elements.investigationErrorMessage.textContent = state.investigationError;
}

function investigationParameters() {
  const task = state.investigation?.current_task;
  if (!task?.parameter_definitions?.length) return [];
  return task.parameter_definitions.map((definition) => ({
    key: definition.key,
    value: Number(elements.investigationDistance.value),
    unit: definition.unit,
  }));
}

function analysisMetric(analysis, key) {
  return analysis?.metrics?.find((metric) => metric.key === key) || null;
}

function renderInvestigationEvidence(item) {
  const recentEvidence = item.evidence.slice(-8).reverse();
  elements.investigationEvidenceTrace.innerHTML = recentEvidence.length ? recentEvidence.map((evidence) => {
    const observed = analysisMetric(evidence.analysis, "median_illuminance_lx");
    const parameters = evidence.parameters.map((entry) => `${entry.key}=${formatMetricValue(Number(entry.value))} ${entry.unit}`).join(" · ");
    const notes = evidence.observation_notes ? `<p>观察：${escapeHtml(evidence.observation_notes)}</p>` : "";
    const warnings = evidence.analysis?.warnings?.length ? `<p>门禁提示：${escapeHtml(evidence.analysis.warnings.join("；"))}</p>` : "";
    return `
      <div data-valid="${evidence.valid}">
        <b>${escapeHtml(evidence.condition_id)} · ${escapeHtml(INVESTIGATION_ROLE_LABELS[evidence.role] || evidence.role)}</b>
        <span>${evidence.valid ? `${escapeHtml(confidenceText(evidence.quality))}证据` : escapeHtml(evidence.rejection_reasons.join("；"))}</span>
        ${observed ? `<small>${escapeHtml(observed.label)} ${escapeHtml(formatMetricValue(observed.value))} ${escapeHtml(observed.unit)}${parameters ? ` · ${escapeHtml(parameters)}` : ""}</small>` : ""}
        <small>${escapeHtml(evidence.recording.source)} · ${escapeHtml(evidence.recording.analyzer_id)} @ ${escapeHtml(evidence.recording.analyzer_version)}</small>
        ${notes}${warnings}
      </div>`;
  }).join("") : "<p>尚无证据；第一步是环境光背景测量。</p>";
}

function renderInvestigationTools(item) {
  const executions = item.tool_trace.slice(-10).reverse();
  elements.investigationToolTrace.innerHTML = executions.length ? executions.map((execution) => {
    const metrics = execution.result_metrics?.length
      ? execution.result_metrics.map((metric) => `${metric.label} ${formatMetricValue(metric.value)} ${metric.unit}`).join(" · ")
      : "没有数值输出";
    const failed = execution.status === "failed" || execution.status === "rejected";
    return `
      <div data-valid="${!failed}">
        <b>${escapeHtml(INVESTIGATION_TOOL_LABELS[execution.tool_id] || execution.tool_id)}</b>
        <span>${escapeHtml(execution.status)} · v${escapeHtml(execution.tool_version)}</span>
        <small>${escapeHtml(metrics)}</small>
        <small>依据 ${execution.input_evidence_ids.length} 条 evidence</small>
        ${execution.error_message ? `<p>${escapeHtml(execution.error_message)}</p>` : ""}
      </div>`;
  }).join("") : "<p>尚无工具调用；提交第一条记录后开始形成审计轨迹。</p>";
}

function renderInvestigationPlanner(item) {
  const decisions = item.planner_trace.slice(-8).reverse();
  elements.investigationPlannerTrace.innerHTML = decisions.length ? decisions.map((decision) => {
    const source = decision.source === "agent" ? "受限 Agent 已接受" : "确定性安全回退";
    const rationale = PLANNER_RATIONALE_LABELS[decision.rationale_code] || decision.rationale_code;
    const transport = PLANNER_TRANSPORT_LABELS[decision.transport] || decision.transport || "未记录";
    const compatibility = decision.transport_fallback_reason
      ? ` · 兼容切换：${decision.transport_fallback_reason}`
      : "";
    const runtime = decision.source === "agent"
      ? `${decision.model || "模型未知"} · ${formatMetricValue(decision.elapsed_s)} s · ${decision.total_tokens ?? "—"} tokens`
      : `回退原因：${decision.fallback_reason || "planner-unavailable"}`;
    return `
      <div data-valid="${decision.outcome === "accepted"}">
        <b>${escapeHtml(source)} · ${escapeHtml(rationale)}</b>
        <span>选择 ${escapeHtml(decision.selected_candidate_id)} / 候选 ${decision.candidate_ids.length} 个</span>
        <small>${escapeHtml(runtime)}</small>
        <small>传输 ${escapeHtml(transport)}${escapeHtml(compatibility)}</small>
        <small>allowlist ${decision.allowlist_respected ? "通过" : "未通过"} · revision ${decision.revision_before} → ${decision.revision_after}</small>
      </div>`;
  }).join("") : `<p>${item.planning_policy === "bounded_agent" ? "尚未到达需要 Agent 选择新距离的阶段。" : "本实验使用确定性规划策略。"}</p>`;
}

function renderInvestigation() {
  const item = state.investigation;
  elements.investigationWorkbench.hidden = !item || routeState().exploreView !== "run";
  renderInvestigationError();
  if (!item) return;
  elements.investigationTitle.textContent = item.title;
  elements.investigationQuestion.textContent = item.research_question;
  elements.investigationRevision.textContent = String(item.revision);
  const task = item.current_task;
  const terminal = !task;
  elements.investigationTaskStep.textContent = task ? `TASK ${task.sequence} · ${INVESTIGATION_ROLE_LABELS[task.role] || task.role}` : "PROTOCOL COMPLETE";
  elements.investigationTaskTitle.textContent = task?.title || "实验已终止";
  elements.investigationTaskInstruction.textContent = task?.instruction || "请查看右侧终止向量和下方实验反馈。";
  elements.investigationDecision.hidden = terminal;
  elements.investigationDecisionSource.textContent = task ? (INVESTIGATION_SOURCE_LABELS[task.selection_source] || task.selection_source) : "—";
  elements.investigationDecisionReason.textContent = task?.selection_reason || "实验已按终止规则停止。";
  const decisionEvidence = task?.selection_evidence_ids?.length
    ? ` · evidence_ids: ${task.selection_evidence_ids.join(", ")}`
    : " · evidence_ids: 无（协议起始任务）";
  elements.investigationDecisionBasis.textContent = task ? `${task.selection_reason_code}${decisionEvidence}` : "";
  elements.investigationControls.innerHTML = task?.controls?.map((control) => `<div><i>✓</i><span>${escapeHtml(control)}</span></div>`).join("") || "";
  const distanceTarget = task?.parameter_targets?.find((value) => value.key === "distance_m");
  elements.investigationDistanceField.hidden = !distanceTarget;
  if (distanceTarget) elements.investigationDistance.value = distanceTarget.value;
  const usedRecordings = new Set(item.evidence.map((evidence) => evidence.recording.recording_id));
  const previousRecording = elements.investigationRecording.value;
  const lightRecordings = state.sensorRecordings.filter((recording) => (
    canBindRecordingToLightDistance(recording) && !usedRecordings.has(recording.session_id)
  ));
  const recordingPlaceholder = lightRecordings.length ? "选择未绑定的 v2 光照记录" : "暂无未绑定的 v2 光照记录";
  elements.investigationRecording.innerHTML = `<option value="">${recordingPlaceholder}</option>${lightRecordings.map((recording) => {
    const median = analysisMetric(recording.analysis, "median_illuminance_lx");
    const metricText = median ? ` · ${formatMetricValue(median.value)} ${median.unit}` : "";
    return `<option value="${escapeHtml(recording.session_id)}">${escapeHtml(recording.label)}${escapeHtml(metricText)} · ${escapeHtml(confidenceText(recording.analysis?.confidence))} · ${escapeHtml(formatDateTime(recording.created_at))}</option>`;
  }).join("")}`;
  if (lightRecordings.some((recording) => recording.session_id === previousRecording)) elements.investigationRecording.value = previousRecording;
  const lightProfile = state.phyphoxProbe?.sensor_profiles?.light;
  const phoneReady = Boolean(state.savedDevice && state.phyphoxProbe && lightProfile);
  elements.investigationCaptureButton.disabled = terminal || state.busy || !phoneReady;
  elements.investigationBindButton.disabled = terminal || state.busy || lightRecordings.length === 0;
  elements.investigationConfirm.disabled = terminal;
  elements.investigationObservation.disabled = terminal;
  const captureLabels = { background: "采集环境光对照", condition: "采集当前距离", replication: "重复当前距离", correction: "按纠偏条件采集" };
  const captureSpan = elements.investigationCaptureButton.querySelector("span");
  if (captureSpan) captureSpan.textContent = task ? (captureLabels[task.role] || "按当前任务从手机采集") : "实验已结束";
  elements.investigationProgress.innerHTML = [
    ["测量", `${item.progress.measurements_used} / ${item.protocol.max_measurements}`],
    ["纠偏", `${item.progress.corrections_used} / ${item.protocol.max_corrections}`],
    ["有效证据", String(item.progress.valid_evidence_count)],
    ["条件覆盖", `${Math.round(item.progress.condition_coverage_ratio * 100)}%`],
    ["质量通过", `${Math.round(item.progress.quality_pass_rate * 100)}%`],
    ["剩余预算", `${Math.max(0, item.protocol.max_measurements - item.progress.measurements_used)} 次`],
    ["现场距离上限", item.execution_constraints?.find((entry) => entry.key === "distance_m")?.maximum ? `${item.execution_constraints.find((entry) => entry.key === "distance_m").maximum} m` : "未设置"],
  ].map(([label, value]) => `<div><span>${label}</span><b>${value}</b></div>`).join("");
  const decisionLabels = { continue: "继续采集", conclude: "证据充分，形成结论", inconclusive: "证据不足，安全停止" };
  elements.investigationDecisionState.dataset.decision = item.progress.decision;
  elements.investigationDecisionState.innerHTML = `<span>当前判定</span><b>${escapeHtml(decisionLabels[item.progress.decision] || item.progress.decision)}</b><small>conclusion_ready=${item.progress.conclusion_ready} · forced_stop=${item.progress.forced_stop}</small>`;
  elements.investigationBlockers.innerHTML = item.progress.blockers.length
    ? item.progress.blockers.map((blocker) => `<li>${escapeHtml(blocker)}</li>`).join("")
    : `<li>${item.progress.conclusion_ready ? "全部预注册门禁已经满足。" : "没有额外终止阻碍。"}</li>`;
  renderInvestigationEvidence(item);
  renderInvestigationTools(item);
  renderInvestigationPlanner(item);
  elements.investigationStatus.dataset.state = terminal ? "ready" : "idle";
  if (terminal) elements.investigationStatus.textContent = "协议已经按终止向量停止。";
  else if (!state.savedDevice) elements.investigationStatus.textContent = "尚未保存默认手机；可以绑定已有记录，或先到“设备与设置”连接手机。";
  else if (!state.phyphoxProbe) elements.investigationStatus.textContent = "默认手机尚未完成检测；可以绑定已有记录，或重新检测设备。";
  else if (!lightProfile) elements.investigationStatus.textContent = "当前 phyphox 实验没有可信 Light 输入；请在手机打开 Light 后重新检测，也可以绑定已有记录。";
  else elements.investigationStatus.textContent = `已验证 Light Profile；请执行 ${task.recommended_phyphox_experiment} 并提交当前任务。`;
  renderInvestigationReport(item);
}

async function createExecutableInvestigation(exploration) {
  const protocol = state.experimentProtocols.find((item) => item.protocol_id === exploration.executable_protocol_id);
  if (!protocol) throw new Error("找不到该探索卡片绑定的精确协议版本。");
  const maxDistanceText = elements.explorationMaxDistance.value.trim();
  const maxDistance = maxDistanceText ? Number(maxDistanceText) : null;
  if (maxDistance !== null && (!Number.isFinite(maxDistance) || maxDistance < 0.5 || maxDistance > 4)) {
    throw new Error("最大可用距离必须在 0.50–4.00 m 之间，并包含协议起点 0.50 m。");
  }
  const executionConstraints = maxDistance === null ? [] : [{
    key: "distance_m",
    unit: "m",
    maximum: maxDistance,
    source: "user_confirmed",
  }];
  clearInvestigationError();
  const response = await fetch("/api/v2/investigations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: exploration.title,
      research_question: exploration.question,
      mode: "explore",
      context: `PocketLab Real-World Exploration / ${exploration.exploration_id}`,
      planning_policy: "bounded_agent",
      protocol_id: protocol.protocol_id,
      protocol_version: protocol.protocol_version,
      parameter_values: [],
      execution_constraints: executionConstraints,
    }),
  });
  if (!response.ok) throw new Error(await readApiError(response));
  state.investigation = await response.json();
  await Promise.all([refreshInvestigationHistory(), loadExplorationHistory()]);
  navigateTo(`/app/explore/runs/${encodeURIComponent(state.investigation.case_id)}`);
}

async function captureInvestigationMeasurement() {
  const item = state.investigation;
  if (!item?.current_task || state.busy) return;
  if (!state.savedDevice) return showToast("请先在设备与设置中保存默认手机。", true);
  if (!state.phyphoxProbe?.sensor_profiles?.light) return setInvestigationError("当前 phyphox 实验没有可信 Light 输入。请在手机打开 Light 后重新检测，或绑定已有记录。");
  if (!elements.investigationConfirm.checked) return showToast("请先确认控制条件和可信局域网提示。", true);
  const duration = Number(elements.investigationDuration.value);
  if (!Number.isFinite(duration) || duration < 1 || duration > 300) return showToast("采集时长必须为 1–300 秒。", true);
  const parameters = investigationParameters();
  if (parameters.some((entry) => !Number.isFinite(entry.value))) return showToast("请填写有效距离。", true);
  const observationNotes = elements.investigationObservation.value.trim();
  clearInvestigationError();
  setBusy(true, elements.investigationCaptureButton, "正在采集…");
  try {
    const response = await fetch(`/api/v2/investigations/${encodeURIComponent(item.case_id)}/phyphox`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: item.revision,
        task_id: item.current_task.task_id,
        base_url: state.savedDevice.base_url,
        duration_s: duration,
        parameters,
        controls_confirmed: true,
        privacy_acknowledged: true,
        observation_notes: observationNotes,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.investigation = data.case;
    await Promise.all([
      refreshSensorRecordings(),
      refreshInvestigationHistory(),
      loadExplorationHistory(),
    ]);
    elements.investigationConfirm.checked = false;
    elements.investigationObservation.value = "";
    clearInvestigationError();
    renderInvestigation();
    showToast(data.evidence.valid ? "测量已成为有效实验依据" : "测量已保存，协议已安排纠偏");
  } catch (error) {
    setInvestigationError(error.message);
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.investigationCaptureButton, "按当前任务从手机采集");
    renderInvestigation();
  }
}

async function bindInvestigationRecording() {
  const item = state.investigation;
  const recordingId = elements.investigationRecording.value;
  if (!item?.current_task || !recordingId || state.busy) return showToast("请选择一条已有光照记录。", true);
  if (!elements.investigationConfirm.checked) return showToast("请先确认该记录是在当前控制条件下采集的。", true);
  const observationNotes = elements.investigationObservation.value.trim();
  clearInvestigationError();
  setBusy(true, elements.investigationBindButton, "正在绑定…");
  try {
    const recordResponse = await fetch(`/api/v2/recordings/${encodeURIComponent(recordingId)}`);
    if (!recordResponse.ok) throw new Error(await readApiError(recordResponse));
    const record = await recordResponse.json();
    const provenance = record.upload.provenance;
    const response = await fetch(`/api/v2/investigations/${encodeURIComponent(item.case_id)}/measurements`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: item.revision,
        task_id: item.current_task.task_id,
        recording: {
          recording_type: "sensor_v2",
          recording_id: record.session_id,
          sensor: record.upload.sensor,
          analyzer_id: record.analysis.analyzer_id,
          analyzer_version: record.analysis.analyzer_version,
          source: provenance.source,
          config_sha256: provenance.config_sha256,
          remote_session: provenance.remote_session || null,
        },
        parameters: investigationParameters(),
        controls_confirmed: true,
        observation_notes: observationNotes,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.investigation = await response.json();
    await Promise.all([refreshInvestigationHistory(), loadExplorationHistory()]);
    elements.investigationConfirm.checked = false;
    elements.investigationObservation.value = "";
    clearInvestigationError();
    renderInvestigation();
    showToast("已有记录已绑定到当前协议任务");
  } catch (error) {
    setInvestigationError(error.message);
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.investigationBindButton, "绑定已有记录");
    renderInvestigation();
  }
}

async function refreshSensorRecordings() {
  const response = await fetch("/api/v2/recordings");
  if (!response.ok) throw new Error(await readApiError(response));
  state.sensorRecordings = await response.json();
}

function canBindRecordingToLightDistance(recording) {
  return recording?.sensor === "light" && recording?.provenance?.source !== "public_replay";
}

function renderInvestigationReport(item) {
  elements.investigationReport.hidden = !item.report;
  if (!item.report) {
    elements.investigationChart.innerHTML = "";
    elements.investigationResultTable.innerHTML = "";
    return;
  }
  elements.investigationOutcome.textContent = item.report.outcome === "completed_with_conclusion" ? "有边界结论" : "证据不足";
  elements.investigationConfidence.textContent = confidenceText(item.report.confidence);
  elements.investigationConclusion.textContent = item.report.conclusion;
  elements.investigationStopReason.textContent = `停止原因 [${item.report.stop_reason_code || "protocol-termination"}]：${item.report.stop_reason}`;
  elements.investigationKeyMetrics.innerHTML = item.report.summary_metrics.length
    ? item.report.summary_metrics.map((metric) => `<div><span>${escapeHtml(metric.label)}</span><b>${escapeHtml(formatMetricValue(metric.value))} ${escapeHtml(metric.unit)}</b></div>`).join("")
    : "<p>实验在拟合前停止，没有生成拟合摘要指标。</p>";
  elements.investigationUncertainties.innerHTML = item.report.remaining_uncertainties.length
    ? item.report.remaining_uncertainties.map((entry) => `<li>${escapeHtml(entry)}</li>`).join("")
    : "<li>报告没有登记额外不确定性。</li>";
  elements.investigationMarketBoundary.textContent = item.report.market_validated
    ? "已经过市场验证"
    : "未经过市场与跨设备验证；本报告只适用于当前实验条件";
  elements.investigationBoundaries.innerHTML = item.report.claim_boundaries.map((entry) => `<li>${escapeHtml(entry)}</li>`).join("");
  const artifact = item.artifacts[0];
  elements.investigationArtifactWarnings.innerHTML = artifact?.warnings?.length
    ? artifact.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")
    : "";
  renderInvestigationChart(artifact);
  renderInvestigationResultTable(artifact);
}

function renderInvestigationChart(artifact) {
  if (!artifact?.series?.length) {
    elements.investigationChart.innerHTML = "<p>本次在形成可视化前停止，没有图表产物。</p>";
    return;
  }
  const xLog = artifact.x_axis.scale === "log";
  const yLog = artifact.y_axis.scale === "log";
  const validPoint = (point) => Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y))
    && (!xLog || Number(point.x) > 0) && (!yLog || Number(point.y) > 0);
  const points = artifact.series.flatMap((series) => series.points).filter(validPoint);
  if (!points.length) {
    elements.investigationChart.innerHTML = "<p>Artifact 没有可绘制的有限正值；请查看数据表和报告边界。</p>";
    return;
  }
  const transformX = (value) => xLog ? Math.log10(Number(value)) : Number(value);
  const transformY = (value) => yLog ? Math.log10(Number(value)) : Number(value);
  const inverseX = (value) => xLog ? 10 ** value : value;
  const inverseY = (value) => yLog ? 10 ** value : value;
  const transformedX = points.map((point) => transformX(point.x));
  const transformedY = points.flatMap((point) => {
    const values = [transformY(point.y)];
    const error = Number(point.y_error || 0);
    if (error > 0 && Number(point.y) - error > 0) values.push(transformY(Number(point.y) - error));
    if (error > 0) values.push(transformY(Number(point.y) + error));
    return values;
  });
  const minX = Math.min(...transformedX), maxX = Math.max(...transformedX), minY = Math.min(...transformedY), maxY = Math.max(...transformedY);
  const sx = (value) => 55 + ((transformX(value) - minX) / Math.max(1e-9, maxX - minX)) * 600;
  const sy = (value) => 300 - ((transformY(value) - minY) / Math.max(1e-9, maxY - minY)) * 250;
  const colors = ["#76f4c3", "#65cfff", "#ffd392"];
  const seriesSvg = artifact.series.map((series, index) => {
    const safePoints = series.points.filter(validPoint);
    if (series.series_type === "observations") return safePoints.map((point) => {
      const error = Number(point.y_error || 0);
      const lower = Number(point.y) - error;
      const upper = Number(point.y) + error;
      const errorBar = error > 0 && upper > 0 && (!yLog || lower > 0)
        ? `<line class="investigation-error-bar" x1="${sx(point.x).toFixed(2)}" y1="${sy(lower).toFixed(2)}" x2="${sx(point.x).toFixed(2)}" y2="${sy(upper).toFixed(2)}" />`
        : "";
      return `${errorBar}<circle cx="${sx(point.x).toFixed(2)}" cy="${sy(point.y).toFixed(2)}" r="5" fill="${colors[index]}" />`;
    }).join("");
    const path = safePoints.map((point, pointIndex) => `${pointIndex ? "L" : "M"}${sx(point.x).toFixed(2)},${sy(point.y).toFixed(2)}`).join(" ");
    return `<path d="${path}" fill="none" stroke="${colors[index]}" stroke-width="2" ${series.series_type === "reference" ? 'stroke-dasharray="7 6"' : ""} />`;
  }).join("");
  const legend = artifact.series.map((series, index) => `<span><i style="background:${colors[index]}"></i>${escapeHtml(series.label)}</span>`).join("");
  const fractions = [0, 0.5, 1];
  const xTicks = fractions.map((fraction) => {
    const transformed = minX + (maxX - minX) * fraction;
    const x = 55 + 600 * fraction;
    return `<line x1="${x}" y1="300" x2="${x}" y2="306"/><text x="${x}" y="320">${escapeHtml(formatMetricValue(inverseX(transformed)))}</text>`;
  }).join("");
  const yTicks = fractions.map((fraction) => {
    const transformed = minY + (maxY - minY) * fraction;
    const y = 300 - 250 * fraction;
    return `<line x1="49" y1="${y}" x2="55" y2="${y}"/><text class="investigation-y-tick" x="43" y="${y + 4}">${escapeHtml(formatMetricValue(inverseY(transformed)))}</text>`;
  }).join("");
  const xScale = xLog ? " · log" : "";
  const yScale = yLog ? " · log" : "";
  elements.investigationChart.innerHTML = `<div class="investigation-chart-legend">${legend}</div><svg viewBox="0 0 700 350" role="img" aria-label="${escapeHtml(artifact.title)}"><text class="investigation-chart-title" x="350" y="18">${escapeHtml(artifact.title)}</text><line x1="55" y1="300" x2="665" y2="300"/><line x1="55" y1="30" x2="55" y2="300"/>${xTicks}${yTicks}${seriesSvg}<text x="350" y="343">${escapeHtml(artifact.x_axis.label)} (${escapeHtml(artifact.x_axis.unit)})${xScale}</text><text class="investigation-y-label" x="58" y="28">${escapeHtml(artifact.y_axis.label)} (${escapeHtml(artifact.y_axis.unit)})${yScale}</text></svg>`;
}

function renderInvestigationResultTable(artifact) {
  const observations = artifact?.series?.find((series) => series.series_type === "observations");
  if (!observations?.points?.length) {
    elements.investigationResultTable.innerHTML = "<p>本次没有形成可核验的条件聚合表。</p>";
    return;
  }
  elements.investigationResultTable.innerHTML = `
    <table>
      <caption>用于拟合的观测条件</caption>
      <thead><tr><th>距离 (${escapeHtml(artifact.x_axis.unit)})</th><th>净照度 (${escapeHtml(artifact.y_axis.unit)})</th><th>MAD</th><th>重复数</th></tr></thead>
      <tbody>${observations.points.map((point) => `<tr><td>${escapeHtml(formatMetricValue(point.x))}</td><td>${escapeHtml(formatMetricValue(point.y))}</td><td>${escapeHtml(formatMetricValue(point.y_error || 0))}</td><td>${point.evidence_ids.length}</td></tr>`).join("")}</tbody>
    </table>`;
}

const PUBLIC_REPLAY_DATA_CLASS_LABELS = {
  public_real_phone_raw: "公开真机原始",
  public_real_phone_derived: "公开真机派生",
  source_numeric_replay: "来源数值回放",
  synthetic: "合成数据（非真机证据）",
};

function publicReplayDataClassLabel(dataClass) {
  return PUBLIC_REPLAY_DATA_CLASS_LABELS[dataClass] || dataClass || "数据类别未知";
}

function publicReplayStatusLabel(status) {
  return ({
    source_validated: "来源已校验",
    public_replay_ready: "公开回放已就绪",
    not_evaluated: "尚未评测",
    fail: "未通过",
    pass: "已通过",
  })[status] || status || "状态未知";
}

function currentPublicReplayDataset() {
  return state.publicReplays.find((item) => item.dataset_id === elements.publicReplayDataset.value) || null;
}

function safeExternalUrl(value) {
  try {
    const url = new URL(String(value));
    return url.protocol === "https:" ? url.href : "";
  } catch (error) {
    return "";
  }
}

const PUBLIC_LIGHT_RATIONALE_LABELS = {
  match_temporal_perturbation_goal: "匹配短时扰动问题",
  match_registered_condition_comparison: "匹配已登记条件比较",
  match_naturalistic_context_goal: "匹配自然场景分布问题",
  add_phone_transfer_crosscheck: "增加手机数据迁移交叉检查",
  request_missing_live_evidence: "公开证据不足，建议补充真机测量",
  minimal_sufficient_evidence: "已达到最小充分证据",
  unsupported_claim_boundary: "问题超出公开证据允许的结论边界",
  privacy_not_acknowledged: "尚未确认本地隐私边界",
  strong_workflow_fallback: "使用强工作流安全回退",
};

const PUBLIC_LIGHT_TOOL_LABELS = {
  inspect_public_light_trace: "检查公开 Light 时间序列",
  compare_registered_light_conditions: "比较已登记光照条件",
  summarize_naturalistic_light_context: "汇总自然场景照度分布",
  audit_light_claim_support: "审计结论与证据支持关系",
};

function publicReplayCanImport(dataset) {
  return dataset?.import_allowed === true;
}

function updatePublicLightAvailability() {
  const question = elements.publicLightQuestion.value.trim();
  const luxText = elements.publicLightQueryLux.value.trim();
  const lux = luxText ? Number(luxText) : null;
  const validLux = lux === null || (Number.isFinite(lux) && lux >= 0 && lux <= 1_000_000_000);
  elements.publicLightRunButton.disabled = state.busy
    || question.length < 5
    || question.length > 800
    || !validLux
    || !elements.publicLightPrivacy.checked;
}

async function publicLightApiError(response) {
  const detail = await readApiError(response);
  if (response.status === 401) return `登录状态已失效：${detail}`;
  if (response.status === 403) return `本地运行权限被拒绝：${detail}`;
  if (response.status === 503) return `公开来源或工具安全校验暂不可用：${detail}`;
  return detail;
}

async function runPublicLightExploration() {
  const question = elements.publicLightQuestion.value.trim();
  const luxText = elements.publicLightQueryLux.value.trim();
  const queryLux = luxText ? Number(luxText) : null;
  if (question.length < 5 || question.length > 800) {
    elements.publicLightStatus.dataset.state = "error";
    elements.publicLightStatus.textContent = "探索问题必须包含 5–800 个字符。";
    return;
  }
  if (queryLux !== null && (!Number.isFinite(queryLux) || queryLux < 0 || queryLux > 1_000_000_000)) {
    elements.publicLightStatus.dataset.state = "error";
    elements.publicLightStatus.textContent = "查询照度必须在 0–1,000,000,000 lx 之间，或留空。";
    return;
  }
  if (!elements.publicLightPrivacy.checked) {
    elements.publicLightStatus.dataset.state = "error";
    elements.publicLightStatus.textContent = "每次运行前都必须显式确认本地隐私边界。";
    return;
  }

  state.publicLightError = "";
  state.publicLightRun = null;
  elements.publicLightResult.hidden = true;
  elements.publicLightStatus.dataset.state = "loading";
  elements.publicLightStatus.textContent = "Agent 正在受限候选中选择证据，并运行确定性工具…";
  setBusy(true, elements.publicLightRunButton, "正在运行闭环…");
  try {
    const body = {
      research_question: question,
      privacy_acknowledged: true,
    };
    if (queryLux !== null) body.query_illuminance_lx = queryLux;
    const response = await fetch("/api/v2/public-replays/light/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await publicLightApiError(response));
    state.publicLightRun = await response.json();
    await loadExplorationHistory();
    renderPublicLightResult(state.publicLightRun);
    elements.publicLightStatus.dataset.state = state.publicLightRun.execution_status === "completed" ? "ready" : "warning";
    const statusLabel = ({ completed: "已完成", limited: "已形成有边界结果", unsupported: "已安全拒绝不受支持的问题" })[state.publicLightRun.execution_status]
      || state.publicLightRun.execution_status;
    elements.publicLightStatus.textContent = `公开 Light 闭环${statusLabel}；报告与审计轨迹已保存到探索历史，未复制公开原始序列。再次运行前需要重新确认隐私边界。`;
    showToast("公开 Light 闭环已生成有来源的报告");
  } catch (error) {
    state.publicLightError = error.message;
    elements.publicLightStatus.dataset.state = "error";
    elements.publicLightStatus.textContent = `公开 Light 闭环运行失败：${error.message}`;
    showToast(error.message, true);
  } finally {
    elements.publicLightPrivacy.checked = false;
    setBusy(false, elements.publicLightRunButton, "运行公开 Light 闭环");
    updatePublicLightAvailability();
  }
}

function renderPublicLightResult(result) {
  const report = result.report;
  const executionLabels = { completed: "已完成", limited: "有边界", unsupported: "不支持" };
  const plannerLabels = { accepted: "Agent 已接受", fallback: "强工作流回退", mixed: "Agent / 回退混合" };
  elements.publicLightResult.hidden = false;
  elements.publicLightReportTitle.textContent = report.title;
  elements.publicLightExecutionStatus.textContent = executionLabels[result.execution_status] || result.execution_status;
  elements.publicLightPlannerStatus.textContent = plannerLabels[result.planner_status] || result.planner_status;
  elements.publicLightReportSummary.textContent = report.summary;
  elements.publicLightGates.innerHTML = [
    ["Gate C credited", String(report.gate_c_credited_records)],
    ["Gate E", report.gate_e_status],
    ["Gate H", report.gate_h_status],
    ["Public replay ready", String(report.public_replay_ready)],
    ["Market validated", String(report.market_validated)],
    ["Agent ready", String(report.agent_ready)],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`).join("");

  elements.publicLightPlannerTrace.innerHTML = result.planner_trace.length
    ? result.planner_trace.map((trace) => {
      const source = trace.source === "agent" ? "受限 Agent" : "强工作流安全回退";
      const outcome = trace.outcome === "accepted" ? "已接受" : "已回退";
      const rationale = PUBLIC_LIGHT_RATIONALE_LABELS[trace.rationale_code] || trace.rationale_code;
      const runtime = trace.runtime_trace;
      const elapsedText = runtime && Number.isFinite(runtime.elapsed_s)
        ? `${formatMetricValue(runtime.elapsed_s)} s`
        : "耗时未上报";
      const tokenText = runtime?.usage_reported === false
        ? "token 用量未上报"
        : Number.isInteger(runtime?.total_tokens)
          ? `${runtime.total_tokens} tokens`
          : "token 用量未知";
      const runtimeText = runtime
        ? `${runtime.model || "模型未披露"} · ${runtime.transport || trace.transport} · ${elapsedText} · ${tokenText}`
        : `${trace.transport} · ${elapsedText} · ${tokenText}`;
      return `<article data-source="${escapeHtml(trace.source)}">
        <header><b>STEP ${trace.step} · ${escapeHtml(source)}</b><span>${escapeHtml(outcome)}</span></header>
        <p>选择：<code>${escapeHtml(trace.selected_candidate_id)}</code></p>
        <small>理由：${escapeHtml(rationale)}（${escapeHtml(trace.rationale_code)}）</small>
        <small>候选：${escapeHtml(trace.candidate_ids.join(" · "))}</small>
        <small>运行：${escapeHtml(runtimeText)}</small>
        ${trace.fallback_reason ? `<small>回退原因：${escapeHtml(trace.fallback_reason)}</small>` : ""}
      </article>`;
    }).join("")
    : "<p>本次没有调用规划器。</p>";

  elements.publicLightToolTrace.innerHTML = result.tool_trace.length
    ? result.tool_trace.map((execution) => `<article data-status="${escapeHtml(execution.status)}">
        <header><b>${execution.sequence}. ${escapeHtml(PUBLIC_LIGHT_TOOL_LABELS[execution.tool_id] || execution.tool_id)}</b><span>${escapeHtml(execution.status)}</span></header>
        <small>evidence_ids：${escapeHtml(execution.evidence_ids.join(" · ") || "无")}</small>
        <small>result_codes：${escapeHtml(execution.result_codes.join(" · ") || "无")}</small>
      </article>`).join("")
    : "<p>本次在调用确定性工具前安全停止。</p>";

  elements.publicLightEvidence.innerHTML = result.evidence.length
    ? result.evidence.map(renderPublicLightEvidence).join("")
    : "<p>本次没有读取公开来源，也没有形成证据快照。</p>";
  elements.publicLightFindings.innerHTML = report.supported_findings.length
    ? report.supported_findings.map((finding) => `<article><b>${escapeHtml(finding.text)}</b><small>evidence_ids：${escapeHtml(finding.evidence_ids.join(" · "))}</small></article>`).join("")
    : "<p>没有形成可由证据直接支持的发现。</p>";
  elements.publicLightSources.innerHTML = report.sources.length
    ? report.sources.map((source) => {
      const sourceUrl = safeExternalUrl(source.source_url);
      const doiUrl = source.doi
        ? safeExternalUrl(String(source.doi).startsWith("https://") ? source.doi : `https://doi.org/${source.doi}`)
        : "";
      return `<article class="public-light-source-card">
        <b>${escapeHtml(publicReplayDataClassLabel(source.data_class))} · Gate C 计入 ${source.gate_c_eligible ? "YES" : "NO"}</b>
        <span>${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(source.source_title)} ↗</a>` : escapeHtml(source.source_title)}</span>
        <small>${escapeHtml(source.license_spdx)} · ${escapeHtml(source.device_scope)}${doiUrl ? ` · <a href="${escapeHtml(doiUrl)}" target="_blank" rel="noreferrer">DOI ${escapeHtml(source.doi)}</a>` : ""}</small>
      </article>`;
    }).join("")
    : "<p>本次没有访问公开来源。</p>";
  elements.publicLightUncertainties.innerHTML = report.uncertainties.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.publicLightForbiddenClaims.innerHTML = report.forbidden_claims.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.publicLightNextLive.hidden = !report.next_live_measurement;
  elements.publicLightNextLive.querySelector("p").textContent = report.next_live_measurement || "";
}

function renderPublicLightEvidence(evidence) {
  const sourceUrl = safeExternalUrl(evidence.source_url);
  const doiUrl = evidence.doi
    ? safeExternalUrl(String(evidence.doi).startsWith("https://") ? evidence.doi : `https://doi.org/${evidence.doi}`)
    : "";
  const factItems = evidence.facts.length
    ? evidence.facts.map((fact) => `<li><code>${escapeHtml(fact.key)}</code><b>${escapeHtml(formatMetricValue(fact.value))} ${escapeHtml(fact.unit)}</b></li>`).join("")
    : "<li>没有可安全显示的数值事实。</li>";
  const analyses = evidence.analyses.length
    ? evidence.analyses.map((analysis) => `${analysis.analyzer_id}@${analysis.analyzer_version} · ${confidenceText(analysis.confidence)}`).join("；")
    : "未附带单记录分析快照";
  return `<article class="public-light-evidence-card" data-data-class="${escapeHtml(evidence.data_class)}">
    <header><div><span>${escapeHtml(publicReplayDataClassLabel(evidence.data_class))}</span><b>${escapeHtml(evidence.evidence_id)}</b></div><strong>Gate C = 0</strong></header>
    <p>${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(evidence.source_title)} ↗</a>` : escapeHtml(evidence.source_title)}</p>
    <small>${escapeHtml(evidence.license_spdx)} · ${escapeHtml(evidence.device_scope)}${doiUrl ? ` · <a href="${escapeHtml(doiUrl)}" target="_blank" rel="noreferrer">DOI ${escapeHtml(evidence.doi)}</a>` : ""} · ${escapeHtml(analyses)}</small>
    <small>recording_ids：${escapeHtml(evidence.recording_ids.join(" · ") || "数据集聚合")}</small>
    <ul class="public-light-facts">${factItems}</ul>
    <details><summary>处理披露</summary><ul>${evidence.processing_disclosures.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>
    <details open><summary>结论边界</summary><ul>${evidence.claim_boundary.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>
  </article>`;
}

const PUBLIC_PRESSURE_RATIONALE_LABELS = {
  match_elevator_goal: "匹配电梯相对高度问题",
  match_stairwell_goal: "匹配楼梯相对高度问题",
  request_live_device_evidence: "问题需要当前手机证据",
  unsupported_claim_boundary: "问题超出相对压力证据边界",
  evidence_quality_sufficient: "稳定端点质量足以形成有边界报告",
  evidence_quality_insufficient: "证据质量不足，转入真机复核",
  privacy_not_acknowledged: "尚未确认本地公开回放边界",
  strong_workflow_fallback: "使用冻结的强工作流安全回退",
};

const PUBLIC_PRESSURE_TOOL_LABELS = {
  inspect_pressure_trace: "检查压力时间轴与稳定端点",
  compare_pressure_height_to_ground_truth: "用隐藏的 NIST 稀疏高程校验相对高度",
  audit_pressure_claim_support: "审计相对压力结论边界",
};

function updatePublicPressureAvailability() {
  const question = elements.publicPressureQuestion.value.trim();
  elements.publicPressureRunButton.disabled = state.busy
    || question.length < 5
    || question.length > 800
    || !elements.publicPressurePrivacy.checked;
}

async function publicPressureApiError(response) {
  const detail = await readApiError(response);
  if (response.status === 401) return `登录状态已失效：${detail}`;
  if (response.status === 403) return `本地运行权限被拒绝：${detail}`;
  if (response.status === 503) return `Pressure 来源或物理工具校验暂不可用：${detail}`;
  return detail;
}

async function runPublicPressureExploration() {
  const question = elements.publicPressureQuestion.value.trim();
  if (question.length < 5 || question.length > 800) {
    elements.publicPressureStatus.dataset.state = "error";
    elements.publicPressureStatus.textContent = "Pressure 探索问题必须包含 5–800 个字符。";
    return;
  }
  if (!elements.publicPressurePrivacy.checked) {
    elements.publicPressureStatus.dataset.state = "error";
    elements.publicPressureStatus.textContent = "每次运行前都必须确认本地公开 Pressure 回放边界。";
    return;
  }

  state.publicPressureError = "";
  state.publicPressureRun = null;
  elements.publicPressureResult.hidden = true;
  elements.publicPressureStatus.dataset.state = "loading";
  elements.publicPressureStatus.textContent = "Pressure Agent 正在受限候选中选择证据，并运行服务端物理质量门…";
  setBusy(true, elements.publicPressureRunButton, "正在运行 Pressure Beta…");
  try {
    const response = await fetch("/api/v2/public-replays/pressure/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        research_question: question,
        privacy_acknowledged: true,
      }),
    });
    if (!response.ok) throw new Error(await publicPressureApiError(response));
    state.publicPressureRun = await response.json();
    await loadExplorationHistory();
    renderPublicPressureResult(state.publicPressureRun);
    elements.publicPressureStatus.dataset.state = state.publicPressureRun.execution_status === "completed" ? "ready" : "warning";
    const statusLabel = ({ completed: "已完成", limited: "已形成有边界结果", unsupported: "已安全拒绝不受支持的问题" })[state.publicPressureRun.execution_status]
      || state.publicPressureRun.execution_status;
    elements.publicPressureStatus.textContent = `公开 Pressure Agent Beta ${statusLabel}；报告与来源摘要已保存到探索历史。再次运行前需要重新确认本地边界。`;
    showToast("公开 Pressure Agent Beta 已生成有来源的报告");
  } catch (error) {
    state.publicPressureError = error.message;
    elements.publicPressureStatus.dataset.state = "error";
    elements.publicPressureStatus.textContent = `公开 Pressure Agent Beta 运行失败：${error.message}`;
    showToast(error.message, true);
  } finally {
    elements.publicPressurePrivacy.checked = false;
    setBusy(false, elements.publicPressureRunButton, "运行公开 Pressure Agent Beta");
    updatePublicPressureAvailability();
  }
}

function renderPublicPressureResult(result) {
  const report = result.report;
  const executionLabels = { completed: "已完成", limited: "有边界", unsupported: "不支持" };
  const plannerLabels = { accepted: "Agent 已接受", fallback: "强工作流回退", mixed: "Agent / 回退混合" };
  elements.publicPressureResult.hidden = false;
  elements.publicPressureReportTitle.textContent = report.title;
  elements.publicPressureExecutionStatus.textContent = executionLabels[result.execution_status] || result.execution_status;
  elements.publicPressurePlannerStatus.textContent = plannerLabels[result.planner_status] || result.planner_status;
  elements.publicPressureReportSummary.textContent = report.summary;
  elements.publicPressureGates.innerHTML = [
    ["Gate C credited", String(report.gate_c_credited_records)],
    ["Gate E", report.gate_e_status],
    ["Gate H", report.gate_h_status],
    ["Public replay ready", String(report.public_replay_ready)],
    ["Market validated", String(report.market_validated)],
    ["Agent ready", String(report.agent_ready)],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`).join("");

  elements.publicPressurePlannerTrace.innerHTML = result.planner_trace.length
    ? result.planner_trace.map((trace) => {
      const source = trace.source === "agent" ? "受限 Agent" : "强工作流安全回退";
      const rationale = PUBLIC_PRESSURE_RATIONALE_LABELS[trace.rationale_code] || trace.rationale_code;
      const runtime = trace.runtime_trace;
      const elapsedText = runtime && Number.isFinite(runtime.elapsed_s) ? `${formatMetricValue(runtime.elapsed_s)} s` : "耗时未上报";
      const tokenText = runtime?.usage_reported === false
        ? "token 用量未上报"
        : Number.isInteger(runtime?.total_tokens) ? `${runtime.total_tokens} tokens` : "token 用量未知";
      return `<article data-source="${escapeHtml(trace.source)}">
        <header><b>STEP ${trace.step} · ${escapeHtml(source)}</b><span>${escapeHtml(trace.outcome)}</span></header>
        <p>选择：<code>${escapeHtml(trace.selected_candidate_id)}</code></p>
        <small>理由：${escapeHtml(rationale)}（${escapeHtml(trace.rationale_code)}）</small>
        <small>候选：${escapeHtml(trace.candidate_ids.join(" · "))}</small>
        <small>运行：${escapeHtml(runtime?.transport || trace.transport)} · ${escapeHtml(elapsedText)} · ${escapeHtml(tokenText)}</small>
        ${trace.fallback_reason ? `<small>回退原因：${escapeHtml(trace.fallback_reason)}</small>` : ""}
      </article>`;
    }).join("")
    : "<p>本次没有调用 Pressure Planner。</p>";

  elements.publicPressureToolTrace.innerHTML = result.tool_trace.length
    ? result.tool_trace.map((execution) => `<article data-status="${escapeHtml(execution.status)}">
        <header><b>${execution.sequence}. ${escapeHtml(PUBLIC_PRESSURE_TOOL_LABELS[execution.tool_id] || execution.tool_id)}</b><span>${escapeHtml(execution.status)}</span></header>
        <small>evidence_ids：${escapeHtml(execution.evidence_ids.join(" · ") || "无")}</small>
        <small>result_codes：${escapeHtml(execution.result_codes.join(" · ") || "无")}</small>
      </article>`).join("")
    : "<p>本次在读取 Pressure 来源前由服务端安全停止。</p>";

  elements.publicPressureEvidence.innerHTML = result.evidence.length
    ? result.evidence.map(renderPublicPressureEvidence).join("")
    : "<p>本次没有读取公开 Pressure 来源，也没有形成证据快照。</p>";
  elements.publicPressureFindings.innerHTML = report.supported_findings.length
    ? report.supported_findings.map((finding) => `<article><b>${escapeHtml(finding.text)}</b><small>evidence_ids：${escapeHtml(finding.evidence_ids.join(" · "))}</small></article>`).join("")
    : "<p>没有形成可由证据直接支持的 Pressure 发现。</p>";
  elements.publicPressureSources.innerHTML = report.sources.length
    ? report.sources.map(renderPublicPressureSource).join("")
    : "<p>本次没有访问公开 Pressure 来源。</p>";
  elements.publicPressureUncertainties.innerHTML = report.uncertainties.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.publicPressureForbiddenClaims.innerHTML = report.forbidden_claims.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.publicPressureNextLive.hidden = !report.next_live_measurement;
  elements.publicPressureNextLive.querySelector("p").textContent = report.next_live_measurement || "";
}

function renderPublicPressureSource(source) {
  const sourceUrl = safeExternalUrl(source.source_url);
  const doiUrl = safeExternalUrl(`https://doi.org/${source.doi}`);
  return `<article class="public-light-source-card">
    <b>${escapeHtml(publicReplayDataClassLabel(source.data_class))} · Gate C 计入 ${source.gate_c_eligible ? "YES" : "NO"}</b>
    <span>${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(source.source_title)} ↗</a>` : escapeHtml(source.source_title)}</span>
    <small>${escapeHtml(source.license_spdx)} · ${escapeHtml(source.device_scope)}${doiUrl ? ` · <a href="${escapeHtml(doiUrl)}" target="_blank" rel="noreferrer">DOI ${escapeHtml(source.doi)}</a>` : ""}</small>
  </article>`;
}

function renderPublicPressureEvidence(evidence) {
  const inspection = evidence.inspection;
  const comparison = evidence.comparison;
  const metricItems = inspection.metrics.map((metric) => `<li><code>${escapeHtml(metric.key)}</code><b>${escapeHtml(formatMetricValue(metric.value))} ${escapeHtml(metric.unit)}</b></li>`).join("");
  const comparisonText = comparison.evaluable
    ? `公开相对高程 ${formatMetricValue(comparison.ground_truth_height_change_m)} m · 绝对误差 ${formatMetricValue(comparison.absolute_error_m)} m · ${comparison.status}`
    : `物理对照不可评估：${comparison.missing_requirements.join("；")}`;
  return `<article class="public-light-evidence-card" data-data-class="${escapeHtml(evidence.data_class)}">
    <header><div><span>${escapeHtml(publicReplayDataClassLabel(evidence.data_class))}</span><b>${escapeHtml(evidence.evidence_id)}</b></div><strong>Gate C = 0</strong></header>
    <p>${escapeHtml(evidence.source_title)}</p>
    <small>${escapeHtml(evidence.license_spdx)} · ${escapeHtml(evidence.device_scope)} · ${escapeHtml(inspection.analyzer_id || "pocketlab.pressure.v2")}</small>
    <small>稳定平台：${inspection.platforms_passed ? "通过" : "未通过"} · 置信度：${escapeHtml(confidenceText(inspection.confidence))} · ${escapeHtml(comparisonText)}</small>
    <ul class="public-light-facts">${metricItems}</ul>
    <details><summary>处理披露</summary><ul>${evidence.processing_disclosures.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>
    <details open><summary>结论边界</summary><ul>${evidence.claim_boundary.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>
  </article>`;
}

const PUBLIC_SENSOR_BETA_CONFIG = {
  "walking-cadence-public-exploration.v1": {
    label: "Accelerometer · 步频",
    protocolId: "walking-cadence-public-exploration.v1",
    intro: "模型在三段真实 Android 楼梯记录、单段检查、真机路面对照和医疗安全拒绝中选择；服务端独占 Hz/SNR/CV 门与报告终止。",
    placeholder: "例如：三段公开楼梯上行的步频候选是否稳定？",
    help: "5–800 字；草地/瓷砖等新路面会转入真机设计，医疗诊断与身份推断会安全拒绝。",
  },
  "elevator-motion-public-exploration.v1": {
    label: "Accelerometer · 电梯阶段",
    protocolId: "elevator-motion-public-exploration.v1",
    intro: "模型在完整行程、两段半程、三段对照、真机请求和位移积分拒绝中选择；服务端独占阶段分割阈值与终止。",
    placeholder: "例如：三段公开电梯记录能否重复检测加速、稳定和减速？",
    help: "5–800 字；具体楼层和当前电梯会转入真机实验，双积分位移、监控和安全认证会拒绝。",
  },
  "vibration-response-public-exploration.v1": {
    label: "Accelerometer · 振动响应",
    protocolId: "vibration-response-public-exploration.v1",
    intro: "模型在静止锚点、手持运动、成对响应、真机设备诊断和安全停止中选择；公开回放只验证测量链，不冒充设备故障证据。",
    placeholder: "例如：公开静止和手持加速度能否验证测量链响应？",
    help: "5–800 字；偏载、松动和传振会转入当前设备真机对照，漏电、漏水、焦糊或结构危险会停止。",
  },
  gyroscope: {
    label: "Gyroscope",
    protocolId: "gyroscope-public-exploration.v1",
    intro: "模型在静止锚点、手持转动、成对对照、真机请求和安全拒绝中选择；服务端独占 rad/s 分析、状态分离阈值和报告终止。",
    placeholder: "例如：公开陀螺仪记录能否区分静止和手持转动？",
    help: "5–800 字；当前手机状态、精确转角和绝对姿态会转入真机测量或安全拒绝。",
  },
  magnetometer: {
    label: "Magnetometer",
    protocolId: "magnetometer-public-exploration.v1",
    intro: "模型在稳定背景、场变化、成对对照、真机请求和安全拒绝中选择；服务端独占 uT/accuracy 分析、变化阈值、因果边界和报告终止。",
    placeholder: "例如：公开磁力计记录能否区分稳定背景和局部场变化？",
    help: "5–800 字；物体识别、空间定位、绝对航向和当前手机因果问题会转入真机测量或安全拒绝。",
  },
  proximity: {
    label: "Proximity",
    protocolId: "proximity-public-exploration.v1",
    intro: "模型在早期事件、后期事件、成对二态对照、真机请求和安全拒绝中选择；服务端独占 0/5 cm 状态门、稀疏事件边界和报告终止。",
    placeholder: "例如：公开接近传感器事件是二态还是连续距离？",
    help: "5–800 字；真实触发距离、材质/角度因果与人物监控请求会转入真机测量或安全拒绝。",
  },
  microphone: {
    label: "Microphone",
    protocolId: "microphone-public-exploration.v1",
    intro: "模型在公开轨迹前段、后段、相对级别对照、真机请求和隐私拒绝中选择；服务端独占来源校验、派生级别质量门、无原始音频边界和报告终止。",
    placeholder: "例如：公开声音派生级别序列的前后变化范围是否明显？",
    help: "5–800 字；原始音频、转写、说话人识别、校准 SPL 和房间因果会被拒绝或转入真机测量。",
  },
  location: {
    label: "Location",
    protocolId: "location-public-exploration.v1",
    intro: "模型在两次隐私变换路线、成对几何对照、真机请求和位置安全拒绝中选择；服务端独占来源校验、相对轨迹质量门、坐标脱敏边界和报告终止。",
    placeholder: "例如：两次公开相似路线的长度和相对形状有多一致？",
    help: "5–800 字；真实地址、绝对坐标、人员跟踪与当前手机绝对误差会被拒绝或转入真机测量。",
  },
};

const PUBLIC_SENSOR_RATIONALE_LABELS = {
  match_lower_stair_cadence: "匹配第一段公开楼梯步频",
  match_middle_stair_cadence: "匹配第二段公开楼梯步频",
  compare_stair_cadence_repeats: "需要三段楼梯步频重复性对照",
  match_full_elevator_ascent: "匹配一段完整电梯上行",
  compare_half_elevator_ascents: "需要两段半程电梯对照",
  compare_elevator_phase_repeats: "需要三段电梯阶段重复性对照",
  match_stationary_acceleration: "匹配静止加速度基线",
  match_handheld_acceleration: "匹配手持加速度响应",
  compare_acceleration_motion_states: "需要静止—运动响应分离",
  match_stationary_bias_goal: "匹配静止零偏问题",
  match_handheld_response_goal: "匹配手持角运动问题",
  compare_motion_states: "需要静止—手持成对对照",
  request_live_device_evidence: "问题需要当前手机证据",
  unsupported_claim_boundary: "问题超出当前传感器证据边界",
  evidence_quality_sufficient: "确定性质量门足以形成受限报告",
  evidence_quality_insufficient: "证据质量不足，转入真机复核",
  privacy_not_acknowledged: "尚未确认本地公开回放边界",
  match_stable_field_goal: "匹配稳定背景问题",
  match_field_change_goal: "匹配磁场变化问题",
  compare_field_states: "需要稳定—变化成对对照",
  match_early_event_slice: "匹配较早的稀疏事件切片",
  match_late_event_slice: "匹配较晚的稀疏事件切片",
  compare_binary_event_slices: "需要前后二态事件对照",
  match_early_relative_window: "匹配公开轨迹前段相对级别",
  match_late_relative_window: "匹配公开轨迹后段相对级别",
  compare_chronological_relative_levels: "需要前后派生相对级别对照",
  match_location_route_a: "匹配第一次公开路线 acquisition",
  match_location_route_b: "匹配第二次公开路线 acquisition",
  compare_repeated_route_geometry: "需要两次相似路线的相对几何对照",
};

const PUBLIC_SENSOR_TOOL_LABELS = {
  analyze_accelerometer_recording: "运行 Accelerometer v2 专用确定性分析器",
  compare_stair_cadence_repeats: "应用楼梯步频频带、SNR 与 CV 质量门",
  segment_elevator_motion_phases: "分割正加速—稳定—减速阶段",
  compare_elevator_phase_sequences: "应用电梯阶段顺序与持续时间门",
  compare_acceleration_motion_states: "应用静止—运动 RMS 响应分离门",
  analyze_gyroscope_recording: "运行 Gyroscope 专用确定性分析器",
  compare_gyroscope_motion_states: "应用静止—手持预注册质量门",
  analyze_magnetometer_recording: "运行 Magnetometer 专用确定性分析器",
  compare_magnetic_field_states: "应用稳定—变化预注册质量门",
  analyze_proximity_event_slice: "运行 Proximity 稀疏事件专用分析器",
  compare_proximity_state_codes: "应用二态编码与前后一致性质量门",
  analyze_microphone_relative_window: "运行 Microphone 派生相对级别专用分析器",
  compare_microphone_chronological_windows: "应用前后窗口相对级别质量门",
  analyze_location_relative_route: "运行 Location 相对路线专用确定性分析器",
  compare_location_repeated_routes: "应用重复路线长度、形状与终点质量门",
};

function publicSensorBetaConfig(sensor, protocolId = null) {
  return PUBLIC_SENSOR_BETA_CONFIG[protocolId] || PUBLIC_SENSOR_BETA_CONFIG[sensor];
}

function updatePublicSensorAvailability() {
  const question = elements.publicSensorQuestion.value.trim();
  const config = publicSensorBetaConfig(state.publicSensorActive, state.publicSensorProtocol);
  elements.publicSensorRunButton.disabled = state.busy
    || !config
    || question.length < 5
    || question.length > 800
    || !elements.publicSensorPrivacy.checked;
}

async function publicSensorApiError(response) {
  const detail = await readApiError(response);
  if (response.status === 401) return `登录状态已失效：${detail}`;
  if (response.status === 403) return `本地运行权限被拒绝：${detail}`;
  if (response.status === 503) return `专用协议、来源或确定性工具暂不可用：${detail}`;
  return detail;
}

async function runPublicSensorExploration() {
  const sensor = state.publicSensorActive;
  const protocolId = state.publicSensorProtocol;
  const config = publicSensorBetaConfig(sensor, protocolId);
  const question = elements.publicSensorQuestion.value.trim();
  if (!config) {
    elements.publicSensorStatus.dataset.state = "error";
    elements.publicSensorStatus.textContent = "尚未选择可执行的 Sensor Agent Beta 协议。";
    return;
  }
  if (question.length < 5 || question.length > 800) {
    elements.publicSensorStatus.dataset.state = "error";
    elements.publicSensorStatus.textContent = `${config.label} 探索问题必须包含 5–800 个字符。`;
    return;
  }
  if (!elements.publicSensorPrivacy.checked) {
    elements.publicSensorStatus.dataset.state = "error";
    elements.publicSensorStatus.textContent = "每次运行前都必须确认本地公开传感器回放边界。";
    return;
  }

  state.publicSensorError = "";
  state.publicSensorRun = null;
  elements.publicSensorResult.hidden = true;
  elements.publicSensorStatus.dataset.state = "loading";
  elements.publicSensorStatus.textContent = `${config.label} Agent 正在冻结候选中选择证据，并运行专用质量门…`;
  setBusy(true, elements.publicSensorRunButton, `正在运行 ${config.label} Beta…`);
  try {
    const response = await fetch(`/api/v2/public-replays/sensors/${encodeURIComponent(sensor)}/explore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sensor,
        protocol_id: protocolId,
        research_question: question,
        privacy_acknowledged: true,
      }),
    });
    if (!response.ok) throw new Error(await publicSensorApiError(response));
    state.publicSensorRun = await response.json();
    await loadExplorationHistory();
    renderPublicSensorResult(state.publicSensorRun);
    elements.publicSensorStatus.dataset.state = state.publicSensorRun.execution_status === "completed" ? "ready" : "warning";
    const statusLabel = ({ completed: "已完成", limited: "已形成有边界结果", unsupported: "已安全拒绝" })[state.publicSensorRun.execution_status]
      || state.publicSensorRun.execution_status;
    elements.publicSensorStatus.textContent = `公开 ${config.label} Agent Beta ${statusLabel}；报告与审计轨迹已保存到探索历史。再次运行前需要重新确认。`;
    showToast(`公开 ${config.label} Agent Beta 已生成有来源报告`);
  } catch (error) {
    state.publicSensorError = error.message;
    elements.publicSensorStatus.dataset.state = "error";
    elements.publicSensorStatus.textContent = `公开 ${config.label} Agent Beta 运行失败：${error.message}`;
    showToast(error.message, true);
  } finally {
    elements.publicSensorPrivacy.checked = false;
    setBusy(false, elements.publicSensorRunButton, "运行公开 Sensor Agent Beta");
    updatePublicSensorAvailability();
  }
}

function renderPublicSensorResult(result) {
  const report = result.report;
  const executionLabels = { completed: "已完成", limited: "有边界", unsupported: "不支持" };
  const hasServerTermination = result.planner_trace.some((trace) => trace.fallback_reason === "server-owned-termination");
  const plannerLabels = { accepted: "Agent 已接受", fallback: "强工作流回退", mixed: hasServerTermination ? "Agent + 服务端终止" : "Agent / 回退混合" };
  elements.publicSensorResult.hidden = false;
  elements.publicSensorReportTitle.textContent = report.title;
  elements.publicSensorExecutionStatus.textContent = executionLabels[result.execution_status] || result.execution_status;
  elements.publicSensorPlannerStatus.textContent = plannerLabels[result.planner_status] || result.planner_status;
  elements.publicSensorReportSummary.textContent = report.summary;
  elements.publicSensorGates.innerHTML = [
    ["Gate C credited", String(report.gate_c_credited_records)],
    ["Gate E", report.gate_e_status],
    ["Gate H", report.gate_h_status],
    ["Public replay ready", String(report.public_replay_ready)],
    ["Market validated", String(report.market_validated)],
    ["Agent ready", String(report.agent_ready)],
  ].map(([label, value]) => `<div><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`).join("");

  elements.publicSensorPlannerTrace.innerHTML = result.planner_trace.length
    ? result.planner_trace.map((trace) => {
      const source = trace.source === "agent"
        ? "受限 Agent"
        : trace.fallback_reason === "server-owned-termination"
          ? "服务端终止策略"
          : "强工作流安全回退";
      const rationale = PUBLIC_SENSOR_RATIONALE_LABELS[trace.rationale_code] || trace.rationale_code;
      const runtime = trace.runtime_trace;
      const elapsedText = runtime && Number.isFinite(runtime.elapsed_s) ? `${formatMetricValue(runtime.elapsed_s)} s` : "耗时未上报";
      const tokenText = runtime?.usage_reported === false
        ? "token 用量未上报"
        : Number.isInteger(runtime?.total_tokens) ? `${runtime.total_tokens} tokens` : "token 用量未知";
      return `<article data-source="${escapeHtml(trace.source)}">
        <header><b>STEP ${trace.step} · ${escapeHtml(source)}</b><span>${escapeHtml(trace.outcome)}</span></header>
        <p>选择：<code>${escapeHtml(trace.selected_candidate_id)}</code></p>
        <small>理由：${escapeHtml(rationale)}（${escapeHtml(trace.rationale_code)}）</small>
        <small>候选：${escapeHtml(trace.candidate_ids.join(" · "))}</small>
        <small>运行：${escapeHtml(runtime?.transport || trace.transport)} · ${escapeHtml(elapsedText)} · ${escapeHtml(tokenText)}</small>
        ${trace.fallback_reason ? `<small>回退原因：${escapeHtml(trace.fallback_reason)}</small>` : ""}
      </article>`;
    }).join("")
    : "<p>本次没有调用传感器 Planner。</p>";

  elements.publicSensorToolTrace.innerHTML = result.tool_trace.length
    ? result.tool_trace.map((execution) => `<article data-status="${escapeHtml(execution.status)}">
        <header><b>${execution.sequence}. ${escapeHtml(PUBLIC_SENSOR_TOOL_LABELS[execution.tool_id] || execution.tool_id)}</b><span>${escapeHtml(execution.status)}</span></header>
        <small>evidence_ids：${escapeHtml(execution.evidence_ids.join(" · ") || "无")}</small>
        <small>result_codes：${escapeHtml(execution.result_codes.join(" · ") || "无")}</small>
      </article>`).join("")
    : "<p>本次在读取公开传感器来源前由服务端安全停止。</p>";

  elements.publicSensorEvidence.innerHTML = result.evidence.length
    ? result.evidence.map(renderPublicSensorEvidence).join("")
    : "<p>本次没有读取公开传感器来源，也没有形成证据快照。</p>";
  renderPublicSensorComparison(result.comparison);
  renderPublicSensorVisual(result.sensor, result.protocol_id, result.comparison);
  elements.publicSensorFindings.innerHTML = report.supported_findings.length
    ? report.supported_findings.map((finding) => `<article><b>${escapeHtml(finding.text)}</b><small>evidence_ids：${escapeHtml(finding.evidence_ids.join(" · "))}</small></article>`).join("")
    : "<p>没有形成可由证据直接支持的传感器发现。</p>";
  elements.publicSensorSources.innerHTML = report.sources.length
    ? report.sources.map(renderPublicSensorSource).join("")
    : "<p>本次没有访问公开传感器来源。</p>";
  elements.publicSensorUncertainties.innerHTML = report.uncertainties.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.publicSensorForbiddenClaims.innerHTML = report.forbidden_claims.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.publicSensorNextLive.hidden = !report.next_live_measurement;
  elements.publicSensorNextLive.querySelector("p").textContent = report.next_live_measurement || "";
}

function renderPublicSensorEvidence(evidence) {
  const analysis = evidence.analysis;
  const metrics = analysis.metrics.map((metric) => `<li><code>${escapeHtml(metric.key)}</code><b>${escapeHtml(formatMetricValue(metric.value))} ${escapeHtml(metric.unit)}</b></li>`).join("");
  return `<article class="public-light-evidence-card" data-data-class="${escapeHtml(evidence.data_class)}">
    <header><div><span>${escapeHtml(publicReplayDataClassLabel(evidence.data_class))}</span><b>${escapeHtml(evidence.evidence_id)}</b></div><strong>Gate C = 0</strong></header>
    <p>${escapeHtml(evidence.condition_label)}</p>
    <small>${escapeHtml(evidence.license_spdx)} · ${escapeHtml(evidence.device_scope)}</small>
    <small>${escapeHtml(analysis.analyzer_id)}@${escapeHtml(analysis.analyzer_version)} · ${analysis.sample_count} samples · ${escapeHtml(confidenceText(analysis.confidence))}</small>
    <ul class="public-light-facts">${metrics}</ul>
    <details><summary>处理披露</summary><ul>${evidence.processing_disclosures.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>
    <details open><summary>结论边界</summary><ul>${evidence.claim_boundary.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>
  </article>`;
}

function renderPublicSensorComparison(comparison) {
  elements.publicSensorComparison.hidden = !comparison;
  if (!comparison) return;
  const metrics = comparison.metrics.map((metric) => `<li><code>${escapeHtml(metric.key)}</code><b>${escapeHtml(formatMetricValue(metric.value))} ${escapeHtml(metric.unit)}</b></li>`).join("");
  elements.publicSensorComparison.querySelector("div").innerHTML = `<article>
    <header><b>${escapeHtml(comparison.comparison_id)}</b><span>${comparison.quality_passed ? "QUALITY PASS" : "QUALITY NOT PASSED"}</span></header>
    <p>${escapeHtml(comparison.interpretation)}</p>
    <small>result_codes：${escapeHtml(comparison.result_codes.join(" · "))} · Gate C 计入 ${comparison.gate_c_eligible ? "YES" : "NO"}</small>
    <ul>${metrics}</ul>
  </article>`;
}

function renderPublicSensorVisual(sensor, protocolId, comparison) {
  elements.publicSensorVisual.hidden = true;
  elements.publicSensorVisual.querySelector("div").innerHTML = "";
  const captions = {
    accelerometer: "图中只展示服务端确定性派生指标，不从加速度积分位移，也不诊断当前设备原因。",
    location: "图中只展示隐私变换后的相对几何指标，不显示或恢复真实坐标。",
    microphone: "图中只展示服务端确定性派生指标，不展示或重建原始音频。",
  };
  elements.publicSensorVisual.querySelector("p").textContent = captions[sensor]
    || "图中只展示服务端确定性派生指标，不替代当前手机的真机证据。";
  if (!comparison) return;
  const metrics = Object.fromEntries(comparison.metrics.map((item) => [item.key, item.value]));
  if (sensor === "accelerometer") {
    if (protocolId === "walking-cadence-public-exploration.v1") {
      const cadenceBars = [1, 2, 3].map((index) => [
        `重复 ${index}`,
        metrics[`repeat_${index}_cadence_hz`],
      ]).filter((item) => Number.isFinite(item[1]));
      if (!cadenceBars.length) return;
      const maximum = Math.max(...cadenceBars.map((item) => item[1]), 1);
      const barSvg = cadenceBars.map(([label, value], index) => {
        const height = Math.max(1, value / maximum * 145);
        const x = 150 + index * 170;
        return `<g data-acceleration-metric="cadence"><rect x="${x}" y="${185 - height}" width="92" height="${height}" rx="9" /><text class="value" x="${x + 46}" y="${Math.max(25, 177 - height)}" text-anchor="middle">${escapeHtml(formatMetricValue(value))} Hz</text><text class="label" x="${x + 46}" y="214" text-anchor="middle">${escapeHtml(label)}</text></g>`;
      }).join("");
      elements.publicSensorVisual.querySelector("div").innerHTML = `<svg viewBox="0 0 720 235" role="img" aria-label="三段公开楼梯步频候选柱状图"><line x1="70" y1="185" x2="680" y2="185" />${barSvg}<text class="unit" x="70" y="20">周期主频候选 · 不代表路面因果或医疗结论</text></svg>`;
      elements.publicSensorVisual.hidden = false;
      return;
    }
    if (protocolId === "elevator-motion-public-exploration.v1") {
      const accelerationStart = metrics.mean_acceleration_start_s;
      const accelerationDuration = metrics.mean_acceleration_duration_s;
      const cruiseDuration = metrics.mean_cruise_duration_s;
      const decelerationDuration = metrics.mean_deceleration_duration_s;
      if (![accelerationStart, accelerationDuration, cruiseDuration, decelerationDuration].every(Number.isFinite)) return;
      const total = Math.max(accelerationDuration + cruiseDuration + decelerationDuration, 1);
      const scale = 570 / total;
      const x = 80;
      const accelWidth = accelerationDuration * scale;
      const cruiseWidth = cruiseDuration * scale;
      const decelWidth = decelerationDuration * scale;
      elements.publicSensorVisual.querySelector("div").innerHTML = `<svg viewBox="0 0 720 235" role="img" aria-label="公开电梯正加速稳定减速阶段时间线"><text class="unit" x="70" y="28">平均阶段时间线 · 记录起点后约 ${escapeHtml(formatMetricValue(accelerationStart))} s 进入加速</text><g data-acceleration-metric="elevator-phases"><rect x="${x}" y="78" width="${accelWidth}" height="64" rx="10" /><rect x="${x + accelWidth}" y="78" width="${cruiseWidth}" height="64" rx="10" /><rect x="${x + accelWidth + cruiseWidth}" y="78" width="${decelWidth}" height="64" rx="10" /><text x="${x + accelWidth / 2}" y="113" text-anchor="middle">加速</text><text x="${x + accelWidth + cruiseWidth / 2}" y="113" text-anchor="middle">稳定中段</text><text x="${x + accelWidth + cruiseWidth + decelWidth / 2}" y="113" text-anchor="middle">减速</text></g><text class="label" x="360" y="188" text-anchor="middle">仅做阶段分割，不从加速度积分楼层或位移</text></svg>`;
      elements.publicSensorVisual.hidden = false;
      return;
    }
    const responseBars = [
      ["静止 RMS", metrics.stationary_rms_m_s2, "stationary"],
      ["手持 RMS", metrics.handheld_rms_m_s2, "motion"],
    ].filter((item) => Number.isFinite(item[1]));
    if (!responseBars.length) return;
    const maximum = Math.max(...responseBars.map((item) => item[1]), 0.001);
    const responseSvg = responseBars.map(([label, value, group], index) => {
      const height = Math.max(2, value / maximum * 140);
      const x = 205 + index * 220;
      return `<g data-acceleration-metric="${escapeHtml(group)}"><rect x="${x}" y="${182 - height}" width="108" height="${height}" rx="9" /><text class="value" x="${x + 54}" y="${Math.max(25, 174 - height)}" text-anchor="middle">${escapeHtml(formatMetricValue(value))}</text><text class="label" x="${x + 54}" y="212" text-anchor="middle">${escapeHtml(label)}</text></g>`;
    }).join("");
    elements.publicSensorVisual.querySelector("div").innerHTML = `<svg viewBox="0 0 720 235" role="img" aria-label="公开静止与手持加速度 RMS 对照图">${responseSvg}<text class="unit" x="70" y="20">m/s² · 只验证运动响应，不诊断设备原因</text></svg>`;
    elements.publicSensorVisual.hidden = false;
    return;
  }
  if (sensor === "location") {
    const routeBars = [
      ["路线 A 长度", metrics.route_a_distance_m, "route-a"],
      ["路线 B 长度", metrics.route_b_distance_m, "route-b"],
      ["最近点中位", metrics.symmetric_median_nearest_distance_m, "shape"],
      ["最近点 P95", metrics.symmetric_p95_nearest_distance_m, "shape"],
      ["相对终点差", metrics.relative_endpoint_separation_m, "endpoint"],
    ].filter((item) => Number.isFinite(item[1]));
    if (routeBars.length < 2) return;
    const maximum = Math.max(...routeBars.map((item) => item[1]), 1);
    const plotTop = 24;
    const plotBottom = 190;
    const plotHeight = plotBottom - plotTop;
    const barWidth = 76;
    const gap = 43;
    const startX = 72;
    const grid = [0, 0.5, 1].map((ratio) => {
      const y = plotBottom - ratio * plotHeight;
      return `<line x1="48" y1="${y}" x2="700" y2="${y}" />
        <text x="40" y="${y + 4}" text-anchor="end">${escapeHtml(formatMetricValue(maximum * ratio))}</text>`;
    }).join("");
    const barSvg = routeBars.map(([label, value, group], index) => {
      const height = Math.max(1, (value / maximum) * plotHeight);
      const x = startX + index * (barWidth + gap);
      const y = plotBottom - height;
      return `<g data-location-metric="${escapeHtml(group)}">
        <rect x="${x}" y="${y}" width="${barWidth}" height="${height}" rx="8" />
        <text class="value" x="${x + barWidth / 2}" y="${Math.max(16, y - 7)}" text-anchor="middle">${escapeHtml(formatMetricValue(value))}</text>
        <text class="label" x="${x + barWidth / 2}" y="214" text-anchor="middle">${escapeHtml(label)}</text>
      </g>`;
    }).join("");
    elements.publicSensorVisual.querySelector("div").innerHTML = `<svg viewBox="0 0 720 235" role="img" aria-label="隐私变换后的重复路线相对指标柱状图">
      <g class="grid">${grid}</g>
      ${barSvg}
      <text class="unit" x="48" y="14">m · 相对指标，不含真实坐标</text>
    </svg>`;
    elements.publicSensorVisual.hidden = false;
    return;
  }
  if (sensor !== "microphone") return;
  const bars = [
    ["前段平均", metrics.early_mean_relative_level_db, "early"],
    ["后段平均", metrics.late_mean_relative_level_db, "late"],
    ["前段峰值", metrics.early_peak_relative_level_db, "early"],
    ["后段峰值", metrics.late_peak_relative_level_db, "late"],
    ["前段范围", metrics.early_relative_level_span_db, "early"],
    ["后段范围", metrics.late_relative_level_span_db, "late"],
  ].filter((item) => Number.isFinite(item[1]));
  if (bars.length < 2) return;
  const maximum = Math.max(...bars.map((item) => item[1]), 1);
  const plotTop = 24;
  const plotBottom = 190;
  const plotHeight = plotBottom - plotTop;
  const barWidth = 64;
  const gap = 38;
  const startX = 75;
  const grid = [0, 0.5, 1].map((ratio) => {
    const y = plotBottom - ratio * plotHeight;
    return `<line x1="48" y1="${y}" x2="700" y2="${y}" />
      <text x="40" y="${y + 4}" text-anchor="end">${escapeHtml(formatMetricValue(maximum * ratio))}</text>`;
  }).join("");
  const barSvg = bars.map(([label, value, group], index) => {
    const height = Math.max(1, (value / maximum) * plotHeight);
    const x = startX + index * (barWidth + gap);
    const y = plotBottom - height;
    return `<g data-window="${escapeHtml(group)}">
      <rect x="${x}" y="${y}" width="${barWidth}" height="${height}" rx="8" />
      <text class="value" x="${x + barWidth / 2}" y="${Math.max(16, y - 7)}" text-anchor="middle">${escapeHtml(formatMetricValue(value))}</text>
      <text class="label" x="${x + barWidth / 2}" y="214" text-anchor="middle">${escapeHtml(label)}</text>
    </g>`;
  }).join("");
  elements.publicSensorVisual.querySelector("div").innerHTML = `<svg viewBox="0 0 720 235" role="img" aria-label="NoiseCapture 前后窗口派生相对级别指标柱状图">
    <g class="grid">${grid}</g>
    ${barSvg}
    <text class="unit" x="48" y="14">dB_relative</text>
  </svg>`;
  elements.publicSensorVisual.hidden = false;
}

function renderPublicSensorSource(source) {
  const sourceUrl = safeExternalUrl(source.source_url);
  const doiUrl = safeExternalUrl(`https://doi.org/${source.doi}`);
  return `<article class="public-light-source-card">
    <b>${escapeHtml(publicReplayDataClassLabel(source.data_class))} · Gate C 计入 ${source.gate_c_eligible ? "YES" : "NO"}</b>
    <span>${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(source.source_title)} ↗</a>` : escapeHtml(source.source_title)}</span>
    <small>${escapeHtml(source.license_spdx)} · ${escapeHtml(source.device_scope)}${doiUrl ? ` · <a href="${escapeHtml(doiUrl)}" target="_blank" rel="noreferrer">DOI ${escapeHtml(source.doi)}</a>` : ""}</small>
  </article>`;
}

async function loadPublicReplays() {
  elements.publicReplayStatus.dataset.state = "loading";
  elements.publicReplayStatus.textContent = "正在读取带来源证明的公开数据目录…";
  try {
    const response = await fetch("/api/v2/public-replays");
    if (!response.ok) throw new Error(await readApiError(response));
    const catalog = await response.json();
    if (!Array.isArray(catalog)) throw new Error("公开数据目录格式无效。");
    state.publicReplays = catalog;
    elements.publicReplayDataset.innerHTML = catalog.length
      ? `<option value="">选择一个来源数据包</option>${catalog.map((item) => `<option value="${escapeHtml(item.dataset_id)}">${escapeHtml(item.title)} · ${escapeHtml(publicReplayDataClassLabel(item.data_class))}</option>`).join("")}`
      : '<option value="">当前没有可导入的公开数据包</option>';
    elements.publicReplayDataset.disabled = catalog.length === 0;
    elements.publicReplayStatus.dataset.state = catalog.length ? "idle" : "error";
    elements.publicReplayStatus.textContent = catalog.length
      ? `已校验 ${catalog.length} 个公开数据包；请选择数据包并先阅读结论边界。`
      : "当前没有通过来源校验的公开数据包。";
    renderPublicReplaySelection();
  } catch (error) {
    state.publicReplays = [];
    elements.publicReplayDataset.innerHTML = '<option value="">公开数据目录读取失败</option>';
    elements.publicReplayDataset.disabled = true;
    elements.publicReplayRecording.innerHTML = '<option value="">暂时无法选择记录</option>';
    elements.publicReplayRecording.disabled = true;
    elements.publicReplayDetails.hidden = true;
    elements.publicReplayStatus.dataset.state = "error";
    elements.publicReplayStatus.textContent = `公开数据目录读取失败：${error.message}`;
    updatePublicReplayAvailability();
  }
}

function renderPublicReplaySelection() {
  const dataset = currentPublicReplayDataset();
  const recordings = dataset?.recordings || [];
  elements.publicReplayRecording.innerHTML = dataset
    ? `<option value="">选择一条记录</option>${recordings.map((recording) => {
      const independence = recording.independent_measurement ? "独立测量" : "非独立派生";
      return `<option value="${escapeHtml(recording.recording_id)}">${escapeHtml(recording.label)} · ${recording.sample_count} 点 · ${escapeHtml(independence)}</option>`;
    }).join("")}`
    : '<option value="">请先选择数据包</option>';
  elements.publicReplayRecording.disabled = !dataset || recordings.length === 0 || state.busy;
  elements.publicReplayDetails.hidden = !dataset;
  if (!dataset) {
    elements.publicReplayDetails.innerHTML = "";
    updatePublicReplayAvailability();
    return;
  }

  const sourceUrl = safeExternalUrl(dataset.source_url);
  const doiUrl = dataset.doi
    ? safeExternalUrl(String(dataset.doi).startsWith("https://") ? dataset.doi : `https://doi.org/${dataset.doi}`)
    : "";
  const replayStatus = dataset.public_replay_status || dataset.status;
  const agentValueStatus = dataset.agent_value_status || dataset.agent_value;
  const license = dataset.license_spdx || dataset.license || "未声明";
  const claimBoundary = Array.isArray(dataset.claim_boundary) ? dataset.claim_boundary : [];
  const importAllowed = publicReplayCanImport(dataset);
  const privacyRisks = Array.isArray(dataset.privacy_risk_categories) ? dataset.privacy_risk_categories : [];
  elements.publicReplayDetails.dataset.dataClass = dataset.data_class;
  elements.publicReplayDetails.innerHTML = `
    <div class="public-replay-badges">
      <b>${escapeHtml(publicReplayDataClassLabel(dataset.data_class))}</b>
      <span>${escapeHtml(publicReplayStatusLabel(replayStatus))}</span>
      <span>Agent Value：${escapeHtml(publicReplayStatusLabel(agentValueStatus))}</span>
      <span>记录 ${Number(dataset.recording_count) || recordings.length}</span>
      <span>${importAllowed ? "允许账号导入" : "仅无持久化本地回放"}</span>
    </div>
    <h4>${escapeHtml(dataset.title)}</h4>
    <p>${escapeHtml(dataset.description)}</p>
    <dl class="public-replay-source">
      <div><dt>来源</dt><dd>${sourceUrl ? `<a href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(dataset.source_title)} ↗</a>` : escapeHtml(dataset.source_title)}</dd></div>
      <div><dt>许可</dt><dd>${escapeHtml(license)}</dd></div>
      <div><dt>DOI</dt><dd>${doiUrl ? `<a href="${escapeHtml(doiUrl)}" target="_blank" rel="noreferrer">${escapeHtml(dataset.doi)} ↗</a>` : escapeHtml(dataset.doi || "无")}</dd></div>
    </dl>
    <div class="public-replay-claims">
      <b>结论边界</b>
      <ul>${claimBoundary.length ? claimBoundary.map((entry) => `<li>${escapeHtml(entry)}</li>`).join("") : "<li>该数据包没有提供可扩大的结论边界。</li>"}</ul>
    </div>
    <div class="public-replay-import-policy" data-import-allowed="${importAllowed}">
      <b>${importAllowed ? "账号导入已获来源审查允许" : "账号导入已禁用"}</b>
      <span>${importAllowed ? "可选择一条记录导入当前账号；仍不计作现场真机证据。" : "该来源只能通过上方闭环在本机无持久化回放，不能写入账号记录。"}</span>
      ${privacyRisks.length ? `<small>隐私风险：${escapeHtml(privacyRisks.join(" · "))} · 部署范围：${escapeHtml(dataset.deployment_scope || "local_only")}</small>` : ""}
    </div>
    <div class="public-replay-readiness" data-ready="${Boolean(dataset.public_replay_ready)}">
      <span>PUBLIC REPLAY READY</span><b>${dataset.public_replay_ready ? "YES" : "NO"}</b>
      <span>AGENT READY</span><b>${dataset.agent_ready ? "YES" : "NO"}</b>
    </div>`;
  elements.publicReplayStatus.dataset.state = "idle";
  elements.publicReplayStatus.textContent = importAllowed
    ? "请选择一条记录；导入只会新增当前账号的 v2 记录，不会成为 Light 距离协议证据。"
    : "该数据包禁止账号导入；只能使用上方公开 Light 闭环进行无持久化本地回放。";
  updatePublicReplayAvailability();
}

function updatePublicReplayAvailability() {
  const dataset = currentPublicReplayDataset();
  const hasRecording = Boolean(elements.publicReplayRecording.value);
  elements.publicReplayRecording.disabled = !dataset || !(dataset.recordings || []).length || state.busy;
  elements.publicReplayImportButton.disabled = !dataset || !hasRecording || !publicReplayCanImport(dataset) || state.busy;
  if (!state.busy) {
    const label = elements.publicReplayImportButton.querySelector("span");
    if (label) label.textContent = dataset && !publicReplayCanImport(dataset)
      ? "该数据包禁止账号导入"
      : "导入当前账号并查看分析";
  }
}

async function importPublicReplayRecording() {
  const dataset = currentPublicReplayDataset();
  const recordingId = elements.publicReplayRecording.value;
  if (!dataset || !recordingId || state.busy) return;
  if (!publicReplayCanImport(dataset)) {
    elements.publicReplayStatus.dataset.state = "error";
    elements.publicReplayStatus.textContent = "该来源不允许账号导入，只能通过公开 Light 闭环进行无持久化本地回放。";
    return;
  }
  let importedSessionId = "";
  setBusy(true, elements.publicReplayImportButton, "正在校验并导入…");
  updatePublicReplayAvailability();
  elements.publicReplayStatus.dataset.state = "loading";
  elements.publicReplayStatus.textContent = "正在校验来源文件并运行已冻结的确定性分析器…";
  try {
    const response = await fetch(`/api/v2/public-replays/${encodeURIComponent(dataset.dataset_id)}/recordings/${encodeURIComponent(recordingId)}/import`, {
      method: "POST",
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const created = await response.json();
    importedSessionId = created.session_id;
    renderSensorLabAnalysis(
      created.sensor,
      created.analysis,
      `公开回放 · ${created.label}`,
    );
    await refreshSensorRecordings();
    if (state.investigation) renderInvestigation();
    elements.publicReplayStatus.dataset.state = "ready";
    elements.publicReplayStatus.textContent = `已导入当前账号：${created.session_id} · ${publicReplayDataClassLabel(dataset.data_class)}。结果显示在上方确定性分析区，仍不计作现场真机证据。`;
    showToast("公开回放已导入，并完成确定性分析");
  } catch (error) {
    elements.publicReplayStatus.dataset.state = "error";
    elements.publicReplayStatus.textContent = importedSessionId
      ? `记录 ${importedSessionId} 已导入，但界面记录列表刷新失败：${error.message}。请刷新页面，避免重复导入。`
      : `公开回放导入失败：${error.message}`;
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.publicReplayImportButton, "导入当前账号并查看分析");
    updatePublicReplayAvailability();
  }
}

async function loadSensorCapabilities() {
  try {
    const response = await fetch("/api/v2/sensors/capabilities");
    if (!response.ok) throw new Error(await readApiError(response));
    state.sensorCapabilities = await response.json();
    renderSensorCapabilities();
  } catch (error) {
    elements.sensorCapabilityGrid.innerHTML = `<div class="sensor-capability-empty">分析器能力读取失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderSensorCapabilities() {
  const maturityLabels = {
    detectable: "仅可识别",
    capture_ready: "可采集",
    analysis_ready: "确定性分析可用",
    agent_ready: "Agent 闭环可用",
    release_candidate: "发布候选",
  };
  elements.sensorCapabilityGrid.innerHTML = state.sensorCapabilities.map((item) => `
    <article class="sensor-capability-card" data-maturity="${escapeHtml(item.maturity)}">
      <span>${escapeHtml(item.analyzer_id || "尚无通用分析器")}</span>
      <b>${escapeHtml(SENSOR_LABELS[item.sensor] || item.sensor)}</b>
      <i>${escapeHtml(maturityLabels[item.maturity] || item.maturity)}</i>
    </article>`).join("");

  const previous = elements.sensorLabSensor.value;
  const options = state.sensorCapabilities
    .filter((item) => item.maturity === "analysis_ready")
    .map((item) => `<option value="${escapeHtml(item.sensor)}">${escapeHtml(SENSOR_LABELS[item.sensor] || item.sensor)}</option>`)
    .join("");
  elements.sensorLabSensor.innerHTML = `<option value="">请选择当前 phyphox 实验对应的传感器</option>${options}`;
  if (state.sensorCapabilities.some((item) => item.sensor === previous && item.maturity === "analysis_ready")) {
    elements.sensorLabSensor.value = previous;
  }
  updateSensorLabAvailability();
}

function updateSensorLabAvailability() {
  const sensor = elements.sensorLabSensor.value;
  const profile = state.phyphoxProbe?.sensor_profiles?.[sensor];
  const needsPrivacy = sensor === "microphone" || sensor === "location";
  elements.sensorLabPrivacy.closest("label").hidden = !needsPrivacy;
  const privacyReady = !needsPrivacy || elements.sensorLabPrivacy.checked;
  const ready = Boolean(sensor && state.savedDevice && state.phyphoxProbe && profile && privacyReady && !state.busy);
  elements.sensorLabCaptureButton.disabled = !ready;
  if (!sensor) {
    elements.sensorLabStatus.dataset.state = "idle";
    elements.sensorLabStatus.textContent = "请选择传感器；列表只显示已通过离线确定性分析门禁的能力。";
  } else if (!state.savedDevice || !state.phyphoxProbe) {
    elements.sensorLabStatus.dataset.state = "error";
    elements.sensorLabStatus.textContent = "默认手机尚未连接，请先到“设备与设置”完成检测。";
  } else if (!profile) {
    elements.sensorLabStatus.dataset.state = "error";
    elements.sensorLabStatus.textContent = `当前 phyphox 实验没有可验证的${SENSOR_LABELS[sensor] || sensor}输入映射，请在手机切换实验后重新检测。`;
  } else if (!privacyReady) {
    elements.sensorLabStatus.dataset.state = "error";
    elements.sensorLabStatus.textContent = "该数据可能涉及声音或位置隐私，请先阅读并确认隐私提示。";
  } else {
    const channels = Object.entries(profile.channel_buffers).map(([role, buffer]) => `${role} ← ${buffer}`).join(" · ");
    elements.sensorLabStatus.dataset.state = "ready";
    elements.sensorLabStatus.textContent = `已验证 ${SENSOR_LABELS[sensor] || sensor} Profile：${channels}`;
  }
}

function renderSensorLabAnalysis(sensor, analysis, title) {
  elements.sensorLabResultTitle.textContent = title || `${SENSOR_LABELS[sensor] || sensor} · ${analysis.sample_count} 点`;
  elements.sensorLabConfidence.textContent = `${String(analysis.confidence).toUpperCase()} CONFIDENCE`;
  elements.sensorLabMetrics.innerHTML = analysis.metrics.map((metric) => `
    <div><span>${escapeHtml(metric.label)}</span><b>${escapeHtml(formatMetricValue(metric.value))} ${escapeHtml(metric.unit)}</b></div>`).join("");
  elements.sensorLabWarnings.innerHTML = analysis.warnings.length
    ? analysis.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")
    : "<li>当前离线质量规则未发现额外警告。</li>";
  elements.sensorLabResult.hidden = false;
}

async function captureSensorLabRecording() {
  if (state.busy || elements.sensorLabCaptureButton.disabled) return;
  const sensor = elements.sensorLabSensor.value;
  const duration = Number(elements.sensorLabDuration.value);
  const label = elements.sensorLabLabel.value.trim();
  if (!Number.isFinite(duration) || duration < 1 || duration > 300) {
    showToast("采集时长必须在 1 到 300 秒之间。", true);
    return;
  }
  if (!label) {
    showToast("请填写记录名称。", true);
    return;
  }
  if ((sensor.sensor === "microphone" || sensor.sensor === "location") && !elements.mobilePrivacyCheckbox.checked) {
    showToast("本次传感器涉及隐私敏感派生量，请先确认本机分析边界。", true);
    return;
  }
  setBusy(true, elements.sensorLabCaptureButton, "正在采集…");
  updateSensorLabAvailability();
  elements.sensorLabResult.hidden = true;
  try {
    const response = await fetch("/api/v2/phyphox/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: state.savedDevice.base_url,
        sensor,
        duration_s: duration,
        label,
        privacy_acknowledged: elements.sensorLabPrivacy.checked,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    renderSensorLabAnalysis(sensor, data.session.analysis);
    elements.sensorLabStatus.dataset.state = "ready";
    elements.sensorLabStatus.textContent = `v2 记录 ${data.session.session_id} 已保存；它尚未绑定到 Agent 诊断证据。`;
    showToast("传感器记录已完成确定性分析");
  } catch (error) {
    elements.sensorLabStatus.dataset.state = "error";
    elements.sensorLabStatus.textContent = error.message;
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.sensorLabCaptureButton, "从当前 phyphox 实验采集");
    updateSensorLabAvailability();
  }
}

function formatMetricValue(value) {
  if (!Number.isFinite(value)) return String(value);
  const magnitude = Math.abs(value);
  if ((magnitude > 0 && magnitude < 0.001) || magnitude >= 100000) return value.toExponential(3);
  return Number(value.toFixed(4)).toString();
}

function renderExplorations() {
  if (!elements.explorationGrid) return;
  const matches = new Set(state.phyphoxProbe?.exploration_matches || []);
  const visible = state.explorations.filter((item) => {
    if (state.explorationFilter === "ready") {
      return item.action_kind === "diagnostic_agent" || item.action_kind === "bounded_agent";
    }
    if (state.explorationFilter === "phone") return matches.has(item.exploration_id);
    return true;
  });
  elements.explorationEmpty.hidden = visible.length > 0;
  elements.explorationGrid.innerHTML = visible.map((item) => {
    const matched = matches.has(item.exploration_id);
    const readiness = item.action_kind === "bounded_agent"
      ? "受限 Agent Beta · 待真机验证"
      : item.readiness === "ready"
        ? "端到端可运行"
        : item.readiness === "analysis_ready"
        ? "确定性分析可用"
        : item.readiness === "capability_detectable" ? "可识别通道" : "规划接入";
    let action;
    if (item.action_kind === "bounded_agent") {
      action = `<button class="exploration-action" type="button" data-start-exploration="${escapeHtml(item.exploration_id)}">启动受限 Agent 实验 Beta →</button>`;
    } else if (item.action_kind === "diagnostic_agent") {
      action = `<button class="exploration-action" type="button" data-start-exploration="${escapeHtml(item.exploration_id)}">使用诊断 Agent 开始 →</button>`;
    } else if (item.action_kind === "sensor_analysis") {
      action = `<button class="exploration-action" type="button" data-start-exploration="${escapeHtml(item.exploration_id)}">打开确定性分析实验台 →</button>`;
    } else {
      action = `<button class="exploration-action capability" type="button" data-start-exploration="${escapeHtml(item.exploration_id)}">检验当前设备能力 →</button>`;
    }
    return `
      <article class="exploration-card ${escapeHtml(item.readiness)} ${item.action_kind === "bounded_agent" ? "executable-beta" : ""} ${matched ? "phone-match" : ""}">
        <header>
          <span class="readiness-dot"></span>
          <small>${escapeHtml(readiness)}${matched ? " · 当前实验匹配" : ""}</small>
          <b>${escapeHtml(SENSOR_LABELS[item.primary_sensor] || item.primary_sensor)}</b>
        </header>
        <div class="exploration-meta"><span>${escapeHtml(item.category)}</span><span>${item.duration_minutes} MIN</span><span>${escapeHtml(item.difficulty)}</span></div>
        <h3>${escapeHtml(item.title)}</h3>
        <p class="exploration-question">${escapeHtml(item.question)}</p>
        <details>
          <summary>查看实验设计与产出</summary>
          <ol>${item.protocol.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol>
          <p><b>预期信号：</b>${escapeHtml(item.expected_signal)}</p>
          <p><b>最终产出：</b>${escapeHtml(item.output_value)}</p>
          ${item.safety_notes.length ? `<p class="safety-note"><b>安全：</b>${escapeHtml(item.safety_notes.join(" "))}</p>` : ""}
          ${item.privacy_notes.length ? `<p class="privacy-note"><b>隐私：</b>${escapeHtml(item.privacy_notes.join(" "))}</p>` : ""}
          <p class="readiness-note">${escapeHtml(item.readiness_note)}</p>
        </details>
        ${action}
      </article>`;
  }).join("");
  elements.explorationGrid.querySelectorAll("[data-start-exploration]").forEach((button) => {
    button.addEventListener("click", () => startExploration(button.dataset.startExploration));
  });
  renderExplorationCapability();
}

function renderExplorationCapability() {
  const probe = state.phyphoxProbe;
  if (!probe) {
    elements.explorationCapabilityOverview.innerHTML = "<span>当前 phyphox 实验</span><b>尚未检测</b><small>在“设备与设置”中连接手机后，这里会显示当前打开实验的输入能力。</small>";
    updateSensorLabAvailability();
    return;
  }
  const sensors = probe.detected_sensors.length
    ? probe.detected_sensors.map((item) => SENSOR_LABELS[item] || item).join(" · ")
    : "未识别传感器类型";
  const count = probe.exploration_matches?.length || 0;
  elements.explorationCapabilityOverview.innerHTML = `
    <span>当前 phyphox 实验 · ${escapeHtml(probe.experiment_title)}</span>
    <b>${escapeHtml(sensors)}</b>
    <small>匹配 ${count} 个探索模板。这里只代表当前打开的实验，不代表手机拥有的全部传感器。</small>`;
  updateSensorLabAvailability();
}

async function startExploration(explorationId) {
  const item = state.explorations.find((entry) => entry.exploration_id === explorationId);
  if (!item || state.busy) return;
  state.pendingExploration = null;
  elements.explorationSetupPanel.hidden = true;
  hidePublicRunners();
  if (item.action_kind === "diagnostic_agent") {
    startDiagnosticExploration(item);
    return;
  }
  if (item.action_kind === "sensor_analysis") {
    openSensorAnalysisExploration(item);
    return;
  }
  if (item.action_kind === "capability_check") {
    await runExplorationCapabilityCheck(item);
    return;
  }
  if (item.action_kind === "bounded_agent") {
    openExplorationSetup(item);
    return;
  }
  openSensorAnalysisExploration(item);
}

function openExplorationSetup(exploration) {
  const isLightDistance = exploration.executable_protocol_id === "light-distance-law.v1";
  state.pendingExploration = exploration;
  elements.explorationSetupTitle.textContent = exploration.title;
  elements.explorationSetupQuestion.textContent = exploration.question;
  elements.explorationSetupPhoneTitle.textContent = isLightDistance
    ? "连接 phyphox 执行 Light 距离协议"
    : `连接 phyphox 采集${SENSOR_LABELS[exploration.primary_sensor] || exploration.primary_sensor}数据`;
  elements.explorationSetupPhoneDescription.textContent = isLightDistance
    ? "创建带背景、距离、重复、质量门与终止判断的真机实验。"
    : "打开对应传感器的真机采集区；当前先保存单条记录并运行确定性分析，不把它伪装成已完成的专用 Agent 循环。";
  elements.explorationDistanceConstraint.hidden = !isLightDistance;
  elements.explorationSimulationQuestion.textContent = exploration.simulation_question || exploration.question;
  elements.explorationSimulationScope.textContent = exploration.simulation_scope_note || "公开回放不替代当前手机的现场证据。";
  elements.explorationSetupStartButton.textContent = isLightDistance ? "创建真机实验" : "进入真机实验区";
  elements.explorationMaxDistance.value = "";
  elements.explorationSetupPanel.hidden = false;
  hidePublicRunners();
  elements.explorationSetupPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeExplorationSetup() {
  state.pendingExploration = null;
  elements.explorationSetupPanel.hidden = true;
  elements.explorationCatalogPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showPublicRunner(kind) {
  elements.publicReplayLab.hidden = false;
  elements.publicLightRunner.hidden = kind !== "light";
  elements.publicPressureRunner.hidden = kind !== "pressure";
  elements.publicSensorRunner.hidden = kind !== "sensor";
}

function hidePublicRunners() {
  elements.publicReplayLab.hidden = true;
  elements.publicLightRunner.hidden = true;
  elements.publicPressureRunner.hidden = true;
  elements.publicSensorRunner.hidden = true;
}

function openPendingPublicExploration() {
  const exploration = state.pendingExploration;
  if (!exploration) return;
  elements.explorationSetupPanel.hidden = true;
  state.pendingExploration = null;
  if (exploration.primary_sensor === "light") {
    elements.publicLightQuestion.value = exploration.simulation_question || exploration.question;
    elements.publicLightQueryLux.value = "";
    elements.publicLightPrivacy.checked = false;
    elements.publicLightResult.hidden = true;
    elements.publicLightStatus.dataset.state = "ready";
    elements.publicLightStatus.textContent = "已带入与公开数据匹配的模拟问题；它演示 Light Agent 流程，但不验证距离平方反比。";
    showPublicRunner("light");
    updatePublicLightAvailability();
    elements.publicLightRunner.scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
  if (exploration.executable_protocol_id === "pressure-public-exploration.v1") {
    openPublicPressureExploration(exploration);
    return;
  }
  openPublicSensorExploration(exploration);
}

async function startPendingPhoneInvestigation() {
  const exploration = state.pendingExploration;
  if (!exploration || state.busy) return;
  if (exploration.executable_protocol_id !== "light-distance-law.v1") {
    state.pendingExploration = null;
    elements.explorationSetupPanel.hidden = true;
    hidePublicRunners();
    openSensorAnalysisExploration(exploration);
    showToast(`已打开 ${SENSOR_LABELS[exploration.primary_sensor] || exploration.primary_sensor} 真机实验区`);
    return;
  }
  try {
    await createExecutableInvestigation(exploration);
    state.pendingExploration = null;
    elements.explorationSetupPanel.hidden = true;
    showToast("已创建可执行实验，先测量环境光背景");
  } catch (error) {
    showToast(error.message, true);
  }
}

function openPublicPressureExploration(exploration) {
  showPublicRunner("pressure");
  elements.publicPressureQuestion.value = exploration.simulation_question || exploration.question;
  elements.publicPressureStatus.dataset.state = "ready";
  elements.publicPressureStatus.textContent = "已带入 Pressure 探索问题；确认本次本地回放边界后即可运行受限 Agent Beta。";
  updatePublicPressureAvailability();
  elements.publicPressureQuestion.scrollIntoView({ behavior: "smooth", block: "center" });
  showToast("已打开 Pressure 受限 Agent Beta");
}

function openPublicSensorExploration(exploration) {
  const config = publicSensorBetaConfig(
    exploration.primary_sensor,
    exploration.executable_protocol_id,
  );
  if (!config || config.protocolId !== exploration.executable_protocol_id) {
    showToast("该传感器的受限 Agent Beta 协议尚未注册。", true);
    return;
  }
  state.publicSensorActive = exploration.primary_sensor;
  state.publicSensorProtocol = exploration.executable_protocol_id;
  state.publicSensorRun = null;
  state.publicSensorError = "";
  showPublicRunner("sensor");
  elements.publicSensorName.textContent = config.label;
  elements.publicSensorQuestionLabel.textContent = config.label;
  elements.publicSensorIntro.textContent = config.intro;
  elements.publicSensorQuestion.placeholder = config.placeholder;
  elements.publicSensorQuestionHelp.textContent = config.help;
  elements.publicSensorQuestion.value = exploration.simulation_question || exploration.question;
  elements.publicSensorPrivacy.checked = false;
  elements.publicSensorResult.hidden = true;
  elements.publicSensorStatus.dataset.state = "ready";
  elements.publicSensorStatus.textContent = `已带入 ${config.label} 探索问题；确认本次本地公开回放边界后即可运行受限 Agent Beta。`;
  updatePublicSensorAvailability();
  elements.publicSensorRunner.scrollIntoView({ behavior: "smooth", block: "start" });
  showToast(`已打开 ${config.label} 受限 Agent Beta`);
}

function openSensorAnalysisExploration(exploration) {
  elements.explorationAdvancedTools.open = true;
  elements.sensorLabSensor.value = exploration.primary_sensor;
  elements.sensorLabLabel.value = exploration.title;
  updateSensorLabAvailability();
  elements.sensorLabStatus.dataset.state = "ready";
  elements.sensorLabStatus.textContent = `已选择${SENSOR_LABELS[exploration.primary_sensor] || exploration.primary_sensor}分析器。这里先验证单条记录；专用受限 Agent 协议仍在接入，当前不会伪装成闭环。`;
  elements.explorationAdvancedTools.scrollIntoView({ behavior: "smooth", block: "start" });
  showToast("已打开对应确定性分析实验台");
}

async function runExplorationCapabilityCheck(exploration) {
  elements.capabilityCheckPanel.hidden = false;
  elements.capabilityCheckTitle.textContent = `${SENSOR_LABELS[exploration.primary_sensor] || exploration.primary_sensor} · 能力检查`;
  if (!state.savedDevice) {
    elements.capabilityCheckStatus.textContent = "NEEDS DEVICE";
    elements.capabilityCheckSummary.textContent = "当前账号还没有默认 phyphox 手机；能力检查没有联系任何设备。";
    elements.capabilityCheckChannels.textContent = "尚未读取";
    elements.capabilityCheckBlockers.innerHTML = "<li>请先在“设备与设置”保存手机显示的局域网地址。</li>";
    elements.capabilityCheckNextSteps.innerHTML = "<li>打开对应 phyphox 实验。</li><li>启用远程访问并返回此卡重新检查。</li>";
    elements.capabilityCheckPrivacy.textContent = "未发起网络请求，也没有读取设备标识。";
    elements.capabilityCheckPanel.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }

  state.busy = true;
  renderExplorations();
  elements.capabilityCheckStatus.textContent = "CHECKING";
  elements.capabilityCheckSummary.textContent = "正在读取当前 phyphox 实验的 /config 能力…";
  try {
    const response = await fetch(`/api/v2/phyphox/capability-checks/${encodeURIComponent(exploration.primary_sensor)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: state.savedDevice.base_url }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.capabilityCheck = await response.json();
    renderCapabilityCheck(state.capabilityCheck);
    showToast(state.capabilityCheck.status === "not_detected" ? "当前实验未识别到目标能力" : "设备能力检查完成", state.capabilityCheck.status === "not_detected");
  } catch (error) {
    elements.capabilityCheckStatus.textContent = "CHECK FAILED";
    elements.capabilityCheckSummary.textContent = error.message;
    elements.capabilityCheckChannels.textContent = "读取失败";
    elements.capabilityCheckBlockers.innerHTML = `<li>${escapeHtml(error.message)}</li>`;
    elements.capabilityCheckNextSteps.innerHTML = "<li>确认手机与电脑在同一局域网。</li><li>在手机重新启用 phyphox 远程访问后重试。</li>";
    elements.capabilityCheckPrivacy.textContent = "失败响应不会保存为测量证据。";
    showToast(error.message, true);
  } finally {
    state.busy = false;
    renderExplorations();
    elements.capabilityCheckPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function renderCapabilityCheck(result) {
  const statusLabels = {
    profile_ready: "PROFILE READY",
    detected: "DETECTED ONLY",
    not_detected: "NOT DETECTED",
  };
  elements.capabilityCheckStatus.textContent = statusLabels[result.status] || result.status;
  elements.capabilityCheckSummary.textContent = `${result.experiment_title} · ${result.analyzer_id || "尚无分析器"} · ${result.analyzer_maturity}`;
  const profileChannels = result.profile
    ? Object.entries(result.profile.channel_buffers).map(([role, buffer]) => `${role} ← ${buffer}`)
    : [];
  const visibleBuffers = profileChannels.length ? profileChannels : (result.export_buffers.length ? result.export_buffers : result.available_buffers.slice(0, 12));
  elements.capabilityCheckChannels.innerHTML = visibleBuffers.length
    ? visibleBuffers.map((item) => `<code>${escapeHtml(item)}</code>`).join("")
    : "<span>当前实验没有可展示的数值缓冲区。</span>";
  elements.capabilityCheckBlockers.innerHTML = result.blockers.length
    ? result.blockers.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : "<li>当前能力检查没有发现额外阻塞；这仍不等于 Agent 门禁通过。</li>";
  elements.capabilityCheckNextSteps.innerHTML = result.next_steps.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.capabilityCheckPrivacy.textContent = `${result.privacy_statement} · config ${result.config_sha256.slice(0, 12)}…`;
}

function startDiagnosticExploration(exploration) {
  navigateTo("/app/cases/new");
  elements.caseTitleInput.value = exploration.title;
  elements.problemInput.value = exploration.question;
  elements.caseContextInput.value = [
    `来源：PocketLab Real-World Exploration / ${exploration.exploration_id}`,
    `实验建议：${exploration.protocol.join("；")}`,
    `期待产出：${exploration.output_value}`,
  ].join("\n");
  showToast("探索问题已带入诊断 Agent，请确认后建立实验计划");
}

const INVESTIGATION_WORKFLOW_LABELS = {
  diagnostic: "问题诊断 · 找原因并形成行动建议",
  exploration: "科学探索 · 比较条件并解释物理机制",
};

function invalidateInvestigationRoute() {
  state.investigationRoute = null;
  elements.investigationRouteResult.hidden = true;
}

async function routeInvestigationQuestion() {
  if (state.busy) return;
  const question = elements.investigationRouterQuestion.value.trim();
  if (question.length < 5) {
    showToast("请至少用 5 个字描述你想弄明白的问题。", true);
    elements.investigationRouterQuestion.focus();
    return;
  }
  const context = elements.investigationRouterContext.value.trim();
  setBusy(true, elements.routeInvestigationButton, "正在判断…");
  try {
    const response = await fetch("/api/v2/investigations/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const recommendation = await response.json();
    if (
      question !== elements.investigationRouterQuestion.value.trim()
      || context !== elements.investigationRouterContext.value.trim()
    ) return;
    state.investigationRoute = recommendation;
    renderInvestigationRoute();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.routeInvestigationButton, "判断合适的工作流");
  }
}

function renderInvestigationRoute() {
  const route = state.investigationRoute;
  if (!route) return;
  const recommended = route.recommended_workflow;
  const modelRouted = route.decision_source === "model";
  const sourceLabel = modelRouted
    ? `基模判别 · ${compactModelName(route.model_name || "当前模型")}`
    : "安全降级 · 当前基模未完成判别";
  elements.investigationRouteResult.hidden = false;
  elements.investigationRouteResult.dataset.source = route.decision_source;
  elements.investigationRouteBadge.textContent = `${modelRouted ? "MODEL" : "FALLBACK"} · ${recommended === "diagnostic" ? "DIAGNOSE" : "EXPLORE"}`;
  elements.investigationRouteTitle.textContent = INVESTIGATION_WORKFLOW_LABELS[recommended];
  elements.investigationRouteConfidence.textContent = `${sourceLabel} · 置信度 ${confidenceText(route.confidence)}`;
  elements.investigationRouteSummary.textContent = recommended === "diagnostic" ? route.diagnostic_boundary : route.exploration_boundary;
  elements.investigationRouteSensors.innerHTML = route.suggested_sensors.length
    ? route.suggested_sensors.map((sensor) => `<span>${escapeHtml(SENSOR_LABELS[sensor] || sensor)}</span>`).join("")
    : "<span>传感器将在下一步由专业流程选择</span>";
  elements.investigationRouteReasons.innerHTML = route.reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("");
  elements.investigationRoutePrivacy.hidden = !route.sensitive_sensor_notice;
  elements.investigationRoutePrivacy.textContent = route.sensitive_sensor_notice || "";
  elements.startRecommendedWorkflowButton.querySelector("span").textContent = `进入${recommended === "diagnostic" ? "问题诊断" : "科学探索"}`;
  elements.startAlternativeWorkflowButton.querySelector("span").textContent = `改用${route.alternative_workflow === "diagnostic" ? "问题诊断" : "科学探索"}`;
}

function startRoutedWorkflow(workflow) {
  const route = state.investigationRoute;
  if (!route || !["diagnostic", "exploration"].includes(workflow)) return;
  const question = elements.investigationRouterQuestion.value.trim();
  const context = elements.investigationRouterContext.value.trim();
  if (workflow === "diagnostic") {
    navigateTo("/app/cases/new");
    elements.caseTitleInput.value = route.suggested_title;
    elements.problemInput.value = question;
    elements.caseContextInput.value = context;
    elements.problemInput.focus();
    showToast("问题已带入诊断入口；确认后再让 Agent 建立计划");
    return;
  }
  state.generalRoutedContext = context;
  navigateTo("/app/explore/general");
  elements.generalNaturalQuestion.value = question;
  resetGeneralCompilerClarification();
  updateGeneralCompilerAvailability();
  elements.generalNaturalQuestion.focus();
  showToast("问题已带入自由探索；下一步先生成可编辑协议草案");
}

function renderDashboard() {
  if (!state.currentUser) return;
  elements.dashboardCaseCount.textContent = String(state.caseHistory.length);
  elements.dashboardSessionCount.textContent = String(workbenchEvidenceItems().length);
  const device = state.savedDevice;
  elements.dashboardDeviceState.textContent = device ? (state.phyphoxProbe ? "已连接" : "已保存") : "未设置";
  elements.dashboardDeviceDetail.textContent = device
    ? `${device.name} · ${device.base_url}${device.experiment_title ? ` · ${device.experiment_title}` : ""}`
    : "尚未设置默认手机；模拟数据和文件导入仍然可以使用。";
  const structured = state.workSummaries.map((item) => ({
    kind: item.workflow,
    id: item.work_id,
    title: item.title,
    status: workSummaryStatus(item.status),
    updated_at: item.updated_at,
    resumePath: item.resume_path,
    resumable: item.resumable,
    detail: `${item.evidence_count} 项证据${item.sensors.length ? ` · ${item.sensors.map((sensor) => SENSOR_LABELS[sensor] || sensor).join("+")}` : ""} · ${item.next_action}`,
  }));
  const summarizedKeys = new Set(structured.map((item) => `${item.kind}:${item.id}`));
  const diagnosticFallback = state.workSummaries.length ? [] : state.caseHistory.map((item) => ({
    kind: "diagnostic", id: item.case_id, title: item.title, status: caseStatusText(item.status),
    updated_at: item.updated_at, resumePath: `/app/cases/${item.case_id}`, resumable: !item.status.startsWith("completed"), detail: `${item.evidence_count} 项证据`,
  }));
  const recent = [
    ...structured,
    ...diagnosticFallback,
    ...state.explorationHistory.map((item) => ({
      kind: item.record_kind,
      id: item.record_id,
      title: item.title,
      status: explorationHistoryStatus(item),
      updated_at: item.updated_at,
      resumePath: item.record_kind === "general_exploration" ? `/app/explore/general/runs/${item.record_id}` : null,
      resumable: item.resumable,
      detail: `${item.evidence_count} 项证据 · ${explorationHistorySource(item)}`,
    })).filter((item) => !summarizedKeys.has(`${item.kind}:${item.id}`)),
  ].sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at))).slice(0, 5);
  elements.dashboardRecentEmpty.hidden = recent.length > 0;
  elements.dashboardRecentCases.innerHTML = recent.map((item) => `
    <button class="dashboard-case-row" type="button" data-work-kind="${escapeHtml(item.kind)}" data-work-id="${escapeHtml(item.id)}">
      <span><b>${escapeHtml(item.title)}</b><small>${item.kind === "diagnostic" ? "问题诊断" : "探索实验"} · ${escapeHtml(item.status)} · ${escapeHtml(item.detail)} · ${escapeHtml(formatDateTime(item.updated_at))}</small></span><i>${item.resumable ? "继续" : "查看"} →</i>
    </button>`).join("");
  elements.dashboardRecentCases.querySelectorAll("[data-work-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = recent.find((entry) => entry.kind === button.dataset.workKind && entry.id === button.dataset.workId);
      if (item?.resumePath) navigateTo(item.resumePath);
      else if (button.dataset.workKind === "diagnostic") openCase(button.dataset.workId);
      else openExplorationHistoryRecord(button.dataset.workKind, button.dataset.workId);
    });
  });
  const latest = state.workSummaries.find((item) => item.resumable)
    || state.workSummaries[0]
    || (state.caseHistory[0] ? { resumable: !state.caseHistory[0].status.startsWith("completed") } : null);
  elements.continueCaseButton.hidden = !latest;
  if (latest) elements.continueCaseButton.textContent = latest.resumable ? "继续最近工作" : "查看最近报告";
}

function workSummaryStatus(status) {
  return {
    planning: "等待规划",
    collecting: "采集中",
    awaiting_user_decision: "等待选择",
    completed: "已完成",
    inconclusive: "有边界结束",
  }[status] || status;
}

async function loadSettings(autoProbe = false) {
  try {
    const response = await fetch("/api/v1/settings");
    if (!response.ok) throw new Error(await readApiError(response));
    applySettings(await response.json());
    if (autoProbe && state.savedDevice) await checkSavedDevice(true);
  } catch (error) {
    renderDeviceState("error", "设置读取失败", error.message);
  }
}

function applySettings(settings) {
  state.settings = settings;
  state.savedDevice = settings.default_phyphox_device;
  elements.profileNameInput.value = settings.profile.display_name;
  if (state.currentUser) {
    state.currentUser.display_name = settings.profile.display_name;
    renderAccountIdentity();
  }
  const device = state.savedDevice;
  elements.checkSavedDeviceButton.disabled = !device;
  elements.removeDeviceButton.disabled = !device;
  if (!device) {
    elements.deviceNameInput.value = "我的手机";
    elements.deviceUrlInput.value = "";
    renderDeviceState("idle", "尚未设置默认手机", "使用模拟或文件数据不受影响。");
    renderDashboard();
    if (state.investigation) renderInvestigation();
    if (state.generalCase) renderGeneralLiveSource();
    return;
  }
  elements.deviceNameInput.value = device.name;
  elements.deviceUrlInput.value = device.base_url;
  elements.phyphoxBaseUrl.value = device.base_url;
  renderDeviceState(
    "saved",
    `${device.name} · 已保存`,
    `${device.base_url}${device.experiment_title ? ` · ${device.experiment_title}` : ""}`,
  );
  renderDashboard();
  if (state.investigation) renderInvestigation();
  if (state.generalCase) renderGeneralLiveSource();
}

function renderDeviceState(status, title, detail, probe = null) {
  const overviewState = status === "ready" ? "ready" : status === "error" ? "error" : "idle";
  const sensors = probeSensorKinds(probe);
  const profiledSensors = new Set(Object.keys(probe?.sensor_profiles || {}));
  const sensorSummary = sensors.length
    ? `<div class="device-sensor-summary"><span class="device-sensor-count">${sensors.length} 个实验输入</span><div class="device-sensor-list">${sensors.map((sensor) => `<span class="device-sensor-badge" data-profile="${profiledSensors.has(sensor) ? "ready" : "detected"}"><i></i>${escapeHtml(SENSOR_LABELS[sensor] || sensor)}<small>${profiledSensors.has(sensor) ? "可采集" : "已识别"}</small></span>`).join("")}</div></div>`
    : "";
  elements.deviceOverview.dataset.state = overviewState;
  elements.deviceOverview.innerHTML = `<i></i><div><b>${escapeHtml(title)}</b><span>${escapeHtml(detail)}</span>${sensorSummary}</div>`;
  elements.deviceSaveStatus.dataset.state = overviewState;
  elements.deviceSaveStatus.textContent = `${title}。${detail}`;
  elements.workflowDeviceStatus.innerHTML = `<b>${escapeHtml(title)}</b><span>${escapeHtml(detail)}</span>`;
  renderDashboard();
}

const MODEL_CAPABILITY_LABELS = {
  unverified: "尚未测试",
  unavailable: "接口不可用",
  text_only: "仅文本可用",
  tool_capable: "工具 Agent 可用",
  exploration_compatible: "探索 Agent 可用",
  agent_capable: "诊断 Agent 可用",
};

const MODEL_PROBE_ERROR_MESSAGES = {
  "model:non-chat-modality": "该模型属于图像生成、视频、Embedding 等非聊天模态，不能作为 PocketLab Agent 基模。请选择支持 Chat Completions 的语言模型。",
};

function modelProbeIssue(probe) {
  return probe?.error_codes?.map((code) => MODEL_PROBE_ERROR_MESSAGES[code]).find(Boolean) || "";
}

async function loadModelProfiles() {
  try {
    const response = await fetch("/api/v1/settings/models");
    if (!response.ok) throw new Error(await readApiError(response));
    state.modelCatalog = await response.json();
    renderModelProfiles();
  } catch (error) {
    elements.modelProfileList.innerHTML = `<p class="model-profile-empty error">${escapeHtml(error.message)}</p>`;
    elements.modelSaveStatus.dataset.state = "error";
    elements.modelSaveStatus.textContent = `模型配置读取失败：${error.message}`;
  }
}

function modelCapabilityItem(label, ready) {
  return `<span data-ready="${ready ? "true" : "false"}"><i>${ready ? "✓" : "—"}</i>${escapeHtml(label)}</span>`;
}

function modelStructuredTransport(probe) {
  if (probe?.structured_transport === "native_json_mode") return "原生 JSON";
  if (probe?.structured_transport === "validated_json_text") return "可验证 JSON 文本";
  return "结构化未验证";
}

function modelToolTransport(probe) {
  if (probe?.tool_transport === "named_function") return "强制函数";
  if (probe?.tool_transport === "auto") return "自动函数";
  return "工具未验证";
}

function modelIntegrationPresentation(profile) {
  if (profile.integration_status === "tuned_flash") {
    return {
      label: "已调配 · Flash",
      detail: "草案优先使用已验证的函数工具；证据解释按所选推理策略运行。",
    };
  }
  if (profile.integration_status === "tuned_pro") {
    return {
      label: "已调配 · Pro",
      detail: "草案直接使用严格 JSON，避免先经历不稳定的长工具循环；后续 Agent 分析仍调用基模。",
    };
  }
  return {
    label: "兼容试运行",
    detail: "尚未做专用调配。请先测试能力；若运行达到轮次上限，可提高轮数并重试基模。",
  };
}

function renderModelProfiles() {
  const catalog = state.modelCatalog;
  if (!catalog) return;
  elements.modelSecretBackend.textContent = catalog.secret_backend === "keyring"
    ? "完整密钥保存在操作系统凭据存储；页面和数据库只保留末四位。"
    : "当前只使用只读环境配置；页面不会返回完整密钥。";
  if (!catalog.profiles.length) {
    elements.modelProfileList.innerHTML = `
      <div class="model-profile-empty"><b>还没有可用模型</b><span>在右侧添加 OpenAI-compatible 接口；保存后再运行能力测试。</span></div>`;
    return;
  }
  elements.modelProfileList.innerHTML = catalog.profiles.map((profile) => {
    const probe = profile.probe;
    const probeIssue = modelProbeIssue(probe);
    const status = probe?.status || "unverified";
    const isActive = profile.profile_id === catalog.active_profile_id;
    const integration = modelIntegrationPresentation(profile);
    const cost = profile.input_cost_per_million == null && profile.output_cost_per_million == null
      ? "未填写价格"
      : `输入 ${profile.input_cost_per_million ?? "—"} / 输出 ${profile.output_cost_per_million ?? "—"}`;
    const reasoningLabel = ({ high: "High · 质量优先", fast: "Fast · 速度优先", auto: "High · 兼容旧配置", deep: "High · 兼容旧配置" })[profile.reasoning_strategy] || "供应商默认";
    const reasoningMode = profile.reasoning_strategy === "fast" ? "fast" : "high";
    const runtimeModeControl = isActive ? `
      <section class="model-runtime-mode" aria-label="当前活动模型推理模式">
        <div><b>运行模式</b><span>立即作用于后续诊断、探索、计划与报告生成</span></div>
        <div class="model-mode-switch" role="group" aria-label="选择 Fast 或 High">
          <button type="button" data-model-action="mode" data-profile-id="${escapeHtml(profile.profile_id)}" data-reasoning-strategy="high" aria-pressed="${reasoningMode === "high"}" ${reasoningMode === "high" ? "disabled" : ""}>High</button>
          <button type="button" data-model-action="mode" data-profile-id="${escapeHtml(profile.profile_id)}" data-reasoning-strategy="fast" aria-pressed="${reasoningMode === "fast"}" ${reasoningMode === "fast" ? "disabled" : ""}>Fast</button>
        </div>
      </section>` : "";
    return `
      <article class="model-profile-card" data-active="${isActive}" data-status="${escapeHtml(status)}">
        <header><div><span>${escapeHtml(profile.source === "environment" ? "ENV · CREDENTIALS READ ONLY" : "USER PROFILE")}</span><h4>${escapeHtml(profile.name)}</h4></div><em>${isActive ? "当前使用" : MODEL_CAPABILITY_LABELS[status]}</em></header>
        <div class="model-profile-endpoint"><b>${escapeHtml(profile.model_name)}</b><span title="${escapeHtml(profile.base_url)}">${escapeHtml(profile.base_url)}</span></div>
        <div class="model-profile-meta"><span>Key ${escapeHtml(profile.api_key_hint)}</span><span>${escapeHtml(reasoningLabel)}</span><span>${escapeHtml(cost)}</span></div>
        ${runtimeModeControl}
        <div class="model-integration-note" data-integration="${escapeHtml(profile.integration_status || "compatibility_trial")}"><b>${escapeHtml(integration.label)}</b><span>${escapeHtml(integration.detail)}</span></div>
        <div class="model-capability-row">
          ${modelCapabilityItem("文本", Boolean(probe?.text_generation))}
          ${modelCapabilityItem("结构化", Boolean(probe?.structured_json))}
          ${modelCapabilityItem("工具调用", Boolean(probe?.function_tools))}
        </div>
        ${probeIssue ? `<p class="model-probe-note model-probe-error">${escapeHtml(probeIssue)}</p>` : probe ? `<p class="model-probe-note">${escapeHtml(MODEL_CAPABILITY_LABELS[status])} · ${escapeHtml(modelStructuredTransport(probe))} · ${escapeHtml(modelToolTransport(probe))} · ${probe.latency_ms} ms · ${escapeHtml(formatDateTime(probe.checked_at))}</p>` : `<p class="model-probe-note">尚未验证该接口能否支撑 PocketLab Agent。</p>`}
        <footer>
          <button type="button" data-model-action="probe" data-profile-id="${escapeHtml(profile.profile_id)}">测试能力</button>
          ${isActive ? "" : `<button type="button" data-model-action="activate" data-profile-id="${escapeHtml(profile.profile_id)}">设为当前</button>`}
          ${profile.readonly ? "" : `<button type="button" data-model-action="edit" data-profile-id="${escapeHtml(profile.profile_id)}">编辑</button><button class="danger" type="button" data-model-action="delete" data-profile-id="${escapeHtml(profile.profile_id)}">删除</button>`}
        </footer>
      </article>`;
  }).join("");
}

function resetModelProfileForm() {
  state.editingModelProfileId = null;
  elements.modelProfileForm.reset();
  elements.modelReasoningStrategy.value = "high";
  elements.modelFormTitle.textContent = "新增模型配置";
  elements.modelFormMode.textContent = "NEW";
  elements.modelApiKeyHint.textContent = "新配置必须填写；编辑时留空表示保持原密钥";
  elements.saveModelProfileButton.querySelector("span").textContent = "安全保存配置";
  elements.cancelModelEditButton.hidden = true;
  elements.modelSaveStatus.dataset.state = "idle";
  elements.modelSaveStatus.textContent = "新增配置保存后会自动设为当前模型；请再运行能力测试。";
  if (elements.modelApiKey.type !== "password") toggleModelApiKey();
  elements.modelProfileName.focus();
}

function editModelProfile(profileId) {
  const profile = state.modelCatalog?.profiles.find((item) => item.profile_id === profileId);
  if (!profile || profile.readonly) return;
  state.editingModelProfileId = profileId;
  elements.modelProfileName.value = profile.name;
  elements.modelBaseUrl.value = profile.base_url;
  elements.modelNameInput.value = profile.model_name;
  elements.modelReasoningStrategy.value = ["fast", "high"].includes(profile.reasoning_strategy)
    ? profile.reasoning_strategy
    : "high";
  elements.modelApiKey.value = "";
  elements.modelInputCost.value = profile.input_cost_per_million ?? "";
  elements.modelOutputCost.value = profile.output_cost_per_million ?? "";
  elements.modelFormTitle.textContent = `编辑 ${profile.name}`;
  elements.modelFormMode.textContent = "EDIT";
  elements.modelApiKeyHint.textContent = `当前 ${profile.api_key_hint}；留空表示保持原密钥`;
  elements.saveModelProfileButton.querySelector("span").textContent = "保存修改";
  elements.cancelModelEditButton.hidden = false;
  elements.modelSaveStatus.dataset.state = "idle";
  elements.modelSaveStatus.textContent = "修改接口或模型名会清除旧测试结论；保存后请重新测试。";
  elements.modelProfileForm.scrollIntoView({ behavior: "smooth", block: "center" });
}

function toggleModelApiKey() {
  const showing = elements.modelApiKey.type === "text";
  elements.modelApiKey.type = showing ? "password" : "text";
  elements.modelApiKeyToggle.textContent = showing ? "显示" : "隐藏";
  elements.modelApiKeyToggle.setAttribute("aria-label", showing ? "显示 API Key" : "隐藏 API Key");
  elements.modelApiKeyToggle.setAttribute("aria-pressed", String(!showing));
}

function optionalCostValue(input) {
  return input.value.trim() === "" ? null : Number(input.value);
}

async function saveModelProfile(event) {
  event.preventDefault();
  if (state.busy) return;
  const editingId = state.editingModelProfileId;
  const apiKey = elements.modelApiKey.value.trim();
  if (!editingId && !apiKey) {
    elements.modelSaveStatus.dataset.state = "error";
    elements.modelSaveStatus.textContent = "新配置必须填写 API Key。";
    elements.modelApiKey.focus();
    return;
  }
  const payload = {
    name: elements.modelProfileName.value.trim(),
    base_url: elements.modelBaseUrl.value.trim(),
    model_name: elements.modelNameInput.value.trim(),
    reasoning_strategy: elements.modelReasoningStrategy.value,
    input_cost_per_million: optionalCostValue(elements.modelInputCost),
    output_cost_per_million: optionalCostValue(elements.modelOutputCost),
  };
  if (apiKey) payload.api_key = apiKey;
  if (!editingId) payload.make_default = true;
  setBusy(true, elements.saveModelProfileButton, "正在安全保存…");
  elements.modelSaveStatus.dataset.state = "working";
  elements.modelSaveStatus.textContent = "正在将密钥写入操作系统凭据存储…";
  try {
    const response = await fetch(editingId ? `/api/v1/settings/models/${encodeURIComponent(editingId)}` : "/api/v1/settings/models", {
      method: editingId ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    await response.json();
    await loadModelProfiles();
    resetModelProfileForm();
    await checkHealth();
    showToast(editingId ? "模型配置已更新，请重新测试能力" : "模型配置已安全保存并设为当前模型");
  } catch (error) {
    elements.modelSaveStatus.dataset.state = "error";
    elements.modelSaveStatus.textContent = error.message;
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.saveModelProfileButton, state.editingModelProfileId ? "保存修改" : "安全保存配置");
  }
}

async function handleModelProfileAction(event) {
  const button = event.target.closest("[data-model-action]");
  if (!button || state.busy) return;
  const profileId = button.dataset.profileId;
  const profile = state.modelCatalog?.profiles.find((item) => item.profile_id === profileId);
  if (!profile) return;
  const action = button.dataset.modelAction;
  if (action === "edit") {
    editModelProfile(profileId);
    return;
  }
  if (action === "delete") {
    if (!window.confirm(`确定删除模型配置“${profile.name}”吗？系统凭据库中的对应 API Key 也会移除。`)) return;
    await mutateModelProfile(button, `/api/v1/settings/models/${encodeURIComponent(profileId)}`, "DELETE", "正在删除…", "模型配置与对应凭据已删除");
    return;
  }
  if (action === "activate") {
    await mutateModelProfile(button, `/api/v1/settings/models/${encodeURIComponent(profileId)}/activate`, "POST", "正在切换…", `当前模型已切换为 ${profile.name}`);
    await checkHealth();
    return;
  }
  if (action === "mode") {
    await setActiveModelReasoningMode(button, button.dataset.reasoningStrategy);
    return;
  }
  if (action === "probe") await probeModelProfile(button, profile);
}

async function setActiveModelReasoningMode(button, reasoningStrategy) {
  const modeLabel = reasoningStrategy === "fast" ? "Fast" : "High";
  const idleLabel = button.textContent;
  setBusy(true, button, "切换中…");
  try {
    const response = await fetch("/api/v1/settings/models/active-mode", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reasoning_strategy: reasoningStrategy }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    state.modelCatalog = await response.json();
    renderModelProfiles();
    await checkHealth();
    showToast(`后续基模任务将使用 ${modeLabel} 模式`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, button, idleLabel);
  }
}

async function mutateModelProfile(button, url, method, busyLabel, successMessage) {
  const idleLabel = button.textContent;
  setBusy(true, button, busyLabel);
  try {
    const response = await fetch(url, { method });
    if (!response.ok) throw new Error(await readApiError(response));
    state.modelCatalog = await response.json();
    renderModelProfiles();
    if (state.editingModelProfileId && !state.modelCatalog.profiles.some((item) => item.profile_id === state.editingModelProfileId)) resetModelProfileForm();
    showToast(successMessage);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, button, idleLabel);
  }
}

async function probeModelProfile(button, profile) {
  const idleLabel = button.textContent;
  setBusy(true, button, "正在测试…");
  elements.modelSaveStatus.dataset.state = "working";
  elements.modelSaveStatus.textContent = "正在进行受限文本、JSON 与工具调用测试；通常需要数秒。";
  try {
    const response = await fetch(`/api/v1/settings/models/${encodeURIComponent(profile.profile_id)}/probe`, { method: "POST" });
    if (!response.ok) throw new Error(await readApiError(response));
    const probe = await response.json();
    const target = state.modelCatalog.profiles.find((item) => item.profile_id === profile.profile_id);
    if (target) target.probe = probe;
    renderModelProfiles();
    const probeIssue = modelProbeIssue(probe);
    elements.modelSaveStatus.dataset.state = probe.status === "unavailable" ? "error" : "ready";
    elements.modelSaveStatus.textContent = probeIssue || `${profile.name}：${MODEL_CAPABILITY_LABELS[probe.status]}。文本 ${probe.text_generation ? "通过" : "未通过"}；结构化：${modelStructuredTransport(probe)}；工具调用：${modelToolTransport(probe)}。`;
    showToast(`${profile.name}：${MODEL_CAPABILITY_LABELS[probe.status]}`, probe.status === "unavailable");
  } catch (error) {
    elements.modelSaveStatus.dataset.state = "error";
    elements.modelSaveStatus.textContent = error.message;
    showToast(error.message, true);
  } finally {
    setBusy(false, button, idleLabel);
  }
}

async function loadAgentRuns(silent = false) {
  if (!silent && state.busy) return;
  if (!silent) setBusy(true, elements.refreshAgentRunsButton, "正在刷新…");
  try {
    const response = await fetch("/api/v1/settings/agent-runs?limit=20");
    if (!response.ok) throw new Error(await readApiError(response));
    state.agentRunCatalog = await response.json();
    renderAgentRuns();
    if (!silent) showToast("Agent 运行审计已刷新");
  } catch (error) {
    elements.agentRuntimeSummary.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    if (!silent) showToast(error.message, true);
  } finally {
    if (!silent) setBusy(false, elements.refreshAgentRunsButton, "刷新审计");
  }
}

function renderAgentRuns() {
  const catalog = state.agentRunCatalog;
  if (!catalog) return;
  const summary = catalog.summary;
  elements.agentRuntimeSummary.innerHTML = `
    <div><span>最近运行</span><b>${summary.run_count}</b></div>
    <div><span>完成率</span><b>${formatNumber(summary.completion_rate * 100, 1)}%</b></div>
    <div><span>平均耗时</span><b>${formatNumber(summary.average_elapsed_s, 2)} s</b></div>
    <div><span>Tokens</span><b>${summary.total_tokens ?? "未上报"}</b></div>
    <div><span>估算成本</span><b>${summary.estimated_cost == null ? "未配置" : formatNumber(summary.estimated_cost, 6)}</b></div>`;
  elements.agentRuntimeList.innerHTML = catalog.runs.length
    ? catalog.runs.slice(0, 8).map((run) => {
      const reasoningMode = run.reasoning_mode === "deep"
        ? `High 推理${run.reasoning_effort ? ` · ${run.reasoning_effort}` : ""}`
        : run.reasoning_mode === "fast"
          ? "Fast 推理"
          : run.reasoning_mode === "provider_default"
            ? "供应商默认"
            : "模式未披露";
      return `
      <div class="agent-runtime-row" data-status="${escapeHtml(run.status)}">
        <span><b>${escapeHtml(run.operation)} · ${escapeHtml(run.model)}</b><small>${escapeHtml(reasoningMode)} · ${escapeHtml(formatDateTime(run.finished_at))} · ${formatNumber(run.elapsed_s, 2)} s · ${run.attempts} attempt(s) · ${run.total_tokens ?? "tokens 未上报"}${run.error_kind ? ` · ${escapeHtml(run.error_kind)}` : ""}</small></span>
        <strong>${run.status === "completed" ? "完成" : run.status === "cancelled" ? "已取消" : "失败"}</strong>
      </div>`;
    }).join("")
    : '<div class="agent-runtime-row"><span><b>还没有 Agent 运行</b><small>完成一次诊断规划、探索编译或证据解释后会显示在这里。</small></span><strong>等待</strong></div>';
  elements.agentRuntimeBoundary.textContent = catalog.privacy_boundary;
}

async function saveProfile() {
  const displayName = elements.profileNameInput.value.trim();
  if (!displayName || state.busy) return;
  setBusy(true, elements.saveProfileButton, "正在保存…");
  try {
    const response = await fetch("/api/v1/settings/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const profile = await response.json();
    state.settings.profile = profile;
    state.currentUser.display_name = profile.display_name;
    renderAccountIdentity();
    renderDashboard();
    showToast("档案名称已保存");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.saveProfileButton, "保存名称");
  }
}

async function saveDefaultDevice() {
  if (state.busy) return;
  const name = elements.deviceNameInput.value.trim();
  const baseUrl = elements.deviceUrlInput.value.trim();
  if (!name || !baseUrl) {
    showToast("请填写设备名称和 phyphox 地址。", true);
    return;
  }
  setBusy(true, elements.saveDeviceButton, "正在测试并保存…");
  renderDeviceState("saved", "正在连接手机", baseUrl);
  try {
    const body = { name, base_url: baseUrl };
    if (state.savedDevice?.buffer_mapping) body.buffer_mapping = state.savedDevice.buffer_mapping;
    const response = await fetch("/api/v1/settings/phyphox", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.savedDevice = data.device;
    state.settings.default_phyphox_device = data.device;
    state.phyphoxProbe = data.probe;
    elements.deviceUrlInput.value = data.device.base_url;
    elements.phyphoxBaseUrl.value = data.device.base_url;
    elements.checkSavedDeviceButton.disabled = false;
    elements.removeDeviceButton.disabled = false;
    const taskStatus = renderPhyphoxStatus(data.probe);
    renderDeviceState(
      "ready",
      `${data.device.name} · 手机已连接`,
      `${data.device.base_url} · ${data.probe.experiment_title} · ${probeInputText(data.probe)}`,
      data.probe,
    );
    showToast(`默认手机已保存；${taskStatus.title}`, taskStatus.error);
  } catch (error) {
    renderDeviceState("error", "设备没有保存", error.message);
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.saveDeviceButton, "测试连接并保存");
    updateSubmitButton();
    if (state.investigation) renderInvestigation();
  }
}

async function checkSavedDevice(silent = false) {
  if (!state.savedDevice || state.busy) return;
  setBusy(true, elements.checkSavedDeviceButton, "正在检测…");
  renderDeviceState("saved", `${state.savedDevice.name} · 正在检测`, state.savedDevice.base_url);
  try {
    const response = await fetch("/api/v1/settings/phyphox/probe", { method: "POST" });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.savedDevice = data.device;
    state.settings.default_phyphox_device = data.device;
    state.phyphoxProbe = data.probe;
    elements.phyphoxBaseUrl.value = data.device.base_url;
    const taskStatus = renderPhyphoxStatus(data.probe);
    renderDeviceState(
      "ready",
      `${data.device.name} · 手机已连接`,
      `${data.device.base_url} · ${data.probe.experiment_title} · ${probeInputText(data.probe)}`,
      data.probe,
    );
    if (!silent) showToast(taskStatus.title, taskStatus.error);
  } catch (error) {
    state.phyphoxProbe = null;
    resetPhyphoxStatus("已保存的地址当前不可达；手机换网后请更新设置。", true);
    renderDeviceState("error", `${state.savedDevice.name} · 当前离线`, error.message);
    if (!silent) showToast(error.message, true);
  } finally {
    setBusy(false, elements.checkSavedDeviceButton, "重新检测");
    updateSubmitButton();
    if (state.investigation) renderInvestigation();
  }
}

async function removeSavedDevice() {
  if (!state.savedDevice || state.busy) return;
  if (!window.confirm(`确定移除默认设备“${state.savedDevice.name}”吗？历史测量不会被删除。`)) return;
  setBusy(true, elements.removeDeviceButton, "正在移除…");
  try {
    const response = await fetch("/api/v1/settings/phyphox", { method: "DELETE" });
    if (!response.ok) throw new Error(await readApiError(response));
    state.phyphoxProbe = null;
    applySettings(await response.json());
    resetPhyphoxStatus();
    showToast("默认设备已移除，历史测量仍然保留");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.removeDeviceButton, "移除设备");
    elements.removeDeviceButton.disabled = !state.savedDevice;
    elements.checkSavedDeviceButton.disabled = !state.savedDevice;
    if (state.investigation) renderInvestigation();
  }
}

async function loadCaseHistory() {
  const response = await fetch("/api/v1/diagnostic-cases");
  if (!response.ok) throw new Error(await readApiError(response));
  state.caseHistory = await response.json();
  renderCaseHistory();
  await loadWorkSummaries();
}

async function loadExplorationHistory() {
  const response = await fetch("/api/v2/exploration-history");
  if (!response.ok) throw new Error(await readApiError(response));
  state.explorationHistory = await response.json();
  renderExplorationHistory();
  await loadWorkSummaries();
}

async function loadWorkSummaries() {
  if (workSummaryRequest) return workSummaryRequest;
  workSummaryRequest = (async () => {
    const response = await fetch("/api/v2/work-summaries");
    if (!response.ok) throw new Error(await readApiError(response));
    state.workSummaries = await response.json();
    renderDashboard();
  })();
  try {
    return await workSummaryRequest;
  } finally {
    workSummaryRequest = null;
  }
}

function explorationHistoryStatus(item) {
  return {
    in_progress: "进行中",
    completed: "已完成",
    limited: "有边界结果",
    unsupported: "已安全停止",
    inconclusive: "证据不足",
  }[item.status] || item.status;
}

function explorationHistorySource(item) {
  return {
    public_replay: "公开证据演示",
    simulated_rehearsal: "模拟演练 · 非现实证据",
    phone_or_import: "手机或导入实验",
  }[item.data_source] || item.data_source;
}

function renderExplorationHistory() {
  elements.explorationHistoryCount.textContent = String(state.explorationHistory.length);
  elements.explorationHistoryEmpty.hidden = state.explorationHistory.length > 0;
  elements.explorationHistoryList.innerHTML = state.explorationHistory.map((item) => `
    <article class="exploration-history-row" data-kind="${escapeHtml(item.record_kind)}" data-id="${escapeHtml(item.record_id)}">
      <button class="exploration-history-main" type="button" data-action="open">
        <span class="exploration-history-heading"><b>${escapeHtml(item.title)}</b><i class="exploration-history-status ${escapeHtml(item.status)}">${item.superseded_by_case_id ? "旧版本 · 已修订" : escapeHtml(explorationHistoryStatus(item))}</i></span>
        <p>${escapeHtml(item.research_question)}</p>
        <div class="exploration-history-meta"><span>${escapeHtml(SENSOR_LABELS[item.primary_sensor] || item.primary_sensor)}</span><span>${escapeHtml(explorationHistorySource(item))}</span>${item.compiler_source ? `<span>${item.compiler_source === "bounded_agent_compiler" ? "Agent 编译凭证" : "手工审阅协议"}</span>` : ""}<span>${item.evidence_count} 项证据 · ${item.tool_count} 次工具</span>${item.superseded_by_case_id ? "<span>旧证据仅供回看</span>" : ""}<span>${escapeHtml(formatDateTime(item.updated_at))}</span></div>
        ${item.report_summary ? `<small>${escapeHtml(item.report_summary)}</small>` : ""}
      </button>
      <button class="button button-secondary" type="button" data-action="open">${item.superseded_by_case_id ? "查看旧版本" : item.resumable ? "继续实验" : "查看报告"}</button>
    </article>`).join("");
  elements.explorationHistoryList.querySelectorAll(".exploration-history-row").forEach((row) => {
    row.querySelectorAll('[data-action="open"]').forEach((button) => {
      button.addEventListener("click", () => openExplorationHistoryRecord(row.dataset.kind, row.dataset.id));
    });
  });
}

async function openExplorationHistoryRecord(kind, recordId) {
  if (state.busy) return;
  if (kind === "investigation") {
    navigateTo(`/app/explore/runs/${encodeURIComponent(recordId)}`);
    return;
  }
  if (kind === "general_exploration") {
    navigateTo(`/app/explore/general/runs/${encodeURIComponent(recordId)}`);
    return;
  }
  try {
    const response = await fetch(`/api/v2/exploration-history/public/${encodeURIComponent(recordId)}`);
    if (!response.ok) throw new Error(await readApiError(response));
    const { result } = await response.json();
    navigateTo("/app/explore/presets");
    if (result.protocol_id === "light-public-exploration.v1") {
      state.publicLightRun = result;
      elements.publicLightQuestion.value = result.research_question;
      showPublicRunner("light");
      renderPublicLightResult(result);
      elements.publicLightRunner.scrollIntoView({ behavior: "smooth", block: "start" });
    } else if (result.protocol_id === "pressure-public-exploration.v1") {
      state.publicPressureRun = result;
      elements.publicPressureQuestion.value = result.research_question;
      showPublicRunner("pressure");
      renderPublicPressureResult(result);
      elements.publicPressureRunner.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      const config = publicSensorBetaConfig(result.sensor, result.protocol_id);
      if (!config) throw new Error("该历史实验的预设协议已不可用。");
      state.publicSensorRun = result;
      state.publicSensorActive = result.sensor;
      state.publicSensorProtocol = result.protocol_id;
      elements.publicSensorName.textContent = config.label;
      elements.publicSensorQuestionLabel.textContent = config.label;
      elements.publicSensorIntro.textContent = config.intro;
      elements.publicSensorQuestion.value = result.research_question;
      showPublicRunner("sensor");
      renderPublicSensorResult(result);
      elements.publicSensorRunner.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    showToast("已从探索历史恢复实验报告");
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderCaseHistory() {
  elements.caseHistoryCount.textContent = String(state.caseHistory.length);
  elements.caseHistoryEmpty.hidden = state.caseHistory.length > 0;
  elements.caseHistoryList.innerHTML = state.caseHistory.map((item) => `
    <article class="case-history-row" data-id="${escapeHtml(item.case_id)}">
      <button class="case-history-main" type="button" data-action="open">
        <span><b>${escapeHtml(item.title)}</b><i class="case-status ${escapeHtml(item.status)}">${item.superseded_by_case_id ? "旧版本 · 已修订" : escapeHtml(caseStatusText(item.status))}</i></span>
        <p>${escapeHtml(item.problem_statement)}</p>
        <div class="case-history-meta"><span>${item.evidence_count} 项证据</span><span>${item.superseded_by_case_id ? "旧证据仅供回看" : item.current_task_title ? `当前：${escapeHtml(item.current_task_title)}` : "无待执行任务"}</span><span>${escapeHtml(formatDateTime(item.updated_at))}</span></div>
      </button>
      <div class="case-history-actions">
        <button class="open-case" type="button" data-action="open">${item.superseded_by_case_id ? "查看旧版本" : item.status.startsWith("completed") ? "查看报告" : "继续案例"}</button>
        <button class="delete-case" type="button" data-action="delete">删除</button>
      </div>
    </article>`).join("");
  elements.caseHistoryList.querySelectorAll(".case-history-row").forEach((row) => {
    row.querySelectorAll('[data-action="open"]').forEach((button) => button.addEventListener("click", () => openCase(row.dataset.id)));
    row.querySelector('[data-action="delete"]').addEventListener("click", () => deleteCase(row.dataset.id));
  });
  renderDashboard();
}

async function openCase(caseId, options = {}) {
  const { updateRoute = true, scroll = true } = options;
  if (state.busy) return;
  try {
    const response = await fetch(`/api/v1/diagnostic-cases/${encodeURIComponent(caseId)}/snapshot`);
    if (!response.ok) throw new Error(await readApiError(response));
    const snapshot = await response.json();
    state.diagnosticCase = snapshot.case;
    state.diagnosticRetryRecording = null;
    state.latestAgentMessage = snapshot.latest_agent_message;
    state.pendingFile = null;
    if (updateRoute) {
      window.history.pushState({}, "", `/app/cases/${encodeURIComponent(caseId)}`);
      applyRoute(false);
    }
    elements.caseSetup.hidden = true;
    elements.activeWorkflow.hidden = false;
    renderDiagnosticCase(snapshot.latest_agent_message || "案例已经从本地历史记录恢复。可以继续当前任务，或复查最终报告。");
    switchMeasurementMode(state.diagnosticCase.current_task ? "public" : state.measurementMode);
    const lastEvidence = state.diagnosticCase.evidence.at(-1);
    if (lastEvidence) await viewSession(lastEvidence.session_id, snapshot.latest_agent_message, false);
    if (scroll) elements.activeWorkflow.scrollIntoView({ behavior: "smooth", block: "start" });
    showToast(state.diagnosticCase.final_report ? "最终报告已恢复" : "未完成案例已恢复，可以继续测量");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function deleteCase(caseId) {
  const item = state.caseHistory.find((candidate) => candidate.case_id === caseId);
  if (!item || state.busy) return;
  if (!window.confirm(`确定删除诊断案例“${item.title}”吗？测量 Session 将继续保留。`)) return;
  try {
    const response = await fetch(`/api/v1/diagnostic-cases/${encodeURIComponent(caseId)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await readApiError(response));
    if (state.diagnosticCase?.case_id === caseId) resetCaseView();
    await loadCaseHistory();
    showToast("案例已删除，测量记录仍然保留");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function loadSessionHistory() {
  const response = await fetch("/api/v1/sessions");
  if (!response.ok) throw new Error(await readApiError(response));
  state.sessions = (await response.json()).map((item) => ({ ...item, samples: null }));
  renderSessions();
  renderSelectedEvidence();
  renderDashboard();
}

async function createDiagnosticCase() {
  if (state.busy) return;
  const title = elements.caseTitleInput.value.trim();
  const problem = elements.problemInput.value.trim();
  if (title.length < 2 || problem.length < 10) {
    showToast("请填写至少 2 字的案例名称和至少 10 字的问题描述。", true);
    return;
  }
  setBusy(true, elements.createDiagnosticButton, "Agent 正在规划…");
  try {
    const response = await fetch("/api/v1/diagnostic-cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        problem_statement: problem,
        context: elements.caseContextInput.value.trim(),
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticCase = data.case;
    state.pendingFile = null;
    state.measurementMode = "public";
    elements.caseSetup.hidden = true;
    elements.activeWorkflow.hidden = false;
    renderDiagnosticCase(data.agent_message);
    switchMeasurementMode("public");
    await loadCaseHistory();
    window.history.replaceState({}, "", `/app/cases/${encodeURIComponent(data.case.case_id)}`);
    applyRoute(false);
    showToast("第一项实验已经准备好");
    elements.activeWorkflow.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.createDiagnosticButton, "让 Agent 规划第一项实验");
  }
}

function resetCaseView() {
  showNewCaseForm();
  elements.caseTitleInput.value = "";
  elements.problemInput.value = "";
  elements.caseContextInput.value = "";
  navigateTo("/app/cases/new");
}

function renderDiagnosticFeedbackSelection() {
  const diagnosticCase = state.diagnosticCase;
  const feedbackType = elements.diagnosticFeedbackType.value;
  const targetsHypothesis = ["hypothesis_not_applicable", "hypothesis_needs_correction"].includes(feedbackType);
  if (!targetsHypothesis) {
    state.diagnosticFeedbackHypothesisIds.clear();
    elements.diagnosticFeedbackSelection.textContent = feedbackType === "task_not_feasible"
      ? "将重新设计当前这一步；请说明家中什么条件使它做不了。"
      : feedbackType === "instruction_unclear"
        ? "请指出哪句话看不懂，Agent 会改成可照做的步骤。"
        : "请补充 Agent 不知道的现场事实，例如设备结构、可接近位置或安全限制。";
    return;
  }
  const selected = [...state.diagnosticFeedbackHypothesisIds];
  if (!selected.length) {
    elements.diagnosticFeedbackSelection.textContent = "请先点击上方某条解释中的“不符合我家实际”。";
    return;
  }
  const labels = selected.map((id) => {
    const hypothesis = diagnosticCase?.hypotheses?.find((candidate) => candidate.hypothesis_id === id);
    return hypothesis ? `${id}：${hypothesis.statement}` : id;
  });
  elements.diagnosticFeedbackSelection.textContent = feedbackType === "hypothesis_not_applicable"
    ? `将整条排除：${labels.join("；")}`
    : `将按你的说明修正：${labels.join("；")}`;
}

function handleDiagnosticFeedbackTarget(event) {
  const button = event.target.closest("[data-diagnostic-feedback-hypothesis]");
  if (!button || !state.diagnosticCase) return;
  state.diagnosticFeedbackHypothesisIds.clear();
  state.diagnosticFeedbackHypothesisIds.add(button.dataset.diagnosticFeedbackHypothesis);
  elements.diagnosticFeedbackType.value = "hypothesis_needs_correction";
  elements.diagnosticRealityFeedback.open = true;
  renderDiagnosticFeedbackSelection();
  elements.diagnosticFeedbackMessage.focus();
}

function handleDiagnosticTaskFeedback(event) {
  const successor = event.target.closest("[data-open-diagnostic-successor]");
  if (successor) {
    navigateTo(`/app/cases/${encodeURIComponent(successor.dataset.openDiagnosticSuccessor)}`);
    return;
  }
  const button = event.target.closest("[data-diagnostic-task-feedback]");
  if (!button || !state.diagnosticCase) return;
  state.diagnosticFeedbackHypothesisIds.clear();
  elements.diagnosticFeedbackType.value = button.dataset.diagnosticTaskFeedback;
  elements.diagnosticRealityFeedback.open = true;
  renderDiagnosticFeedbackSelection();
  elements.diagnosticFeedbackMessage.focus();
}

async function submitDiagnosticRealityFeedback() {
  const diagnosticCase = state.diagnosticCase;
  if (!diagnosticCase || state.busy || diagnosticCase.superseded_by_case_id) return;
  const feedbackType = elements.diagnosticFeedbackType.value;
  const message = elements.diagnosticFeedbackMessage.value.trim();
  const targetsHypothesis = ["hypothesis_not_applicable", "hypothesis_needs_correction"].includes(feedbackType);
  const hypothesisIds = targetsHypothesis
    ? [...state.diagnosticFeedbackHypothesisIds]
    : [];
  if (message.length < 3) {
    elements.diagnosticFeedbackStatus.dataset.state = "error";
    elements.diagnosticFeedbackStatus.textContent = "请用一句日常语言说明实际情况。";
    return;
  }
  if (targetsHypothesis && !hypothesisIds.length) {
    elements.diagnosticFeedbackStatus.dataset.state = "error";
    elements.diagnosticFeedbackStatus.textContent = "请先选择哪条解释不符合实际。";
    return;
  }
  state.busy = true;
  elements.diagnosticFeedbackSubmit.disabled = true;
  elements.diagnosticFeedbackStatus.dataset.state = "loading";
  elements.diagnosticFeedbackStatus.textContent = "正在保留旧记录，并按你家的实际情况重做诊断计划…";
  try {
    const response = await fetch(`/api/v1/diagnostic-cases/${encodeURIComponent(diagnosticCase.case_id)}/reality-feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        feedback_type: feedbackType,
        message,
        hypothesis_ids: hypothesisIds,
        expected_task_id: diagnosticCase.current_task?.task_id || null,
        confirm_sensitive_sensor_reuse: elements.diagnosticFeedbackPrivacy.checked,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const result = await response.json();
    state.diagnosticCase = result.case;
    state.diagnosticFeedbackHypothesisIds.clear();
    elements.diagnosticFeedbackMessage.value = "";
    elements.diagnosticFeedbackPrivacy.checked = false;
    window.history.replaceState({}, "", `/app/cases/${encodeURIComponent(result.case.case_id)}`);
    applyRoute(false);
    renderDiagnosticCase(result.agent_message);
    switchMeasurementMode("public");
    await loadCaseHistory();
    elements.diagnosticFeedbackStatus.dataset.state = "success";
    elements.diagnosticFeedbackStatus.textContent = `新诊断已按现场事实生成。${realityFeedbackReuseSummary(result.case.revision_feedback, "诊断")}`;
    showToast("已按你家的实际情况生成新诊断");
  } catch (error) {
    elements.diagnosticFeedbackStatus.dataset.state = "error";
    elements.diagnosticFeedbackStatus.textContent = error.message;
  } finally {
    state.busy = false;
    elements.diagnosticFeedbackSubmit.disabled = false;
    updateSubmitButton();
  }
}

function renderDiagnosticCase(agentMessage = "") {
  const diagnosticCase = state.diagnosticCase;
  if (!diagnosticCase) return;
  const showcase = isDiagnosticShowcaseCase(diagnosticCase);
  elements.activeWorkflow.dataset.showcase = String(showcase);
  const feedbackAllowed = !showcase && !diagnosticCase.superseded_by_case_id && !diagnosticCase.final_report;
  elements.diagnosticRealityFeedback.hidden = !feedbackAllowed;
  elements.diagnosticFeedbackPrivacy.parentElement.hidden = !diagnosticCase.evidence.some(
    (item) => ["microphone", "location"].includes(item.sensor),
  );
  elements.diagnosticCaseTitle.textContent = diagnosticCase.title;
  elements.diagnosticCaseId.textContent = `CASE ${diagnosticCase.case_id}`;
  renderDiagnosticRuntimeNotice(diagnosticCase);
  elements.hypothesisList.innerHTML = diagnosticCase.hypotheses.map((item) => `
    <article class="hypothesis-card ${escapeHtml(item.status)}">
      <div class="hypothesis-state"><span>${escapeHtml(item.hypothesis_id.toUpperCase())}</span><b>${escapeHtml(hypothesisStatusText(item.status))}</b></div>
      <h4>${escapeHtml(item.statement)}</h4>
      <p>${escapeHtml(item.latest_reasoning || item.rationale)}</p>
      ${feedbackAllowed ? `<button type="button" class="hypothesis-feedback-button" data-diagnostic-feedback-hypothesis="${escapeHtml(item.hypothesis_id)}">这条解释不符合我家实际</button>` : ""}
    </article>`).join("");
  renderCurrentTask();
  renderTerminationProgress();
  renderFinalReport();
  if (agentMessage) {
    state.latestAgentMessage = agentMessage;
    elements.diagnosticAgentMessage.classList.add("rich-output");
    elements.diagnosticAgentMessage.innerHTML = renderRichText(agentMessage);
  }
}

function renderCurrentTask() {
  const successorCaseId = state.diagnosticCase?.superseded_by_case_id;
  if (successorCaseId) {
    elements.currentTask.innerHTML = `
      <span class="task-code">ARCHIVED PLAN</span>
      <h4>这一步已被现场反馈替换</h4>
      <p>原任务和已有测量仍保留供你回看，但不会继续写入，也不会混入新诊断。</p>
      <button type="button" class="task-feedback-button" data-open-diagnostic-successor="${escapeHtml(successorCaseId)}">打开按现场事实修订后的诊断</button>`;
    elements.submitTaskButton.disabled = true;
    elements.taskAnalyzerNotice.hidden = true;
    return;
  }
  const task = state.diagnosticCase?.current_task;
  if (!task) {
    const completed = Boolean(state.diagnosticCase?.final_report);
    elements.currentTask.innerHTML = completed
      ? "<span class=\"task-code\">CASE COMPLETE</span><h4>诊断已结束</h4><p>终止向量已经作出裁决，系统不会再生成新的 Task。</p>"
      : "<p>当前没有待执行的测量任务。</p>";
    elements.submitTaskButton.disabled = true;
    elements.taskAnalyzerNotice.hidden = true;
    return;
  }
  const sensor = taskSensorDetails(task);
  const showcase = isDiagnosticShowcaseCase();
  const sensorPlan = state.diagnosticCase?.sensor_plan || [];
  const taskOperationLabel = diagnosticTaskOperationLabel(task.task_kind);
  const controlledVariables = [...new Set(task.controlled_variables.map((item) => item.trim()).filter(Boolean))];
  const measurementSummary = showcase
    ? `<span><b>后台回放</b>服务器提交冻结的${escapeHtml(sensor.quantity)}序列，页面仍使用标准分析证据组件</span>`
    : `<span><b>手机上打开</b>${escapeHtml(sensor.label)}实验，数据由 PocketLab 自动分析</span>`;
  const taskFeedback = showcase
    ? ""
    : `<div class="task-feedback-actions"><button type="button" class="task-feedback-button" data-diagnostic-task-feedback="task_not_feasible">这一步在我家做不了</button><button type="button" class="task-feedback-button" data-diagnostic-task-feedback="instruction_unclear">我没看懂怎样操作</button></div>`;
  elements.currentTask.innerHTML = `
    <span class="task-code">${escapeHtml(task.task_id.toUpperCase())}</span>
    <h4>${escapeHtml(task.title)}</h4>
    ${renderHouseholdInstruction(task.instruction)}
    <div class="task-plain-summary">${measurementSummary}<span><b>${escapeHtml(taskOperationLabel)}</b>${escapeHtml(task.variable_to_change)}</span></div>
    ${controlledVariables.length ? `<details class="task-controls"><summary>实验中还要保持哪些条件不变</summary><ul>${controlledVariables.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul><small>${showcase ? "这些控制条件已经冻结在后台回放数据中。" : "如果其中一项在你家无法做到，请点下方“这一步在我家做不了”。"}</small></details>` : ""}
    ${taskFeedback}
    ${sensorPlan.length ? `<div class="diagnostic-sensor-plan"><b>Agent 传感器计划</b>${sensorPlan.map((item) => `<span class="${escapeHtml(item.role)}"><i>${escapeHtml(item.role === "primary" ? "主" : item.role === "supporting" ? "辅" : "选")}</i>${escapeHtml(SENSOR_LABELS[item.sensor] || item.sensor)}<small>${escapeHtml(item.rationale)}</small></span>`).join("")}</div>` : ""}`;
  elements.measurementLabelInput.value = `${state.diagnosticCase.title} · ${task.task_id}`;
  elements.phyphoxLabel.value = `${state.diagnosticCase.title} · ${task.task_id} · phyphox`;
  elements.mobileServerUrl.textContent = window.location.origin;
  elements.mobileCaseCode.textContent = state.diagnosticCase.case_id;
  elements.mobileTaskCode.textContent = task.task_id;
  renderTaskSensorGuidance(task);
  elements.publicDiagnosticTitle.textContent = showcase
    ? `零等待诊断回放 · ${sensor.label}证据已冻结`
    : `使用${sensor.label}的已审阅公开记录`;
  elements.publicDiagnosticBoundary.textContent = showcase
    ? `点击后立即提交本轮${sensor.quantity}后台数据，更新假设并进入下一步或报告；不请求基模，也不代表你家现场。`
    : `系统会运行${sensor.quantity}分析器并让 Agent 更新竞争机制；公开回放不是你家现场或你手机的证据。`;
  elements.publicDiagnosticPrivacy.parentElement.hidden = showcase;
  elements.publicDiagnosticRunButton.querySelector("span").textContent = showcase
    ? "回放本步并立即推进"
    : "运行公开回放并交给 Agent";
  if (showcase && state.measurementMode !== "public") switchMeasurementMode("public");
  if (state.measurementMode === "simulation" && sensor.sensor === "accelerometer") {
    const recommendedProfile = suggestProfileForTask(state.diagnosticCase, task);
    if (recommendedProfile) elements.simulationProfile.value = recommendedProfile;
    updateSimulationProfile();
  } else if (state.measurementMode === "simulation") {
    switchMeasurementMode("public");
  }
  updateSubmitButton();
}

function renderTaskSensorGuidance(task) {
  const sensor = taskSensorDetails(task);
  elements.mobileSensorIntro.innerHTML = `当前 Task 需要 <b>${escapeHtml(sensor.label)}</b>（${escapeHtml(sensor.quantity)}）。请在手机 <a href="https://phyphox.org/download/" target="_blank" rel="noreferrer">phyphox</a> 中打开${escapeHtml(sensor.experiment)}，再启用远程访问。`;
  elements.taskAnalyzerNotice.hidden = sensor.analyzerReady;
  const privacySensitive = sensor.sensor === "microphone" || sensor.sensor === "location";
  elements.mobilePrivacyConfirm.hidden = !privacySensitive;
  if (!privacySensitive) elements.mobilePrivacyCheckbox.checked = false;
  if (!sensor.analyzerReady) {
    elements.taskAnalyzerNotice.innerHTML = `<b>当前分析器尚未接入</b><br />这个 Task 的传感器要求已经正确识别为“${escapeHtml(sensor.label)}”，但 PocketLab 目前不能把该数据提交为正式诊断证据。连接检测仍会告诉你 phyphox 实验是否匹配；采集、模拟和文件提交暂时禁用，避免错误使用加速度算法。`;
  }
  if (state.phyphoxProbe) renderPhyphoxStatus(state.phyphoxProbe);
  else resetPhyphoxStatus();
}

function renderTerminationProgress() {
  const diagnosticCase = state.diagnosticCase;
  const vector = diagnosticCase?.termination_vector;
  if (!vector || diagnosticCase.final_report) {
    elements.terminationProgress.hidden = true;
    return;
  }
  elements.terminationProgress.hidden = false;
  const blockers = vector.blockers?.length
    ? vector.blockers.slice(0, 3).join("；")
    : "等待首项有效证据与对照实验";
  if (vector.user_decision_required || diagnosticCase.checkpoint_pending) {
    elements.terminationProgress.innerHTML = `
      <b>已到 20 次诊断检查点</b>
      <span>当前仍没有足够清晰的唯一解释。你可以继续执行 Agent 已选好的判别任务，或现在停止并领取带置信度的当前答案。</span>
      <div class="diagnostic-checkpoint-actions"><button type="button" data-diagnostic-decision="continue">继续探求</button><button type="button" data-diagnostic-decision="stop">停止并生成报告</button></div>`;
    elements.terminationProgress.querySelectorAll("[data-diagnostic-decision]").forEach((button) => {
      button.addEventListener("click", () => decideDiagnosticCheckpoint(button.dataset.diagnosticDecision));
    });
  } else {
    const title = vector.hypothesis_revision_required
      ? "原候选解释已被同时削弱 · Agent 正在重规划"
      : "终止向量尚未达标";
    elements.terminationProgress.innerHTML = `<b>${escapeHtml(title)}</b><span>${escapeHtml(blockers)}</span>`;
  }
}

function diagnosticTaskOperationLabel(taskKind) {
  return ({
    baseline: "这次先记录",
    control: "这次只改变",
    replication: "这次照上次重复",
    correction: "这次先修正采样",
    exploration: "这次观察",
  })[taskKind] || "这次要做";
}

function renderDiagnosticRuntimeNotice(diagnosticCase) {
  if (isDiagnosticShowcaseCase(diagnosticCase)) {
    elements.diagnosticRuntimeNotice.innerHTML = "<b>零等待服务器回放 · 0 次模型请求</b><p>候选解释、两轮加速度序列与推进规则均已冻结；点击只会提交后台数据并运行标准分析、证据和终止状态机。</p><small>这是产品流程演示，不是当前家庭、洗衣机或手机的现场诊断。</small>";
    elements.diagnosticRuntimeNotice.hidden = false;
    return;
  }
  const intakeFallback = diagnosticCase.intake_transport === "deterministic_fallback";
  const latestReceipt = diagnosticCase.evidence?.at(-1)?.reasoning_receipt;
  const measurementFallback = latestReceipt?.transport === "deterministic_fallback";
  if (!intakeFallback && !measurementFallback) {
    elements.diagnosticRuntimeNotice.hidden = true;
    elements.diagnosticRuntimeNotice.innerHTML = "";
    return;
  }
  const reasonText = [
    diagnosticCase.intake_fallback_reason,
    latestReceipt?.fallback_reason,
  ].filter(Boolean).join(" ").toLowerCase();
  const timedOut = reasonText.includes("timeout");
  const attempts = Number(diagnosticCase.intake_model_requests || 0);
  const title = timedOut ? "模型连续超时 · 当前为弱基线" : "模型规划已安全降级";
  const intakeText = intakeFallback
    ? `初始规划在 ${attempts || "多"} 次模型请求后仍未取得可校验草案；当前候选假设和第一项任务来自安全 fallback，不是完整诊断结果。`
    : "";
  const measurementText = measurementFallback
    ? "最近一轮证据只完成了确定性指标更新，尚未获得完整的模型物理解释。"
    : "";
  elements.diagnosticRuntimeNotice.innerHTML = `<b>${escapeHtml(title)}</b><p>${escapeHtml([intakeText, measurementText].filter(Boolean).join(" "))}</p><small>可以把当前任务用于保留现场基线，但不要把弱基线或降级解释当作最终原因判断。</small>`;
  elements.diagnosticRuntimeNotice.hidden = false;
}

async function decideDiagnosticCheckpoint(decision) {
  const diagnosticCase = state.diagnosticCase;
  if (!diagnosticCase?.checkpoint_pending || state.busy) return;
  state.busy = true;
  updateSubmitButton();
  try {
    const response = await fetch(`/api/v1/diagnostic-cases/${encodeURIComponent(diagnosticCase.case_id)}/checkpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision,
        expected_completed_task_count: diagnosticCase.termination_vector.completed_task_count,
      }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticCase = data.case;
    renderDiagnosticCase(data.agent_message);
    if (data.case.current_task) switchMeasurementMode("public");
    await loadCaseHistory();
    showToast(decision === "continue" ? "诊断继续，下一任务已恢复" : "已生成当前证据范围内的报告");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.busy = false;
    updateSubmitButton();
  }
}

function renderFinalReport() {
  const report = state.diagnosticCase?.final_report;
  elements.finalReportBlock.hidden = !report;
  elements.measureBlock.hidden = Boolean(report);
  if (!report) return;
  const vector = report.vector;
  elements.finalOutcomeBadge.textContent = report.outcome === "completed_with_conclusion"
    ? `有倾向性结论 · ${confidenceText(report.confidence)}`
    : state.diagnosticCase?.termination_invalidated
      ? "旧版结论已失效 · 需要重规划"
      : "证据不足 · 已停止";
  elements.finalConclusion.textContent = report.answer_headline || report.conclusion;
  elements.finalUserTakeaway.textContent = report.user_takeaway || report.answer_headline || "";
  elements.finalUserTakeaway.hidden = !elements.finalUserTakeaway.textContent;
  elements.finalMechanismExplanation.textContent = report.mechanism_explanation || "";
  elements.finalMechanismExplanation.hidden = !report.mechanism_explanation;
  elements.finalConfidenceExplanation.textContent = report.confidence_explanation || "";
  elements.finalConfidenceExplanation.hidden = !report.confidence_explanation;
  elements.finalTerminationReason.textContent = report.termination_reason;
  const evidenceExplanation = report.evidence_explanation?.length
    ? report.evidence_explanation
    : ["旧报告未保存逐条数值解释；请查看下方终止向量与证据记录。"];
  elements.finalEvidenceExplanation.innerHTML = evidenceExplanation.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const metrics = [
    ["有效证据", vector.effective_evidence_count, "要求 ≥ 2", vector.effective_evidence_count >= 2],
    ["匹配对照", vector.matched_control_count, "要求 ≥ 1", vector.matched_control_count >= 1],
    ["区分覆盖率", `${Math.round(vector.hypothesis_coverage_ratio * 100)}%`, "要求 ≥ 67%", vector.hypothesis_coverage_ratio >= (2 / 3)],
    ["领先支持度", formatNumber(vector.leading_support, 2), "要求 ≥ 0.72", vector.leading_support >= 0.72],
    ["领先差值", formatNumber(vector.leading_margin, 2), "要求 ≥ 0.25", vector.leading_margin >= 0.25],
    ["正向权重", formatNumber(vector.leading_positive_weight, 1), "要求 ≥ 1.6", vector.leading_positive_weight >= 1.6],
    ["高质反证", vector.high_quality_contradictions, "要求 = 0", vector.high_quality_contradictions === 0],
    ["信息增益", formatNumber(vector.recent_information_gain, 2), "本轮最大变化", true],
  ];
  elements.terminationGrid.innerHTML = metrics.map(([label, value, note, pass]) => `
    <div class="${pass ? "pass" : ""}"><span>${escapeHtml(String(label))}</span><b>${escapeHtml(String(value))}</b><small>${escapeHtml(String(note))}</small></div>
  `).join("");
  const uncertainties = report.remaining_uncertainties.length
    ? report.remaining_uncertainties
    : ["没有额外记录；结论仍只适用于本案例的设备、测点和控制条件。"];
  elements.finalUncertainties.innerHTML = uncertainties.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.finalScopeBoundary.textContent = report.scope_boundary || "";
  elements.finalScopeBoundary.hidden = !report.scope_boundary;
  renderSolutionPlan(report.solution_plan, report);
}

function renderSolutionPlan(plan, report = state.diagnosticCase?.final_report) {
  elements.finalSolutionPlan.hidden = !plan;
  if (!plan) return;
  elements.solutionBasisBadge.textContent = plan.basis === "evidence_supported"
    ? "证据支持的处理路径"
    : "证据不足 · 安全下一步";
  const source = report?.finalization_source || "legacy_unattributed";
  const fallbackCount = (state.diagnosticCase?.evidence || []).filter(
    (item) => item.reasoning_receipt?.transport === "deterministic_fallback",
  ).length;
  const evidenceCount = state.diagnosticCase?.evidence?.length || 0;
  if (isDiagnosticShowcaseCase()) {
    elements.solutionProvenance.className = "solution-provenance showcase";
    elements.solutionProvenanceBadge.textContent = "服务器冻结回放 · 0 次模型请求";
    elements.solutionProvenanceNote.textContent = "处理建议来自受约束的演示方案，并保留现实现场复测与安全升级边界。";
    elements.retryFinalReportButton.hidden = true;
  } else if (source === "model_generated") {
    elements.solutionProvenance.className = "solution-provenance model-generated";
    elements.solutionProvenanceBadge.textContent = "基模生成 · 安全审查通过";
    elements.solutionProvenanceNote.textContent = [
      report.finalization_model ? `模型：${report.finalization_model}` : "",
      report.finalization_model_requests ? `请求 ${report.finalization_model_requests} 次` : "",
      fallbackCount ? `其中 ${fallbackCount}/${evidenceCount} 轮测量推理曾使用安全兜底` : "测量推理未记录兜底",
    ].filter(Boolean).join(" · ");
    elements.retryFinalReportButton.hidden = true;
  } else if (source === "deterministic_fallback") {
    elements.solutionProvenance.className = "solution-provenance degraded";
    elements.solutionProvenanceBadge.textContent = "安全兜底 · 不是完整基模结果";
    const reason = report.finalization_fallback_reason || "模型未返回可安全采纳的完整方案";
    elements.solutionProvenanceNote.textContent = `${reason} · 已有证据未丢失，可以只重试最终报告。`;
    elements.retryFinalReportButton.hidden = !report.finalization_retryable;
  } else {
    elements.solutionProvenance.className = "solution-provenance legacy";
    elements.solutionProvenanceBadge.textContent = "历史方案 · 旧版本未记录生成来源";
    elements.solutionProvenanceNote.textContent = "可以重新请求一次，生成带明确来源和安全审查记录的完整报告。";
    elements.retryFinalReportButton.hidden = false;
  }
  elements.solutionSummary.textContent = plan.summary;
  elements.solutionActions.innerHTML = plan.actions.map((action, index) => `
    <article>
      <div class="solution-action-title">
        <span>${String(index + 1).padStart(2, "0")}</span>
        <b>${escapeHtml(action.title)}</b>
        <div class="solution-action-badges">
          <small class="action-role ${escapeHtml(action.action_role || "resolve")}">${escapeHtml(actionRoleText(action.action_role))}</small>
          <small>${escapeHtml(riskLevelText(action.risk_level))}</small>
        </div>
      </div>
      <p>${escapeHtml(action.rationale)}</p>
      ${action.preparation?.length ? `<div class="action-preparation"><b>开始前准备</b><ul>${action.preparation.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul></div>` : ""}
      <ol>${action.steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol>
      <div class="expected-result"><b>期待变化</b><span>${escapeHtml(action.expected_result)}</span></div>
      ${action.how_to_verify ? `<div class="action-decision"><b>怎样确认有效</b><span>${escapeHtml(action.how_to_verify)}</span></div>` : ""}
      ${action.if_not_improved ? `<div class="action-decision no-improvement"><b>如果没有改善</b><span>${escapeHtml(action.if_not_improved)}</span></div>` : ""}
      ${(action.estimated_time || action.tools_needed?.length) ? `<div class="action-resources"><span>预计 ${escapeHtml(action.estimated_time || "时间视现场而定")}</span><span>需要：${escapeHtml((action.tools_needed || []).join("、") || "无需额外工具")}</span></div>` : ""}
      ${action.do_not_do?.length ? `<div class="action-prohibitions"><b>不要这样做</b><ul>${action.do_not_do.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
      ${action.safety_notes.length ? `<div class="action-safety">${escapeHtml(action.safety_notes.join(" "))}</div>` : ""}
    </article>`).join("");
  elements.solutionEscalation.innerHTML = plan.escalation_conditions.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  const retest = plan.optional_retest;
  elements.optionalRetest.hidden = !retest;
  if (retest) {
    elements.optionalRetestTitle.textContent = retest.title;
    elements.optionalRetestInstruction.textContent = retest.instruction;
    elements.optionalRetestCriteria.innerHTML = retest.success_criteria.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  }
}

function riskLevelText(level) {
  if (level === "professional") return "专业处理";
  if (level === "caution") return "谨慎操作";
  return "低风险";
}

async function createOptionalRetestCase() {
  const sourceCase = state.diagnosticCase;
  const retest = state.diagnosticCase?.final_report?.solution_plan?.optional_retest;
  if (!sourceCase || !retest || state.busy) return;
  setBusy(true, elements.copyRetestButton, "Agent 正在规划复测…");
  try {
    const response = await fetch(
      `/api/v1/diagnostic-cases/${encodeURIComponent(sourceCase.case_id)}/retest`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticCase = data.case;
    state.diagnosticRetryRecording = null;
    state.pendingFile = null;
    state.measurementMode = "public";
    elements.caseSetup.hidden = true;
    elements.activeWorkflow.hidden = false;
    renderDiagnosticCase(data.agent_message);
    switchMeasurementMode("public");
    await loadCaseHistory();
    window.history.pushState({}, "", `/app/cases/${encodeURIComponent(data.case.case_id)}`);
    applyRoute(false);
    showToast("处理后复测已建立；请按新的 Step 2 开始测量");
    elements.activeWorkflow.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message || "无法建立处理后复测。", true);
  } finally {
    setBusy(false, elements.copyRetestButton, "创建可执行复测诊断");
  }
}

async function retryDiagnosticFinalReport() {
  const diagnosticCase = state.diagnosticCase;
  if (!diagnosticCase?.final_report || state.busy) return;
  setBusy(true, elements.retryFinalReportButton, "基模正在重新生成完整报告…");
  try {
    const response = await fetch(
      `/api/v1/diagnostic-cases/${encodeURIComponent(diagnosticCase.case_id)}/final-report/retry`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticCase = data.case;
    renderDiagnosticCase(data.agent_message);
    await loadCaseHistory();
    showToast(
      data.case.final_report?.finalization_source === "model_generated"
        ? "完整基模报告已生成并通过安全审查"
        : "模型仍未完成；安全兜底已明确保留，可稍后重试",
      data.case.final_report?.finalization_source !== "model_generated",
    );
  } catch (error) {
    showToast(error.message || "无法重新生成最终报告。", true);
  } finally {
    setBusy(false, elements.retryFinalReportButton, "重新请求完整基模报告");
  }
}

function actionRoleText(role) {
  if (role === "verify") return "验证效果";
  if (role === "escalate") return "升级处理";
  return "直接处理";
}

function switchMeasurementMode(mode) {
  const taskSensor = taskSensorDetails();
  if (mode === "simulation" && taskSensor.sensor !== "accelerometer") {
    showToast(`${taskSensor.label}不使用加速度合成信号替代，已切换到已审阅公开回放。`);
    mode = "public";
  }
  state.measurementMode = mode;
  document.querySelectorAll(".source-choice").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  elements.publicDiagnosticPane.hidden = mode !== "public";
  elements.simulationPane.hidden = mode !== "simulation";
  elements.filePane.hidden = mode !== "file";
  elements.mobilePane.hidden = mode !== "mobile";
  elements.measurementForm.hidden = mode === "mobile" || mode === "public";
  updateSubmitButton();
}

async function loadTaskFile(file) {
  if (file.size > 12 * 1024 * 1024) {
    showToast("文件超过 12 MB，请先缩短记录。", true);
    return;
  }
  try {
    const parsed = await parseDatasetFile(file);
    state.pendingFile = { ...parsed, name: file.name };
    const duration = datasetDuration(parsed.samples);
    elements.taskFileTitle.textContent = file.name;
    elements.taskFileMeta.textContent = `${parsed.samples.length.toLocaleString()} samples · ${duration.toFixed(2)} s`;
    if (parsed.label) elements.measurementLabelInput.value = parsed.label;
    showToast(`已读取 ${parsed.samples.length.toLocaleString()} 个采样点`);
  } catch (error) {
    state.pendingFile = null;
    showToast(error.message || "无法解析这个文件。", true);
  } finally {
    elements.taskFileInput.value = "";
    updateSubmitButton();
  }
}

async function runDiagnosticPublicReplay() {
  const diagnosticCase = state.diagnosticCase;
  const task = diagnosticCase?.current_task;
  if (!task || state.busy) return;
  const showcase = isDiagnosticShowcaseCase(diagnosticCase);
  setBusy(true, elements.publicDiagnosticRunButton, showcase ? "正在回放并推进…" : "正在验证来源、分析并运行 Agent…");
  try {
    const endpoint = showcase
      ? `/api/v2/showcase-replays/diagnostic/${encodeURIComponent(diagnosticCase.case_id)}/tasks/${encodeURIComponent(task.task_id)}`
      : `/api/v2/diagnostic-cases/${encodeURIComponent(diagnosticCase.case_id)}/tasks/${encodeURIComponent(task.task_id)}/public-replay`;
    const request = showcase
      ? { method: "POST" }
      : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          privacy_acknowledged: elements.publicDiagnosticPrivacy.checked,
          observation_notes: "使用已审阅公开记录验收诊断闭环；这不是当前家庭现场证据。",
        }),
      };
    const response = await fetch(endpoint, request);
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      const detail = errorPayload.detail;
      if (detail?.code === "diagnostic_agent_unavailable" && detail.recording_id) {
        state.diagnosticRetryRecording = {
          recordingId: detail.recording_id,
          taskId: detail.task_id,
          observationNotes: "重试已保存的公开记录；不重新导入或采集。",
        };
        renderDiagnosticRetry();
      }
      throw new Error(detail?.message || `请求失败（HTTP ${response.status}）`);
    }
    const data = await response.json();
    state.diagnosticRetryRecording = null;
    renderDiagnosticRetry();
    state.diagnosticCase = data.case;
    state.sensorRecordings = [data.session, ...state.sensorRecordings.filter((item) => item.session_id !== data.session.session_id)];
    renderDiagnosticCase(data.agent_message);
    renderLatestResult(data.session, data.preview_samples, data.agent_message);
    await loadCaseHistory();
    const finished = Boolean(data.case.final_report);
    showToast(finished
      ? (showcase ? "洗衣机零等待诊断回放完成" : "公开回放已形成有边界的诊断报告")
      : (showcase ? "本步已回放，下一实验状态已就绪" : "公开回放已绑定，Agent 已选择下一项测量"));
    elements.latestResult.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.publicDiagnosticRunButton, showcase ? "回放本步并立即推进" : "运行公开回放并交给 Agent");
    updateSubmitButton();
  }
}

function renderDiagnosticRetry() {
  const retry = state.diagnosticRetryRecording;
  const currentTaskId = state.diagnosticCase?.current_task?.task_id;
  const available = Boolean(retry && retry.taskId === currentTaskId);
  elements.diagnosticRetryButton.hidden = !available;
  elements.diagnosticRetryButton.disabled = state.busy || !available;
}

async function retryDiagnosticRecording() {
  const retry = state.diagnosticRetryRecording;
  const diagnosticCase = state.diagnosticCase;
  if (!retry || !diagnosticCase?.current_task || state.busy) return;
  setBusy(true, elements.diagnosticRetryButton, "正在重新请求 Agent…");
  try {
    const response = await fetch(
      `/api/v2/diagnostic-cases/${encodeURIComponent(diagnosticCase.case_id)}/tasks/${encodeURIComponent(retry.taskId)}/recordings`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recording_id: retry.recordingId,
          observation_notes: retry.observationNotes,
        }),
      },
    );
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticRetryRecording = null;
    state.diagnosticCase = data.case;
    state.sensorRecordings = [data.session, ...state.sensorRecordings.filter((item) => item.session_id !== data.session.session_id)];
    renderDiagnosticRetry();
    renderDiagnosticCase(data.agent_message);
    renderLatestResult(data.session, data.preview_samples, data.agent_message);
    await loadCaseHistory();
    showToast(data.case.final_report ? "重试成功，诊断报告已生成" : "重试成功，诊断已经继续");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.diagnosticRetryButton, "重试刚才已保存的记录");
    renderDiagnosticRetry();
  }
}

async function submitTaskMeasurement() {
  const diagnosticCase = state.diagnosticCase;
  const task = diagnosticCase?.current_task;
  if (!task || state.busy || state.measurementMode === "mobile") return;
  const sensor = taskSensorDetails(task);
  if (!sensor.analyzerReady) {
    showToast(`${sensor.label}分析器尚未接入，不能用加速度模拟或文件替代本任务数据。`, true);
    return;
  }
  const label = elements.measurementLabelInput.value.trim();
  if (!label) {
    showToast("请填写本次测量名称。", true);
    return;
  }
  let samples;
  let device;
  let notes;
  if (state.measurementMode === "simulation") {
    const profile = SIMULATION_PROFILES[elements.simulationProfile.value];
    samples = generateProfileSamples(profile);
    device = "PocketLab simulator";
    notes = `模拟信号：${profile.label}。仅用于软件闭环验收，不代表真实物理测量。`;
  } else {
    if (!state.pendingFile) {
      showToast("请先选择 CSV 或 JSON 测量文件。", true);
      return;
    }
    samples = state.pendingFile.samples;
    device = "Imported sensor data";
    notes = state.pendingFile.notes || "网页导入的测量数据";
  }

  setBusy(true, elements.submitTaskButton, "正在分析证据并判断继续或结束…");
  try {
    const response = await fetch(
      `/api/v1/mobile/cases/${encodeURIComponent(diagnosticCase.case_id)}/tasks/${encodeURIComponent(task.task_id)}/samples`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label,
          device,
          sensor: "accelerometer",
          notes,
          observation_notes: elements.observationInput.value.trim(),
          samples,
        }),
      },
    );
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticCase = data.case;
    addSession(data.session, samples);
    state.pendingFile = null;
    elements.taskFileTitle.textContent = "选择或拖入测量文件";
    elements.taskFileMeta.textContent = "至少 64 个采样点，包含时间、x、y、z";
    elements.observationInput.value = "";
    renderDiagnosticCase(data.agent_message);
    renderLatestResult(data.session, samples, data.agent_message);
    await loadCaseHistory();
    const finished = Boolean(data.case.final_report);
    showToast(finished ? "终止向量已达标，最终诊断报告已生成" : "测量已绑定，下一项实验已生成");
    elements.latestResult.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.submitTaskButton, "分析并提交本次测量");
    updateSubmitButton();
  }
}

async function refreshMobileTask() {
  const diagnosticCase = state.diagnosticCase;
  if (!diagnosticCase || state.busy) return;
  setBusy(true, elements.refreshTaskButton, "正在读取手机任务…");
  try {
    const response = await fetch(`/api/v1/mobile/cases/${encodeURIComponent(diagnosticCase.case_id)}/task`);
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    if (data.task) {
      elements.mobileTaskCode.textContent = data.task.task_id;
      showToast(`手机桥在线：当前任务 ${data.task.task_id}`);
    } else {
      elements.mobileTaskCode.textContent = "CASE COMPLETE";
      showToast("该案例已经结束，没有待执行任务");
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.refreshTaskButton, "刷新手机任务状态");
  }
}

function resetPhyphoxStatus(message = "") {
  state.phyphoxProbe = null;
  const task = state.diagnosticCase?.current_task;
  const sensor = taskSensorDetails(task);
  const guidance = message || (task
    ? `当前 Task 需要${sensor.label}；请打开${sensor.experiment}并启用远程访问。`
    : "先在 phyphox 中打开任一实验并启用远程访问，再填写它显示的地址。");
  elements.phyphoxStatus.dataset.state = "idle";
  elements.phyphoxStatus.innerHTML = `<b>等待连接真机</b><span>${escapeHtml(guidance)}</span>`;
  renderExplorations();
}

function renderPhyphoxStatus(probe) {
  const task = state.diagnosticCase?.current_task;
  const sensor = taskSensorDetails(task);
  const inputText = probeInputText(probe);
  let status;
  if (!task) {
    status = {
      state: "ready",
      error: false,
      title: `手机已连接 · ${probe.experiment_title}`,
      detail: `当前实验输入：${inputText}。创建或打开 Task 后，PocketLab 会再检查是否匹配。`,
    };
  } else if (!probeMatchesTask(probe, task)) {
    status = {
      state: "error",
      error: true,
      title: `当前实验不匹配 · Task 需要${sensor.label}`,
      detail: `phyphox 当前识别为 ${inputText}；请打开${sensor.experiment}后重新检测。`,
    };
  } else if (!sensor.analyzerReady) {
    status = {
      state: "blocked",
      error: true,
      title: `实验匹配 · 已识别${sensor.label}`,
      detail: `当前 phyphox 实验满足 Task，但 PocketLab 的${sensor.quantity}分析器尚未接入，因此暂不能提交正式证据。`,
    };
  } else if (sensor.sensor === "accelerometer" && !probe.compatible) {
    status = {
      state: "error",
      error: true,
      title: "已识别加速度输入，但缓冲区不完整",
      detail: `缺少 ${probe.missing_buffers.join("、")}；请换用标准加速度实验后重新检测。`,
    };
  } else {
    const available = probe.available_buffers.slice(0, 8).join("、");
    status = {
      state: "ready",
      error: false,
      title: `实验匹配 · ${sensor.label} · ${probe.experiment_title}`,
      detail: `已识别 ${available}；可以开始采集。`,
    };
  }
  elements.phyphoxStatus.dataset.state = status.state;
  elements.phyphoxStatus.innerHTML = `<b>${escapeHtml(status.title)}</b><span>${escapeHtml(status.detail)}</span>`;
  renderExplorations();
  return status;
}

async function probePhyphoxConnection() {
  if (state.busy) return;
  const baseUrl = elements.phyphoxBaseUrl.value.trim();
  if (!baseUrl) {
    showToast("请填写 phyphox 在手机上显示的远程地址。", true);
    return;
  }
  setBusy(true, elements.probePhyphoxButton, "正在检测…");
  try {
    const response = await fetch("/api/v1/phyphox/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: baseUrl }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const probe = await response.json();
    state.phyphoxProbe = probe;
    elements.phyphoxBaseUrl.value = probe.base_url;
    const status = renderPhyphoxStatus(probe);
    showToast(status.title, status.error);
  } catch (error) {
    resetPhyphoxStatus(error.message);
    elements.phyphoxStatus.dataset.state = "error";
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.probePhyphoxButton, "检测手机连接");
  }
}

async function capturePhyphoxMeasurement() {
  const diagnosticCase = state.diagnosticCase;
  const task = diagnosticCase?.current_task;
  const sensor = taskSensorDetails(task);
  if (!task || state.busy) return;
  if (!sensor.analyzerReady) {
    showToast(`${sensor.label}实验已识别，但对应分析器尚未接入，不能提交正式证据。`, true);
    return;
  }
  if (!probeMatchesTask(state.phyphoxProbe, task)) {
    showToast(`当前 phyphox 实验不匹配，请打开${sensor.experiment}。`, true);
    return;
  }
  const duration = Number(elements.phyphoxDuration.value);
  const label = elements.phyphoxLabel.value.trim();
  if (!Number.isFinite(duration) || duration < 3 || duration > 60) {
    showToast("真机采集时长必须在 3 到 60 秒之间。", true);
    return;
  }
  if (!label) {
    showToast("请填写本次真机测量名称。", true);
    return;
  }

  setBusy(true, elements.capturePhyphoxButton, `正在采集 ${duration} 秒并等待 Agent…`);
  try {
    const response = await fetch(
      `/api/v2/diagnostic-cases/${encodeURIComponent(diagnosticCase.case_id)}/tasks/${encodeURIComponent(task.task_id)}/phyphox`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          base_url: state.phyphoxProbe.base_url,
          duration_s: duration,
          label,
          notes: "诊断任务通过 phyphox 通用传感器桥采集。",
          observation_notes: elements.phyphoxObservation.value.trim(),
          privacy_acknowledged: elements.mobilePrivacyCheckbox.checked,
        }),
      },
    );
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    state.diagnosticCase = data.case;
    state.sensorRecordings = [data.session, ...state.sensorRecordings.filter((item) => item.session_id !== data.session.session_id)];
    elements.phyphoxObservation.value = "";
    renderDiagnosticCase(data.agent_message);
    renderLatestResult(data.session, data.preview_samples, data.agent_message);
    await loadCaseHistory();
    const finished = Boolean(data.case.final_report);
    showToast(finished ? "真机证据已完成最终诊断" : "真机证据已绑定，下一项实验已经生成");
    elements.latestResult.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.capturePhyphoxButton, "采集并交给 Agent");
  }
}

function renderLatestResult(session, samples, message) {
  const analysis = session.analysis;
  elements.latestResult.hidden = false;
  if (Array.isArray(analysis.metrics)) {
    const metrics = analysis.metrics.slice(0, 2);
    const first = metrics[0];
    const second = metrics[1];
    elements.metricOneLabel.textContent = first?.label || "主指标";
    elements.metricFrequency.textContent = first ? formatNumber(first.value, 3) : "—";
    elements.metricOneUnit.textContent = first?.unit || "";
    elements.metricTwoLabel.textContent = second?.label || "辅助指标";
    elements.metricRms.textContent = second ? formatNumber(second.value, 3) : "—";
    elements.metricTwoUnit.textContent = second?.unit || "";
    elements.metricThreeLabel.textContent = "采样率";
    elements.metricRate.textContent = formatNumber(analysis.sampling_rate_hz, 1);
    elements.metricBand.textContent = `${SENSOR_LABELS[analysis.sensor] || analysis.sensor} · ${analysis.analyzer_id}`;
  } else {
    elements.metricOneLabel.textContent = "主频";
    elements.metricFrequency.textContent = formatNumber(analysis.dominant_frequency_hz, 2);
    elements.metricOneUnit.textContent = "Hz";
    elements.metricTwoLabel.textContent = "RMS";
    elements.metricRms.textContent = formatNumber(analysis.rms_acceleration_m_s2, 3);
    elements.metricTwoUnit.textContent = "m/s²";
    elements.metricThreeLabel.textContent = "采样率";
    elements.metricRate.textContent = formatNumber(analysis.sampling_rate_hz, 1);
    elements.metricBand.textContent = `建议分析 ≤ ${formatNumber(analysis.recommended_frequency_limit_hz, 1)} Hz`;
  }
  elements.metricConfidence.textContent = confidenceText(analysis.confidence);
  elements.metricWarning.textContent = analysis.warnings.length ? analysis.warnings.join("；") : "NO WARNING";
  if (message) {
    state.latestAgentMessage = message;
    elements.diagnosticAgentMessage.classList.add("rich-output");
    elements.diagnosticAgentMessage.innerHTML = renderRichText(message);
  }
  if (Array.isArray(samples) && samples.length) drawSignalChart(samples);
  else drawEmptyChart();
}

function addSession(created, samples) {
  const record = { ...created, samples };
  state.sessions = [record, ...state.sessions.filter((item) => item.session_id !== created.session_id)];
  state.activeId = created.session_id;
  state.selectedIds.add(created.session_id);
  renderSessions();
  renderSelectedEvidence();
}

function renderSessions() {
  const records = workbenchEvidenceItems();
  elements.sessionHistoryCount.textContent = String(records.length);
  elements.sessionEmpty.hidden = records.length > 0;
  elements.sessionList.innerHTML = records.map((session) => {
    const selected = state.selectedIds.has(session.session_id);
    const active = state.activeId === session.session_id;
    const sensor = session.sensor || session.analysis?.sensor || "accelerometer";
    const source = session.provenance?.source || "legacy_session";
    return `
      <div class="session-row${active ? " active" : ""}" data-id="${escapeHtml(session.session_id)}">
        <button class="session-main" type="button" data-action="view">
          <span><b>${escapeHtml(session.label)}</b><small>${escapeHtml(session.session_id)} · ${escapeHtml(formatDateTime(session.created_at))} · ${escapeHtml(SENSOR_LABELS[sensor] || sensor)} · ${escapeHtml(source)}</small></span>
          <span class="session-metrics"><span>${escapeHtml(workbenchMetricPreview(session))}</span><span>${escapeHtml(confidenceText(session.analysis?.confidence))}</span></span>
        </button>
        <label class="session-select"><input type="checkbox" data-action="select" ${selected ? "checked" : ""} />加入证据工作台</label>
      </div>`;
  }).join("");
  elements.sessionList.querySelectorAll(".session-row").forEach((row) => {
    const sessionId = row.dataset.id;
    row.querySelector('[data-action="view"]').addEventListener("click", () => viewSession(sessionId));
    row.querySelector('[data-action="select"]').addEventListener("change", (event) => {
      toggleSessionSelection(sessionId, event.target.checked);
    });
  });
}

async function viewSession(sessionId, message = state.latestAgentMessage, scroll = true) {
  state.activeId = sessionId;
  let session = getActiveSession()
    || state.sensorRecordings.find((item) => item.session_id === sessionId)
    || null;
  try {
    if (session && !Array.isArray(session.samples)) session = await loadSessionDetails(sessionId);
    if (session) {
      renderLatestResult(session, session.samples, message);
      if (scroll) elements.latestResult.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    renderSessions();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function loadSessionDetails(sessionId) {
  const isSensorRecording = state.sensorRecordings.some((item) => item.session_id === sessionId);
  const endpoint = isSensorRecording
    ? `/api/v2/recordings/${encodeURIComponent(sessionId)}`
    : `/api/v1/sessions/${encodeURIComponent(sessionId)}`;
  const response = await fetch(endpoint);
  if (!response.ok) throw new Error(await readApiError(response));
  const data = await response.json();
  const record = {
    session_id: data.session_id,
    label: data.upload.label,
    device: data.upload.device,
    notes: data.upload.notes,
    sample_count: data.upload.samples.length,
    sensor: data.upload.sensor || data.analysis?.sensor || "accelerometer",
    provenance: data.upload.provenance || { source: "legacy_session" },
    analysis: data.analysis,
    created_at: data.created_at,
    samples: data.upload.samples,
  };
  if (isSensorRecording) {
    state.sensorRecordings = state.sensorRecordings.map(
      (item) => item.session_id === sessionId ? record : item,
    );
  } else {
    state.sessions = state.sessions.map((item) => item.session_id === sessionId ? record : item);
  }
  return record;
}

function toggleSessionSelection(sessionId, checked) {
  if (checked && state.selectedIds.size >= 4) {
    showToast("证据工作台最多选择 4 次测量。", true);
    renderSessions();
    renderEvidenceWorkbenchLibrary();
    return;
  }
  if (checked) state.selectedIds.add(sessionId);
  else state.selectedIds.delete(sessionId);
  renderSelectedEvidence();
  renderEvidenceWorkbenchLibrary();
}

function workbenchEvidenceItems() {
  const byId = new Map();
  state.sessions.forEach((item) => byId.set(item.session_id, {
    ...item,
    sensor: "accelerometer",
    provenance: { source: "legacy_session" },
    evidenceVersion: "v1",
  }));
  state.sensorRecordings.forEach((item) => byId.set(item.session_id, {
    ...item,
    evidenceVersion: "v2",
  }));
  return [...byId.values()].sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")));
}

function workbenchMetricPreview(item) {
  const metrics = item.analysis?.metrics;
  if (Array.isArray(metrics) && metrics.length) {
    return metrics.slice(0, 2).map((metric) => `${metric.label} ${formatNumber(metric.value, 3)} ${metric.unit || ""}`.trim()).join(" · ");
  }
  if (item.analysis?.dominant_frequency_hz !== undefined) {
    return `主频 ${formatNumber(item.analysis.dominant_frequency_hz, 2)} Hz · RMS ${formatNumber(item.analysis.rms_acceleration_m_s2, 3)} m/s²`;
  }
  return "等待确定性指标";
}

function renderEvidenceWorkbenchLibrary() {
  if (!elements.evidenceWorkbenchLibrary) return;
  const items = workbenchEvidenceItems();
  if (!items.length) {
    elements.evidenceWorkbenchLibrary.innerHTML = "<p>还没有可用记录。请先在诊断或探索中完成一次测量。</p>";
    return;
  }
  elements.evidenceWorkbenchLibrary.innerHTML = items.map((item) => {
    const sensor = item.sensor || item.analysis?.sensor || "accelerometer";
    const selected = state.selectedIds.has(item.session_id);
    const source = item.provenance?.source || "legacy_session";
    return `<label class="workbench-evidence-row ${selected ? "selected" : ""}">
      <input type="checkbox" data-workbench-evidence="${escapeHtml(item.session_id)}" ${selected ? "checked" : ""} />
      <span><b>${escapeHtml(item.label)}</b><small>${escapeHtml(SENSOR_LABELS[sensor] || sensor)} · ${escapeHtml(item.evidenceVersion.toUpperCase())} · ${escapeHtml(source)}</small><em>${escapeHtml(workbenchMetricPreview(item))}</em></span>
      <strong>${escapeHtml(confidenceText(item.analysis?.confidence))}</strong>
    </label>`;
  }).join("");
  elements.evidenceWorkbenchLibrary.querySelectorAll("[data-workbench-evidence]").forEach((input) => {
    input.addEventListener("change", (event) => toggleSessionSelection(input.dataset.workbenchEvidence, event.target.checked));
  });
}

function renderSelectedEvidence() {
  const selected = workbenchEvidenceItems().filter((item) => state.selectedIds.has(item.session_id));
  elements.selectedSessionChips.innerHTML = selected.length
    ? selected.map((item) => `<span class="evidence-chip">${escapeHtml(item.label)}</span>`).join("")
    : "尚未选择测量";
  updateAdvancedButton();
  renderEvidenceWorkbenchLibrary();
}

const WORKBENCH_STATUS_LABELS = {
  model_generated: "MODEL + DETERMINISTIC AUDIT",
  deterministic_only: "DETERMINISTIC AUDIT ONLY",
};

const WORKBENCH_COMPARISON_LABELS = {
  direct: "可以直接对照",
  limited: "受限对照",
  context_only: "仅作机制上下文",
};

async function loadWorkbenchReports() {
  try {
    const response = await fetch("/api/v2/evidence-workbench/reports");
    if (!response.ok) throw new Error(await readApiError(response));
    state.workbenchReports = await response.json();
    renderWorkbenchReportHistory();
  } catch (error) {
    elements.workbenchReportHistory.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
  }
}

function renderWorkbenchReportHistory() {
  const reports = state.workbenchReports.slice(0, 8);
  elements.workbenchReportHistory.innerHTML = reports.length
    ? reports.map((item) => `
      <button class="workbench-history-row" type="button" data-workbench-report-id="${escapeHtml(item.report_id)}">
        <span><b>${escapeHtml(item.question)}</b><small>${item.recording_count} 条证据 · ${escapeHtml(item.sensor_kinds.map((sensor) => SENSOR_LABELS[sensor] || sensor).join(" / "))} · ${escapeHtml(formatDateTime(item.created_at))}</small></span>
        <strong>${escapeHtml(confidenceText(item.overall_confidence))}</strong>
      </button>`).join("")
    : "<p>还没有保存的证据报告。</p>";
}

async function handleWorkbenchHistoryClick(event) {
  const button = event.target.closest("[data-workbench-report-id]");
  if (!button || state.busy) return;
  try {
    const response = await fetch(`/api/v2/evidence-workbench/reports/${encodeURIComponent(button.dataset.workbenchReportId)}`);
    if (!response.ok) throw new Error(await readApiError(response));
    renderWorkbenchReport(await response.json());
    elements.agentResponse.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(error.message, true);
  }
}

function workbenchRecordLabel(report, recordingId) {
  return report.audits.find((item) => item.recording_id === recordingId)?.label || recordingId;
}

function workbenchCitation(report, recordingId) {
  return report.citations?.find((item) => item.recording_id === recordingId)?.citation_id || "—";
}

function renderWorkbenchMatrix(report) {
  const audits = report.audits || [];
  const cells = new Map((report.comparability_matrix || []).map((item) => [`${item.left_recording_id}|${item.right_recording_id}`, item]));
  if (!audits.length || !cells.size) return '<p class="workbench-contrast-empty">旧报告没有矩阵合同；重新运行证据审计后生成。</p>';
  const statusLabels = { same_record: "自身", direct: "直接", limited: "受限", context_only: "互证" };
  const header = audits.map((item) => `<b title="${escapeHtml(item.label)}">[${escapeHtml(workbenchCitation(report, item.recording_id))}]</b>`).join("");
  const rows = audits.map((left) => {
    const values = audits.map((right) => {
      const cell = cells.get(`${left.recording_id}|${right.recording_id}`);
      if (!cell) return '<span data-status="context_only">—</span>';
      const detail = `${cell.reason}${cell.shared_metric_keys.length ? ` 共有指标：${cell.shared_metric_keys.join("、")}` : ""}`;
      return `<span data-status="${escapeHtml(cell.status)}" title="${escapeHtml(detail)}">${escapeHtml(statusLabels[cell.status] || cell.status)}</span>`;
    }).join("");
    return `<b title="${escapeHtml(left.label)}">[${escapeHtml(workbenchCitation(report, left.recording_id))}]</b>${values}`;
  }).join("");
  return `<div class="workbench-matrix-title"><b>记录可比性矩阵</b><small>悬停单元格查看门禁理由</small></div><div class="workbench-matrix-grid" style="grid-template-columns:94px repeat(${audits.length},minmax(76px,1fr))"><i></i>${header}${rows}</div>`;
}

function renderWorkbenchCharts(report) {
  const charts = report.charts || [];
  if (!charts.length) return '<p class="workbench-contrast-empty">当前没有通过同传感器、同指标、同单位门禁的图表序列。</p>';
  return charts.map((chart) => {
    const maximum = Math.max(...chart.points.map((item) => Math.abs(item.value)), 1e-12);
    const points = chart.points.map((point) => {
      const width = Math.max(3, Math.abs(point.value) / maximum * 100);
      return `<div class="workbench-chart-point" title="${escapeHtml(point.label)} · ${formatNumber(point.value, 4)} ${escapeHtml(chart.unit)}"><span>[${escapeHtml(point.citation_id)}] ${escapeHtml(point.label)}</span><div><i class="${point.value < 0 ? "negative" : ""}" style="width:${width.toFixed(2)}%"></i></div><b>${formatNumber(point.value, 4)} ${escapeHtml(chart.unit)}</b></div>`;
    }).join("");
    return `<article class="workbench-chart" data-comparability="${escapeHtml(chart.comparability)}"><header><div><b>${escapeHtml(chart.metric_label)}</b><small>${escapeHtml(SENSOR_LABELS[chart.sensor] || chart.sensor)} · ${escapeHtml(chart.metric_key)}</small></div><span>${chart.comparability === "direct" ? "直接对照" : "受限对照"}</span></header>${points}</article>`;
  }).join("");
}

function renderWorkbenchReport(report) {
  state.activeWorkbenchReport = report;
  elements.agentPlaceholder.hidden = true;
  elements.agentResponse.hidden = false;
  elements.workbenchAnalysisStatus.textContent = WORKBENCH_STATUS_LABELS[report.analysis_status] || report.analysis_status;
  elements.workbenchReportQuestion.textContent = report.question;
  elements.workbenchReportConfidence.textContent = `证据 ${confidenceText(report.quality.overall_confidence)}`;
  elements.workbenchQuality.innerHTML = `
    <div><span>高质量记录</span><b>${report.quality.high_count}</b></div>
    <div><span>中 / 低质量</span><b>${report.quality.medium_count} / ${report.quality.low_count}</b></div>
    <div><span>直接对照组</span><b>${report.quality.direct_comparison_count}</b></div>
    <div><span>受限对照组</span><b>${report.quality.limited_comparison_count}</b></div>`;
  elements.workbenchAudits.innerHTML = report.audits.map((audit) => `
    <article class="workbench-audit-card">
      <header><h4>[${escapeHtml(workbenchCitation(report, audit.recording_id))}] ${escapeHtml(audit.label)}</h4><span>${escapeHtml(confidenceText(audit.confidence))} · ${audit.quality_score}/100</span></header>
      <div class="workbench-audit-meta"><span>${escapeHtml(SENSOR_LABELS[audit.sensor] || audit.sensor)}</span><span>${escapeHtml(audit.source)}</span><span>${audit.sample_count} samples</span><span>${formatNumber(audit.duration_s, 2)} s</span><span>${formatNumber(audit.sampling_rate_hz, 2)} Hz</span><span>${escapeHtml(audit.analyzer_id)} ${escapeHtml(audit.analyzer_version)}</span></div>
      ${Object.keys(audit.source_details || {}).length ? `<div class="workbench-source-line" title="${escapeHtml(Object.entries(audit.source_details).map(([key, value]) => `${key}: ${value}`).join(" · "))}">${escapeHtml(Object.entries(audit.source_details).map(([key, value]) => `${key}: ${value}`).join(" · "))}</div>` : ""}
      <div class="workbench-metric-pills">${audit.metrics.length ? audit.metrics.map((metric) => `<span title="${escapeHtml(metric.key)}">${escapeHtml(metric.label)} ${formatNumber(metric.value, 4)} ${escapeHtml(metric.unit || "")}</span>`).join("") : "<span>无注册数值指标</span>"}</div>
      ${audit.warnings.length ? `<p class="workbench-audit-warning">${escapeHtml(audit.warnings.join("；"))}</p>` : ""}
    </article>`).join("");
  elements.workbenchComparability.innerHTML = report.comparability.map((group) => `
    <div class="workbench-comparison-row" data-status="${escapeHtml(group.status)}">
      <b>${escapeHtml(group.sensor ? SENSOR_LABELS[group.sensor] || group.sensor : "跨传感器组合解释")} · ${escapeHtml(WORKBENCH_COMPARISON_LABELS[group.status] || group.status)}</b>
      <p>${escapeHtml(group.reasons.join("；"))}${group.shared_metric_keys.length ? ` · 共有指标：${escapeHtml(group.shared_metric_keys.join("、"))}` : ""}</p>
    </div>`).join("");
  elements.workbenchMatrix.innerHTML = renderWorkbenchMatrix(report);
  elements.workbenchContrasts.innerHTML = report.contrasts.length
    ? `<table><thead><tr><th>指标</th><th>记录 A</th><th>记录 B</th><th>差值 B−A</th><th>相对变化</th></tr></thead><tbody>${report.contrasts.map((contrast) => `
      <tr><td>${escapeHtml(contrast.metric_label)} (${escapeHtml(contrast.unit)})</td><td title="${escapeHtml(workbenchRecordLabel(report, contrast.left_recording_id))}">[${escapeHtml(workbenchCitation(report, contrast.left_recording_id))}] ${formatNumber(contrast.left_value, 4)}</td><td title="${escapeHtml(workbenchRecordLabel(report, contrast.right_recording_id))}">[${escapeHtml(workbenchCitation(report, contrast.right_recording_id))}] ${formatNumber(contrast.right_value, 4)}</td><td>${formatNumber(contrast.absolute_delta, 4)}</td><td>${contrast.relative_delta_percent == null ? "—" : `${formatNumber(contrast.relative_delta_percent, 1)}%`}</td></tr>`).join("")}</tbody></table>`
    : '<p class="workbench-contrast-empty">当前没有同传感器、同指标、同单位的成对记录；不会强行生成数值差异。</p>';
  elements.workbenchCharts.innerHTML = renderWorkbenchCharts(report);
  elements.workbenchAnswer.innerHTML = renderRichText(report.answer);
  elements.workbenchBoundaries.innerHTML = report.boundaries.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  elements.workbenchUserNote.value = report.user_note || "";
  elements.modelBadge.textContent = compactModelName(report.model);
}

async function saveWorkbenchNote() {
  const report = state.activeWorkbenchReport;
  if (!report || state.busy) return;
  setBusy(true, elements.saveWorkbenchNoteButton, "正在保存…");
  try {
    const response = await fetch(`/api/v2/evidence-workbench/reports/${encodeURIComponent(report.report_id)}/note`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_note: elements.workbenchUserNote.value.trim() }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    renderWorkbenchReport(await response.json());
    await loadWorkbenchReports();
    showToast("实验者注释已保存");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setBusy(false, elements.saveWorkbenchNoteButton, "保存注释");
  }
}

function exportWorkbenchReport() {
  const report = state.activeWorkbenchReport;
  if (!report) return;
  const link = document.createElement("a");
  link.href = `/api/v2/evidence-workbench/reports/${encodeURIComponent(report.report_id)}/export`;
  link.download = `pocketlab-${report.report_id}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function runAdvancedAgent() {
  if (state.busy || state.selectedIds.size === 0) return;
  state.busy = true;
  updateAdvancedButton();
  elements.agentResponse.hidden = true;
  elements.agentPlaceholder.hidden = false;
  elements.agentPlaceholder.innerHTML = '<div class="agent-loading">证据工作台正在核对分析器指标、质量边界与可比性…</div>';
  try {
    const response = await fetch("/api/v2/evidence-workbench/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: elements.questionInput.value.trim(), recording_ids: [...state.selectedIds] }),
    });
    if (!response.ok) throw new Error(await readApiError(response));
    const data = await response.json();
    renderWorkbenchReport(data);
    await loadWorkbenchReports();
    await loadAgentRuns(true);
  } catch (error) {
    elements.agentPlaceholder.hidden = false;
    elements.agentPlaceholder.innerHTML = `<div><b>运行未完成</b><p>${escapeHtml(error.message)}</p></div>`;
    showToast(error.message, true);
  } finally {
    state.busy = false;
    updateAdvancedButton();
  }
}

function updateSubmitButton() {
  const task = state.diagnosticCase?.current_task;
  const hasTask = Boolean(task);
  const analyzerReady = task ? taskSensorDetails(task).analyzerReady : false;
  const phoneReady = task
    ? analyzerReady && probeMatchesTask(state.phyphoxProbe, task)
    : false;
  const hasSource = state.measurementMode === "simulation" || Boolean(state.pendingFile);
  elements.submitTaskButton.disabled = state.busy || !hasTask || !analyzerReady || !hasSource || state.measurementMode === "mobile";
  elements.publicDiagnosticRunButton.disabled = state.busy || !hasTask || !analyzerReady;
  elements.probePhyphoxButton.disabled = state.busy || !elements.phyphoxBaseUrl.value.trim();
  elements.capturePhyphoxButton.disabled = state.busy || !hasTask || !phoneReady;
  renderDiagnosticRetry();
}

function updateAdvancedButton() {
  elements.runAgentButton.disabled = state.busy || state.selectedIds.size === 0 || elements.questionInput.value.trim().length < 3;
}

function setBusy(busy, button, label) {
  state.busy = busy;
  if (button) {
    button.disabled = busy;
    const span = button.querySelector("span");
    if (span) span.textContent = label;
    else button.textContent = label;
  }
  updateSubmitButton();
  updateAdvancedButton();
  if (elements.publicReplayImportButton) updatePublicReplayAvailability();
  if (elements.publicLightRunButton) updatePublicLightAvailability();
  if (elements.publicPressureRunButton) updatePublicPressureAvailability();
  if (elements.publicSensorRunButton) updatePublicSensorAvailability();
}

async function parseDatasetFile(file) {
  const text = await file.text();
  const parsed = file.name.toLowerCase().endsWith(".json") ? parseJsonDataset(text) : parseCsvDataset(text);
  validateSamples(parsed.samples);
  return parsed;
}

function parseJsonDataset(text) {
  const value = JSON.parse(text);
  if (Array.isArray(value)) return { samples: normalizeSamples(value) };
  if (value && Array.isArray(value.samples)) return { label: value.label, notes: value.notes, samples: normalizeSamples(value.samples) };
  throw new Error("JSON 需要是采样点数组，或包含 samples 数组的对象。");
}

function parseCsvDataset(text) {
  const cleanText = text.replace(/^\uFEFF/, "").trim();
  if (!cleanText) throw new Error("CSV 文件为空。");
  const firstLine = cleanText.split(/\r?\n/, 1)[0];
  const delimiter = firstLine.includes("\t") ? "\t" : firstLine.includes(";") ? ";" : ",";
  const rows = cleanText.split(/\r?\n/).filter(Boolean).map((line) => splitCsvLine(line, delimiter));
  const headers = rows[0].map(normalizeHeader);
  const timeIndex = findHeader(headers, ["timestampms", "timestamp", "time", "t", "secondselapsed", "elapsedtime"]);
  const indexes = [timeIndex, findAxisHeader(headers, "x"), findAxisHeader(headers, "y"), findAxisHeader(headers, "z")];
  if (indexes.some((index) => index < 0)) throw new Error("没有识别到时间、x、y、z 四列。");
  const milliseconds = headers[timeIndex].includes("ms") || headers[timeIndex].includes("millisecond");
  const samples = rows.slice(1).map((row) => ({
    timestamp_ms: Number(row[indexes[0]]) * (milliseconds ? 1 : 1000),
    x: Number(row[indexes[1]]), y: Number(row[indexes[2]]), z: Number(row[indexes[3]]),
  })).filter((item) => Object.values(item).every(Number.isFinite));
  return { samples };
}

function splitCsvLine(line, delimiter) {
  const cells = []; let current = ""; let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') { current += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === delimiter && !quoted) { cells.push(current.trim()); current = ""; }
    else current += character;
  }
  cells.push(current.trim()); return cells;
}

function normalizeHeader(header) { return String(header).toLowerCase().replace(/[\s_()[\]{}\/\\.-]/g, ""); }
function findHeader(headers, names) {
  for (const name of names) { const exact = headers.indexOf(name); if (exact >= 0) return exact; }
  return headers.findIndex((header) => names.some((name) => header.includes(name)));
}
function findAxisHeader(headers, axis) {
  const names = [axis, `a${axis}`, `acceleration${axis}`, `linearacceleration${axis}`];
  const exact = headers.findIndex((header) => names.includes(header));
  return exact >= 0 ? exact : headers.findIndex((header) => header.includes("acceleration") && header.endsWith(axis));
}
function normalizeSamples(samples) {
  return samples.map((sample, index) => ({
    timestamp_ms: Number(sample.timestamp_ms ?? sample.timestamp ?? sample.time_ms ?? sample.time ?? index),
    x: Number(sample.x ?? sample.acceleration_x ?? sample.ax),
    y: Number(sample.y ?? sample.acceleration_y ?? sample.ay),
    z: Number(sample.z ?? sample.acceleration_z ?? sample.az),
  }));
}
function validateSamples(samples) {
  if (!Array.isArray(samples) || samples.length < 64) throw new Error("至少需要 64 个有效采样点。");
  if (samples.length > 60000) throw new Error("采样点不能超过 60,000 个。");
  samples.forEach((sample, index) => {
    if (![sample.timestamp_ms, sample.x, sample.y, sample.z].every(Number.isFinite)) throw new Error(`第 ${index + 1} 个采样点包含无效数字。`);
    if (index && sample.timestamp_ms <= samples[index - 1].timestamp_ms) throw new Error("时间戳必须严格递增。");
  });
}

function populateSimulationProfiles() {
  const groups = new Map();
  Object.entries(SIMULATION_PROFILES).forEach(([key, profile]) => {
    if (!groups.has(profile.group)) groups.set(profile.group, []);
    groups.get(profile.group).push([key, profile]);
  });
  elements.simulationProfile.innerHTML = [...groups.entries()].map(([group, profiles]) => `
    <optgroup label="${escapeHtml(group)}">
      ${profiles.map(([key, profile]) => `<option value="${escapeHtml(key)}">${escapeHtml(profile.label)}</option>`).join("")}
    </optgroup>`).join("");
  updateSimulationProfile();
}

function updateSimulationProfile() {
  const profile = SIMULATION_PROFILES[elements.simulationProfile.value];
  if (!profile) return;
  elements.simulationProfileTitle.textContent = profile.label;
  elements.simulationProfileDescription.textContent = profile.description;
  const frequencies = [...new Set(profile.components.map((item) => `${item.frequency} Hz`))];
  elements.simulationProfileTags.innerHTML = [
    `${profile.duration} 秒`, "100 Hz 采样", ...frequencies, `噪声 ${profile.noise}`,
  ].map((item) => `<span>${escapeHtml(item)}</span>`).join("");
  if (state.diagnosticCase?.current_task) {
    elements.measurementLabelInput.value = `${profile.label} · ${state.diagnosticCase.current_task.task_id}`;
  }
}

function suggestProfileForTask(diagnosticCase, task) {
  const caseText = [diagnosticCase?.title, diagnosticCase?.problem_statement]
    .filter(Boolean).join(" ");
  const taskText = [task?.title, task?.instruction, task?.variable_to_change]
    .filter(Boolean).join(" ");
  const rules = [
    [/洗衣机|脱水/, /均匀|重新.*衣物|衣物分布|消除偏载/, "washing_balanced", "washing_unbalanced"],
    [/风扇/, /软垫|隔振/, "fan_isolated", "fan_direct"],
    [/音箱|扬声器|扫频/, /17\s*Hz|非共振/, "speaker_off_resonance", "speaker_resonance"],
    [/冰箱|压缩机/, /停机|停止/, "refrigerator_off", "refrigerator_on"],
    [/外壳|螺丝|面板/, /紧固|拧紧/, "panel_tightened", "panel_loose"],
    [/脚步|楼板|行走/, /远离|远处/, "footsteps_far", "footsteps_near"],
  ];
  for (const [casePattern, controlPattern, control, baseline] of rules) {
    if (casePattern.test(caseText)) return controlPattern.test(taskText) ? control : baseline;
  }
  if (/过短|低质量|噪声/.test(`${caseText} ${taskText}`)) return "low_quality_short";
  if (/46\s*Hz|混叠|奈奎斯特/.test(`${caseText} ${taskText}`)) return "alias_risk";
  return null;
}

function generateProfileSamples(profile) {
  const rate = 100;
  const count = Math.round(profile.duration * rate);
  return Array.from({ length: count }, (_, index) => {
    const time = index / rate;
    const envelope = profile.envelope === "ramp" ? Math.min(1, 0.3 + time / 2.4) : 1;
    const values = { x: 0, y: 0, z: 9.81 };
    profile.components.forEach((component) => {
      values[component.axis] += envelope * component.amplitude * Math.sin(
        2 * Math.PI * component.frequency * time + (component.phase || 0),
      );
    });
    ["x", "y", "z"].forEach((axis, axisIndex) => {
      values[axis] += profile.noise * deterministicNoise(index, axisIndex + 1);
    });
    return { timestamp_ms: time * 1000, ...values };
  });
}

function deterministicNoise(index, seed) {
  const raw = Math.sin((index + 1) * (12.9898 + seed * 7.233)) * 43758.5453;
  return (raw - Math.floor(raw) - 0.5) * 2;
}

function drawSignalChart(samples) {
  const canvas = elements.signalChart; const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2); canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); const width = rect.width; const height = rect.height;
  ctx.clearRect(0, 0, width, height); drawGrid(ctx, width, height);
  const stride = Math.max(1, Math.ceil(samples.length / Math.max(360, width)));
  const plotted = samples.filter((_, index) => index % stride === 0 || index === samples.length - 1);
  const channels = plotted[0]?.values && typeof plotted[0].values === "object"
    ? Object.keys(plotted[0].values).slice(0, 3)
    : ["x", "y", "z"];
  const readValue = (item, channel) => item.values ? item.values[channel] : item[channel];
  const values = plotted.flatMap((item) => channels.map((channel) => readValue(item, channel))); let min = Math.min(...values); let max = Math.max(...values);
  const range = Math.max(max - min, 0.01); min -= range * 0.07; max += range * 0.07;
  const first = plotted[0].timestamp_ms; const timeRange = Math.max(plotted.at(-1).timestamp_ms - first, 1);
  const palette = ["#5ddce6", "#ae95ff", "#76f4c3"];
  channels.forEach((axis, axisIndex) => {
    const color = palette[axisIndex];
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.25; ctx.globalAlpha = 0.86;
    plotted.forEach((item, index) => {
      const x = 38 + ((item.timestamp_ms - first) / timeRange) * (width - 52);
      const y = 17 + (1 - (readValue(item, axis) - min) / (max - min)) * (height - 42);
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }); ctx.stroke();
  });
  ctx.globalAlpha = 1; elements.chartEmpty.hidden = true;
}

function drawEmptyChart() {
  if (!elements.signalChart) return;
  const rect = elements.signalChart.parentElement.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2); elements.signalChart.width = rect.width * dpr; elements.signalChart.height = rect.height * dpr;
  const ctx = elements.signalChart.getContext("2d"); ctx.scale(dpr, dpr); drawGrid(ctx, rect.width, rect.height);
  elements.chartEmpty.hidden = false;
}
function drawGrid(ctx, width, height) {
  ctx.clearRect(0, 0, width, height); ctx.strokeStyle = "rgba(211,239,229,.065)"; ctx.lineWidth = 1;
  for (let i = 1; i < 6; i += 1) { ctx.beginPath(); ctx.moveTo(0, height * i / 6); ctx.lineTo(width, height * i / 6); ctx.stroke(); }
  for (let i = 1; i < 9; i += 1) { ctx.beginPath(); ctx.moveTo(width * i / 9, 0); ctx.lineTo(width * i / 9, height); ctx.stroke(); }
}

function renderRichText(text) {
  const lines = String(text || "").split(/\r?\n/);
  const output = [];
  let listType = null;
  let tableRows = [];
  const closeList = () => {
    if (listType) { output.push(`</${listType}>`); listType = null; }
  };
  const flushTable = () => {
    if (!tableRows.length) return;
    const [headers, ...rows] = tableRows;
    output.push(`<div class="rich-table-wrap"><table class="rich-table"><thead><tr>${headers.map((cell) => `<th>${inlineMarkup(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkup(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
    tableRows = [];
  };
  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) { closeList(); flushTable(); return; }
    if (/^\|.*\|$/.test(line)) {
      closeList();
      const cells = line.slice(1, -1).split("|").map((cell) => cell.trim());
      if (!cells.every((cell) => /^:?-{3,}:?$/.test(cell))) tableRows.push(cells);
      return;
    }
    flushTable();
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    const namedHeading = line.match(/^\*\*(观察|当前判断|判断|下一步实验|注意事项|证据质量|本轮结论)\*\*[：:]?$/);
    if (heading || namedHeading) {
      closeList();
      const content = heading ? heading[2] : namedHeading[1];
      const tag = heading && heading[1].length <= 2 ? "h3" : "h4";
      output.push(`<${tag}>${inlineMarkup(content)}</${tag}>`);
      return;
    }
    const unordered = line.match(/^[-*•]\s+(.+)$/);
    const ordered = line.match(/^\d+[.)、]\s*(.+)$/);
    if (unordered || ordered) {
      const wanted = unordered ? "ul" : "ol";
      if (listType !== wanted) { closeList(); listType = wanted; output.push(`<${wanted}>`); }
      output.push(`<li>${inlineMarkup((unordered || ordered)[1])}</li>`);
      return;
    }
    closeList();
    output.push(`<p>${inlineMarkup(line.replace(/^#{1,4}\s*/, ""))}</p>`);
  });
  closeList(); flushTable();
  return output.join("") || "<p>Agent 没有返回文本。</p>";
}
function inlineMarkup(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}
async function readApiError(response) {
  if (response.status === 401) {
    window.location.replace("/login");
    return "登录状态已失效，请重新登录。";
  }
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (data.detail && typeof data.detail.message === "string") {
      return data.detail.code ? `${data.detail.message} [${data.detail.code}]` : data.detail.message;
    }
    return JSON.stringify(data.detail);
  }
  catch (error) { return `请求失败（HTTP ${response.status}）`; }
}
function getActiveSession() { return state.sessions.find((item) => item.session_id === state.activeId) || null; }
function datasetDuration(samples) { return samples.length > 1 ? (samples.at(-1).timestamp_ms - samples[0].timestamp_ms) / 1000 : 0; }
function hypothesisStatusText(status) { return ({ unverified: "待验证", supported: "证据增强", weakened: "证据削弱", inconclusive: "证据不足" })[status] || status; }
function caseStatusText(status) { return ({ planning: "规划中", collecting: "进行中", awaiting_user_decision: "等待继续/收手选择", completed_descriptive: "已有探索结论", completed_with_conclusion: "已有结论", completed_inconclusive: "证据不足" })[status] || status; }
function confidenceText(value) { return ({ high: "高可信", medium: "中等可信", low: "低可信" })[value] || "未知"; }
function formatNumber(value, digits) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }
function formatDateTime(value) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
}
function compactModelName(model) { return String(model || "MODEL READY").split("/").at(-1).toUpperCase().slice(0, 22); }
function escapeHtml(value) { return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;"); }
function showToast(message, error = false) {
  clearTimeout(toastTimer); elements.toast.textContent = message; elements.toast.classList.toggle("error", error); elements.toast.classList.add("show");
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 4000);
}
function debounce(callback, delay) { let timer; return (...args) => { clearTimeout(timer); timer = setTimeout(() => callback(...args), delay); }; }
