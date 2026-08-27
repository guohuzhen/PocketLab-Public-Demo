"use strict";

const authElements = {};

document.addEventListener("DOMContentLoaded", () => {
  [
    "authTitle", "authSubtitle", "loginTab", "registerTab", "loginForm", "registerForm",
    "loginUsername", "loginPassword", "registerUsername", "registerDisplayName",
    "registerPassword", "claimOption", "claimLocalData", "loginSubmit", "registerSubmit",
    "authMessage",
  ].forEach((id) => { authElements[id] = document.getElementById(id); });
  authElements.loginTab.addEventListener("click", () => switchMode("login"));
  authElements.registerTab.addEventListener("click", () => switchMode("register"));
  authElements.loginForm.addEventListener("submit", submitLogin);
  authElements.registerForm.addEventListener("submit", submitRegistration);
  document.querySelectorAll("[data-password-target]").forEach((button) => {
    button.addEventListener("click", () => togglePasswordVisibility(button));
  });
  loadRegistrationStatus();
});

function togglePasswordVisibility(button) {
  const input = document.getElementById(button.dataset.passwordTarget);
  if (!input) return;
  const showing = input.type === "password";
  input.type = showing ? "text" : "password";
  button.classList.toggle("active", showing);
  button.setAttribute("aria-pressed", String(showing));
  button.setAttribute("aria-label", showing ? "隐藏密码" : "显示密码");
  button.title = showing ? "隐藏密码" : "显示密码";
  input.focus({ preventScroll: true });
}

function switchMode(mode) {
  const registering = mode === "register";
  authElements.loginForm.hidden = registering;
  authElements.registerForm.hidden = !registering;
  authElements.loginTab.classList.toggle("active", !registering);
  authElements.registerTab.classList.toggle("active", registering);
  authElements.loginTab.setAttribute("aria-selected", String(!registering));
  authElements.registerTab.setAttribute("aria-selected", String(registering));
  authElements.authTitle.textContent = registering ? "建立你的工作区" : "欢迎回来";
  authElements.authSubtitle.textContent = registering
    ? "每个账号都有独立的案例、证据和设备设置。"
    : "登录后继续上次的实验。";
  showMessage("");
  (registering ? authElements.registerUsername : authElements.loginUsername).focus();
}

async function loadRegistrationStatus() {
  try {
    const response = await fetch("/api/v1/auth/status");
    if (!response.ok) return;
    const status = await response.json();
    authElements.claimOption.hidden = !status.legacy_data_available;
  } catch (error) {
    // Registration remains available even when the optional legacy-data hint fails.
  }
}

async function submitLogin(event) {
  event.preventDefault();
  await submitAuth(
    "/api/v1/auth/login",
    {
      username: authElements.loginUsername.value.trim(),
      password: authElements.loginPassword.value,
    },
    authElements.loginSubmit,
    "正在登录…",
  );
}

async function submitRegistration(event) {
  event.preventDefault();
  const body = {
    username: authElements.registerUsername.value.trim(),
    password: authElements.registerPassword.value,
    claim_local_data: authElements.claimLocalData.checked,
  };
  const displayName = authElements.registerDisplayName.value.trim();
  if (displayName) body.display_name = displayName;
  await submitAuth(
    "/api/v1/auth/register",
    body,
    authElements.registerSubmit,
    "正在创建账号…",
  );
}

async function submitAuth(url, body, button, busyLabel) {
  const originalLabel = button.textContent;
  button.disabled = true;
  button.textContent = busyLabel;
  showMessage("");
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await readError(response));
    const result = await response.json();
    showMessage(
      result.claimed_local_data ? "账号已创建，旧实验数据已安全迁入。" : "登录成功，正在进入工作区…",
      false,
    );
    window.location.replace("/app");
  } catch (error) {
    showMessage(error.message || "请求失败，请稍后重试。", true);
  } finally {
    button.disabled = false;
    button.textContent = originalLabel;
  }
}

async function readError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) return data.detail[0]?.msg || "输入内容不符合要求。";
    return "请求失败，请检查输入。";
  } catch (error) {
    return `请求失败（HTTP ${response.status}）`;
  }
}

function showMessage(message, error = false) {
  authElements.authMessage.textContent = message;
  authElements.authMessage.classList.toggle("error", error);
}
