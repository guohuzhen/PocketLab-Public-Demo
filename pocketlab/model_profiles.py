from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from pocketlab.auth import get_current_user_id
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.provider_compat import (
    CompilerTransportPreference,
    ModelIntegrationStatus,
    ReasoningStrategy,
    normalize_reasoning_strategy,
    pocketlab_model_integration,
    provider_reasoning_directive,
)

MODEL_SECRET_SERVICE = "PocketLab Agent Model Profiles"
ENVIRONMENT_PROFILE_ID = "environment"


class ModelProfileError(RuntimeError):
    pass


class ModelProfileNotFound(ModelProfileError):
    pass


class ModelSecretUnavailable(ModelProfileError):
    pass


class ModelCredentialVault(Protocol):
    def get(self, reference: str) -> str | None: ...

    def set(self, reference: str, secret: str) -> None: ...

    def delete(self, reference: str) -> None: ...


class KeyringCredentialVault:
    """Store provider keys outside SQLite using the operating-system keyring."""

    def get(self, reference: str) -> str | None:
        try:
            import keyring

            return keyring.get_password(MODEL_SECRET_SERVICE, reference)
        except Exception as exc:  # pragma: no cover - backend-specific
            raise ModelSecretUnavailable(
                "系统凭据存储不可用；请修复系统 Keyring，或继续使用 .env.local。"
            ) from exc

    def set(self, reference: str, secret: str) -> None:
        try:
            import keyring

            keyring.set_password(MODEL_SECRET_SERVICE, reference, secret)
        except Exception as exc:  # pragma: no cover - backend-specific
            raise ModelSecretUnavailable(
                "无法写入系统凭据存储；API Key 没有保存。"
            ) from exc

    def delete(self, reference: str) -> None:
        try:
            import keyring
            from keyring.errors import PasswordDeleteError

            try:
                keyring.delete_password(MODEL_SECRET_SERVICE, reference)
            except PasswordDeleteError:
                return
        except ModelSecretUnavailable:
            raise
        except Exception as exc:  # pragma: no cover - backend-specific
            raise ModelSecretUnavailable("无法从系统凭据存储移除 API Key。") from exc


class MemoryCredentialVault:
    """Process-only vault for hermetic tests; never selected in normal operation."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, reference: str) -> str | None:
        return self._values.get(reference)

    def set(self, reference: str, secret: str) -> None:
        self._values[reference] = secret

    def delete(self, reference: str) -> None:
        self._values.pop(reference, None)


def _default_vault() -> ModelCredentialVault:
    backend = os.getenv("POCKETLAB_SECRET_BACKEND", "keyring").strip().lower()
    if backend == "memory":
        testing = os.getenv("POCKETLAB_TESTING", "").strip().lower()
        if testing not in {"1", "true", "yes", "on"}:
            raise RuntimeError("memory secret backend is restricted to PocketLab tests")
        return MemoryCredentialVault()
    if backend != "keyring":
        raise RuntimeError("POCKETLAB_SECRET_BACKEND must be keyring or test-only memory")
    return KeyringCredentialVault()


def normalize_model_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("模型 Base URL 必须是无凭据、无查询参数的 http(s) 地址。")
    hostname = parsed.hostname.rstrip(".").casefold()
    is_localhost = hostname == "localhost"
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if parsed.scheme == "http" and not (
        is_localhost or (address is not None and address.is_loopback)
    ):
        raise ValueError("远程模型接口必须使用 HTTPS；HTTP 仅允许本机模型。")
    if address is not None and not (address.is_global or address.is_loopback):
        raise ValueError("模型接口不能指向私有、链路本地或保留 IP 地址。")
    return normalized


def agent_model_name_incompatibility(model_name: str) -> str | None:
    """Identify high-confidence model families that do not use chat completions."""

    normalized = model_name.strip().casefold()
    tokens = {
        token
        for token in re.split(r"[/_.:\-]+", normalized)
        if token
    }
    if any(token.startswith("cogview") for token in tokens):
        family = "CogView 图像生成"
    elif any(token.startswith("cogvideo") for token in tokens):
        family = "CogVideo 视频生成"
    elif "embedding" in tokens or "embeddings" in tokens:
        family = "Embedding 向量"
    elif "whisper" in tokens:
        family = "Whisper 语音转写"
    elif "tts" in tokens:
        family = "TTS 语音合成"
    elif "rerank" in tokens or "reranker" in tokens:
        family = "Rerank 排序"
    else:
        return None
    return (
        f"{family}模型不能作为 PocketLab Agent 基模；"
        "请选择支持 OpenAI-compatible Chat Completions 的语言模型。"
    )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ModelProfileCreate(_StrictModel):
    name: str = Field(min_length=1, max_length=80)
    base_url: str = Field(min_length=8, max_length=500)
    model_name: str = Field(min_length=1, max_length=200)
    api_key: SecretStr = Field(min_length=1, max_length=4096)
    reasoning_strategy: ReasoningStrategy = "auto"
    input_cost_per_million: float | None = Field(default=None, ge=0, le=100_000)
    output_cost_per_million: float | None = Field(default=None, ge=0, le=100_000)
    make_default: bool = True

    @field_validator("base_url")
    @classmethod
    def base_url_is_safe(cls, value: str) -> str:
        return normalize_model_base_url(value)

    @field_validator("model_name")
    @classmethod
    def model_name_uses_chat_completions(cls, value: str) -> str:
        issue = agent_model_name_incompatibility(value)
        if issue is not None:
            raise ValueError(issue)
        return value


class ModelProfileUpdate(_StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    base_url: str | None = Field(default=None, min_length=8, max_length=500)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    api_key: SecretStr | None = Field(default=None, min_length=1, max_length=4096)
    reasoning_strategy: ReasoningStrategy | None = None
    input_cost_per_million: float | None = Field(default=None, ge=0, le=100_000)
    output_cost_per_million: float | None = Field(default=None, ge=0, le=100_000)

    @field_validator("base_url")
    @classmethod
    def base_url_is_safe(cls, value: str | None) -> str | None:
        return normalize_model_base_url(value) if value is not None else None

    @field_validator("model_name")
    @classmethod
    def model_name_uses_chat_completions(cls, value: str | None) -> str | None:
        issue = agent_model_name_incompatibility(value) if value is not None else None
        if issue is not None:
            raise ValueError(issue)
        return value


ModelCapabilityTier = Literal[
    "unverified",
    "unavailable",
    "text_only",
    "tool_capable",
    "exploration_compatible",
    "agent_capable",
]


class ModelCapabilityProbe(_StrictModel):
    status: ModelCapabilityTier
    text_generation: bool
    structured_json: bool
    function_tools: bool
    model_listing: bool
    evidence_workbench_ready: bool
    exploration_agent_ready: bool
    diagnostic_agent_ready: bool
    native_json_mode: bool = False
    validated_json_text: bool = False
    structured_transport: Literal[
        "native_json_mode", "validated_json_text", "none"
    ] = "none"
    tool_transport: Literal["named_function", "auto", "none"] = "none"
    probe_requests: int = Field(default=0, ge=0, le=16)
    transient_retries: int = Field(default=0, ge=0, le=4)
    latency_ms: int = Field(ge=0)
    checked_at: str
    error_codes: list[str] = Field(default_factory=list, max_length=8)


class ModelProfileSummary(_StrictModel):
    profile_id: str
    name: str
    source: Literal["environment", "user_profile"]
    base_url: str
    model_name: str
    api_key_configured: bool
    api_key_hint: str
    reasoning_strategy: ReasoningStrategy = "auto"
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    is_default: bool
    readonly: bool
    revision: int = Field(ge=1)
    probe: ModelCapabilityProbe | None = None
    created_at: str | None = None
    updated_at: str | None = None
    integration_status: ModelIntegrationStatus = "compatibility_trial"
    recommended_compiler_transport: CompilerTransportPreference = "auto"


class ModelProfileCatalog(_StrictModel):
    profiles: list[ModelProfileSummary]
    active_profile_id: str | None
    secret_backend: Literal["keyring", "environment", "unavailable"]


def model_capability_tier(
    *,
    text_generation: bool,
    structured_json: bool,
    function_tools: bool,
) -> ModelCapabilityTier:
    """Classify independent provider capabilities without hiding tool support."""

    if structured_json and function_tools:
        return "agent_capable"
    if function_tools:
        return "tool_capable"
    if structured_json:
        return "exploration_compatible"
    return "text_only" if text_generation else "unavailable"


@dataclass(frozen=True)
class ActiveModelConfiguration:
    profile_id: str
    api_key: str
    base_url: str
    model_name: str
    reasoning_strategy: ReasoningStrategy = "auto"
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None


def _first_nonempty(values: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = values.get(name, "").strip()
        if value:
            return value
    return None


def environment_model_configuration(
    values: Mapping[str, str] | None = None,
) -> ActiveModelConfiguration | None:
    source = os.environ if values is None else values
    api_key = _first_nonempty(source, "LLM_API_KEY", "PPIO_API_KEY")
    base_url = _first_nonempty(source, "LLM_BASE_URL", "PPIO_BASE_URL")
    model_name = _first_nonempty(source, "LLM_MODEL", "PPIO_MODEL")
    if not api_key or not base_url or not model_name:
        return None
    try:
        normalized_url = normalize_model_base_url(base_url)
    except ValueError:
        return None

    def optional_cost(name: str) -> float | None:
        raw = source.get(name, "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if 0 <= value <= 100_000 else None

    return ActiveModelConfiguration(
        profile_id=ENVIRONMENT_PROFILE_ID,
        api_key=api_key,
        base_url=normalized_url,
        model_name=model_name,
        reasoning_strategy=normalize_reasoning_strategy(
            source.get("LLM_REASONING_STRATEGY", "auto")
        ),
        input_cost_per_million=optional_cost("LLM_INPUT_COST_PER_MILLION"),
        output_cost_per_million=optional_cost("LLM_OUTPUT_COST_PER_MILLION"),
    )


def _api_key_hint(secret: str) -> str:
    return f"••••{secret[-4:]}" if len(secret) >= 4 else "••••"


class ModelProfileStore:
    def __init__(
        self,
        storage: SQLiteDatabase | None = None,
        vault: ModelCredentialVault | None = None,
        *,
        user_id: str | None = DEFAULT_USER_ID,
    ) -> None:
        self._database = storage or SQLiteDatabase(":memory:")
        self._vault = vault or _default_vault()
        self._user_id = user_id

    @property
    def _active_user_id(self) -> str:
        return self._user_id or get_current_user_id()

    @staticmethod
    def _secret_ref(user_id: str, profile_id: str) -> str:
        return f"{user_id}:{profile_id}"

    def list_profiles(self) -> list[ModelProfileSummary]:
        user_id = self._active_user_id
        rows = self._database.fetch_all(
            "SELECT * FROM model_profiles WHERE user_id = ? "
            "ORDER BY is_default DESC, updated_at DESC",
            (user_id,),
        )
        return [self._summary(row) for row in rows]

    def get_profile(self, profile_id: str) -> ModelProfileSummary:
        row = self._row(profile_id)
        return self._summary(row)

    def create(self, request: ModelProfileCreate) -> ModelProfileSummary:
        user_id = self._active_user_id
        profile_id = f"model_{uuid4().hex[:16]}"
        secret = request.api_key.get_secret_value()
        reference = self._secret_ref(user_id, profile_id)
        self._vault.set(reference, secret)
        now = utc_now()
        try:
            with self._database.transaction() as connection:
                if request.make_default:
                    connection.execute(
                        "UPDATE model_profiles SET is_default = 0 WHERE user_id = ?",
                        (user_id,),
                    )
                connection.execute(
                    """
                    INSERT INTO model_profiles(
                        profile_id, user_id, name, base_url, model_name,
                        api_key_hint, secret_ref, input_cost_per_million,
                        output_cost_per_million, reasoning_strategy, is_default, revision,
                        probe_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?)
                    """,
                    (
                        profile_id,
                        user_id,
                        request.name,
                        request.base_url,
                        request.model_name,
                        _api_key_hint(secret),
                        reference,
                        request.input_cost_per_million,
                        request.output_cost_per_million,
                        request.reasoning_strategy,
                        int(request.make_default),
                        now,
                        now,
                    ),
                )
        except Exception:
            self._vault.delete(reference)
            raise
        return self.get_profile(profile_id)

    def update(self, profile_id: str, request: ModelProfileUpdate) -> ModelProfileSummary:
        row = self._row(profile_id)
        user_id = self._active_user_id
        secret = request.api_key.get_secret_value() if request.api_key is not None else None
        previous_secret = self._vault.get(row["secret_ref"]) if secret is not None else None
        if secret is not None:
            self._vault.set(row["secret_ref"], secret)
        values = {
            "name": request.name if request.name is not None else row["name"],
            "base_url": request.base_url if request.base_url is not None else row["base_url"],
            "model_name": (
                request.model_name if request.model_name is not None else row["model_name"]
            ),
            "api_key_hint": _api_key_hint(secret) if secret is not None else row["api_key_hint"],
            "reasoning_strategy": (
                request.reasoning_strategy
                if request.reasoning_strategy is not None
                else row["reasoning_strategy"]
            ),
            "input_cost": (
                request.input_cost_per_million
                if "input_cost_per_million" in request.model_fields_set
                else row["input_cost_per_million"]
            ),
            "output_cost": (
                request.output_cost_per_million
                if "output_cost_per_million" in request.model_fields_set
                else row["output_cost_per_million"]
            ),
        }
        try:
            self._database.execute(
                """
                UPDATE model_profiles
                SET name = ?, base_url = ?, model_name = ?, api_key_hint = ?,
                    input_cost_per_million = ?, output_cost_per_million = ?,
                    reasoning_strategy = ?,
                    revision = revision + 1, probe_json = NULL, updated_at = ?
                WHERE profile_id = ? AND user_id = ?
                """,
                (
                    values["name"],
                    values["base_url"],
                    values["model_name"],
                    values["api_key_hint"],
                    values["input_cost"],
                    values["output_cost"],
                    values["reasoning_strategy"],
                    utc_now(),
                    profile_id,
                    user_id,
                ),
            )
        except Exception:
            if secret is not None:
                if previous_secret is None:
                    self._vault.delete(row["secret_ref"])
                else:
                    self._vault.set(row["secret_ref"], previous_secret)
            raise
        return self.get_profile(profile_id)

    def activate(self, profile_id: str) -> ModelProfileSummary:
        self._row(profile_id)
        user_id = self._active_user_id
        with self._database.transaction() as connection:
            connection.execute(
                "UPDATE model_profiles SET is_default = 0 WHERE user_id = ?",
                (user_id,),
            )
            connection.execute(
                "UPDATE model_profiles SET is_default = 1, updated_at = ? "
                "WHERE profile_id = ? AND user_id = ?",
                (utc_now(), profile_id, user_id),
            )
        return self.get_profile(profile_id)

    def activate_environment(self) -> None:
        self._database.execute(
            "UPDATE model_profiles SET is_default = 0 WHERE user_id = ?",
            (self._active_user_id,),
        )

    def delete(self, profile_id: str) -> None:
        row = self._row(profile_id)
        user_id = self._active_user_id
        was_default = bool(row["is_default"])
        previous_secret = self._vault.get(row["secret_ref"])
        self._vault.delete(row["secret_ref"])
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    "DELETE FROM model_profiles WHERE profile_id = ? AND user_id = ?",
                    (profile_id, user_id),
                )
                if was_default:
                    replacement = connection.execute(
                        "SELECT profile_id FROM model_profiles WHERE user_id = ? "
                        "ORDER BY updated_at DESC LIMIT 1",
                        (user_id,),
                    ).fetchone()
                    if replacement is not None:
                        connection.execute(
                            "UPDATE model_profiles SET is_default = 1, updated_at = ? "
                            "WHERE profile_id = ? AND user_id = ?",
                            (utc_now(), replacement["profile_id"], user_id),
                        )
        except Exception:
            if previous_secret is not None:
                self._vault.set(row["secret_ref"], previous_secret)
            raise

    def resolve(self, profile_id: str) -> ActiveModelConfiguration:
        row = self._row(profile_id)
        secret = self._vault.get(row["secret_ref"])
        if not secret:
            raise ModelSecretUnavailable(
                "该模型配置的 API Key 不在系统凭据存储中，请重新填写。"
            )
        return ActiveModelConfiguration(
            profile_id=row["profile_id"],
            api_key=secret,
            base_url=row["base_url"],
            model_name=row["model_name"],
            reasoning_strategy=normalize_reasoning_strategy(row["reasoning_strategy"]),
            input_cost_per_million=row["input_cost_per_million"],
            output_cost_per_million=row["output_cost_per_million"],
        )

    def resolve_active(self) -> ActiveModelConfiguration | None:
        row = self._database.fetch_one(
            "SELECT profile_id FROM model_profiles "
            "WHERE user_id = ? AND is_default = 1 LIMIT 1",
            (self._active_user_id,),
        )
        return self.resolve(row["profile_id"]) if row is not None else None

    def save_probe(
        self,
        profile_id: str,
        probe: ModelCapabilityProbe,
    ) -> ModelProfileSummary:
        self._row(profile_id)
        self._database.execute(
            "UPDATE model_profiles SET probe_json = ?, updated_at = ? "
            "WHERE profile_id = ? AND user_id = ?",
            (
                probe.model_dump_json(),
                utc_now(),
                profile_id,
                self._active_user_id,
            ),
        )
        return self.get_profile(profile_id)

    def catalog(self) -> ModelProfileCatalog:
        profiles = self.list_profiles()
        active_user_profile = next((item for item in profiles if item.is_default), None)
        environment = environment_model_configuration()
        if environment is not None:
            integration = pocketlab_model_integration(environment.model_name)
            profiles.append(
                ModelProfileSummary(
                    profile_id=ENVIRONMENT_PROFILE_ID,
                    name="系统环境配置",
                    source="environment",
                    base_url=environment.base_url,
                    model_name=environment.model_name,
                    reasoning_strategy=environment.reasoning_strategy,
                    api_key_configured=True,
                    api_key_hint=_api_key_hint(environment.api_key),
                    input_cost_per_million=environment.input_cost_per_million,
                    output_cost_per_million=environment.output_cost_per_million,
                    is_default=active_user_profile is None,
                    readonly=True,
                    revision=1,
                    integration_status=integration.status,
                    recommended_compiler_transport=integration.compiler_transport,
                )
            )
        backend: Literal["keyring", "environment", "unavailable"]
        if isinstance(self._vault, (KeyringCredentialVault, MemoryCredentialVault)):
            backend = "keyring"
        elif environment is not None:
            backend = "environment"
        else:
            backend = "unavailable"
        active = active_user_profile.profile_id if active_user_profile else None
        if active is None and environment is not None:
            active = ENVIRONMENT_PROFILE_ID
        return ModelProfileCatalog(
            profiles=profiles,
            active_profile_id=active,
            secret_backend=backend,
        )

    def _row(self, profile_id: str):
        row = self._database.fetch_one(
            "SELECT * FROM model_profiles WHERE profile_id = ? AND user_id = ?",
            (profile_id, self._active_user_id),
        )
        if row is None:
            raise ModelProfileNotFound("找不到当前账号的模型配置。")
        return row

    @staticmethod
    def _summary(row: object) -> ModelProfileSummary:
        probe = (
            ModelCapabilityProbe.model_validate_json(row["probe_json"])
            if row["probe_json"]
            else None
        )
        integration = pocketlab_model_integration(row["model_name"])
        return ModelProfileSummary(
            profile_id=row["profile_id"],
            name=row["name"],
            source="user_profile",
            base_url=row["base_url"],
            model_name=row["model_name"],
            reasoning_strategy=normalize_reasoning_strategy(row["reasoning_strategy"]),
            api_key_configured=True,
            api_key_hint=row["api_key_hint"],
            input_cost_per_million=row["input_cost_per_million"],
            output_cost_per_million=row["output_cost_per_million"],
            is_default=bool(row["is_default"]),
            readonly=False,
            revision=int(row["revision"]),
            probe=probe,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            integration_status=integration.status,
            recommended_compiler_transport=integration.compiler_transport,
        )


async def probe_model_compatibility(
    config: ActiveModelConfiguration,
) -> ModelCapabilityProbe:
    """Probe portable model transports without exposing credentials or provider bodies.

    OpenAI-compatible gateways do not all implement the same optional request
    fields.  PocketLab therefore records the transport that actually worked
    instead of treating rejection of one optional feature as rejection of the
    model.  Only transient connectivity/server errors receive one bounded retry.
    """

    started = time.monotonic()
    if agent_model_name_incompatibility(config.model_name) is not None:
        return ModelCapabilityProbe(
            status="unavailable",
            text_generation=False,
            structured_json=False,
            function_tools=False,
            model_listing=False,
            evidence_workbench_ready=False,
            exploration_agent_ready=False,
            diagnostic_agent_ready=False,
            latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            checked_at=datetime.now(UTC).isoformat(),
            error_codes=["model:non-chat-modality"],
        )
    errors: list[str] = []
    text_generation = False
    native_json_mode = False
    validated_json_text = False
    structured_json = False
    function_tools = False
    model_listing = False
    structured_transport: Literal[
        "native_json_mode", "validated_json_text", "none"
    ] = "none"
    tool_transport: Literal["named_function", "auto", "none"] = "none"
    probe_requests = 0
    transient_retries = 0
    transient_errors = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
        TimeoutError,
        asyncio.TimeoutError,
    )
    provider_directive = provider_reasoning_directive(
        config.base_url,
        config.model_name,
        strategy=config.reasoning_strategy,
        purpose="control",
    )

    def chat_options() -> dict[str, object]:
        return provider_directive.chat_completions_kwargs()

    async def bounded_request(label: str, operation):
        nonlocal probe_requests, transient_retries
        for attempt in range(2):
            probe_requests += 1
            try:
                return await operation()
            except transient_errors as exc:
                if attempt == 0:
                    transient_retries += 1
                    continue
                errors.append(f"{label}:{type(exc).__name__}")
                return None
            except (OpenAIError, AttributeError, IndexError, TypeError) as exc:
                errors.append(f"{label}:{type(exc).__name__}")
                return None
        return None

    def valid_ok_payload(response: object, label: str) -> bool:
        try:
            choices = response.choices
            content = choices[0].message.content if choices else None
        except (AttributeError, IndexError, TypeError):
            errors.append(f"{label}:malformed-response")
            return False
        if not content:
            errors.append(f"{label}:empty-content")
            return False
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            errors.append(f"{label}:invalid-content")
            return False
        if isinstance(payload, dict) and payload.get("ok") is True:
            return True
        errors.append(f"{label}:unexpected-payload")
        return False

    def valid_tool_call(response: object, label: str) -> bool:
        try:
            calls = response.choices[0].message.tool_calls if response.choices else None
        except (AttributeError, IndexError, TypeError):
            errors.append(f"{label}:malformed-response")
            return False
        if not calls or calls[0].function.name != "report_ready":
            errors.append(f"{label}:missing-call")
            return False
        raw_arguments = calls[0].function.arguments
        try:
            if isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
            else:
                arguments = json.loads(raw_arguments or "")
        except (json.JSONDecodeError, TypeError):
            errors.append(f"{label}:invalid-arguments")
            return False
        if isinstance(arguments, dict) and arguments.get("ready") is True:
            return True
        errors.append(f"{label}:unexpected-arguments")
        return False

    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=15.0,
        max_retries=0,
    )
    try:
        model_response = await bounded_request("models", client.models.list)
        if model_response is not None:
            model_listing = True

        response = await bounded_request(
            "text",
            lambda: client.chat.completions.create(
                **chat_options(),
                model=config.model_name,
                messages=[{"role": "user", "content": "Reply exactly POCKETLAB_OK"}],
                temperature=0,
                max_tokens=256,
            ),
        )
        if response is not None:
            try:
                text_generation = bool(
                    response.choices and response.choices[0].message.content
                )
            except (AttributeError, IndexError, TypeError):
                errors.append("text:malformed-response")
            if not text_generation:
                errors.append("text:empty-content")

        response = await bounded_request(
            "json",
            lambda: client.chat.completions.create(
                **chat_options(),
                model=config.model_name,
                messages=[{"role": "user", "content": "Return JSON with ok=true."}],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"},
            ),
        )
        if response is not None:
            native_json_mode = valid_ok_payload(response, "json")

        if not native_json_mode:
            response = await bounded_request(
                "json-text",
                lambda: client.chat.completions.create(
                    **chat_options(),
                    model=config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return only compact JSON. Do not use markdown or add "
                                "explanation. The response must be exactly one JSON object."
                            ),
                        },
                        {"role": "user", "content": "Return an object with ok=true."},
                    ],
                    temperature=0,
                    max_tokens=256,
                ),
            )
            if response is not None:
                validated_json_text = valid_ok_payload(response, "json-text")

        structured_json = native_json_mode or validated_json_text
        if native_json_mode:
            structured_transport = "native_json_mode"
        elif validated_json_text:
            structured_transport = "validated_json_text"

        tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "report_ready",
                    "description": "Report compatibility readiness.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ready": {"type": "boolean"}},
                        "required": ["ready"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        tool_messages = [
            {
                "role": "system",
                "content": "You must call report_ready exactly once. Do not answer with text.",
            },
            {"role": "user", "content": "Report ready=true now."},
        ]
        named_choice = {"type": "function", "function": {"name": "report_ready"}}
        response = await bounded_request(
            "tools-named",
            lambda: client.chat.completions.create(
                **chat_options(),
                model=config.model_name,
                messages=tool_messages,
                temperature=0,
                max_tokens=256,
                tools=tool_schema,
                tool_choice=named_choice,
            ),
        )
        if response is not None and valid_tool_call(response, "tools-named"):
            function_tools = True
            tool_transport = "named_function"

        if not function_tools:
            response = await bounded_request(
                "tools-auto",
                lambda: client.chat.completions.create(
                    **chat_options(),
                    model=config.model_name,
                    messages=tool_messages,
                    temperature=0,
                    max_tokens=256,
                    tools=tool_schema,
                    tool_choice="auto",
                ),
            )
            if response is not None and valid_tool_call(response, "tools-auto"):
                function_tools = True
                tool_transport = "auto"
    finally:
        await client.close()

    status = model_capability_tier(
        text_generation=text_generation,
        structured_json=structured_json,
        function_tools=function_tools,
    )
    return ModelCapabilityProbe(
        status=status,
        text_generation=text_generation,
        structured_json=structured_json,
        function_tools=function_tools,
        model_listing=model_listing,
        evidence_workbench_ready=text_generation or structured_json or function_tools,
        exploration_agent_ready=structured_json or function_tools,
        diagnostic_agent_ready=structured_json or function_tools,
        native_json_mode=native_json_mode,
        validated_json_text=validated_json_text,
        structured_transport=structured_transport,
        tool_transport=tool_transport,
        probe_requests=probe_requests,
        transient_retries=transient_retries,
        latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        checked_at=datetime.now(UTC).isoformat(),
        error_codes=errors[:8],
    )


model_profile_store = ModelProfileStore(database, user_id=None)
