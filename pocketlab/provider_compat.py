from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlsplit

PPIO_NON_REASONING_EXTRA_BODY: dict[str, bool] = {"enable_thinking": False}
DEEPSEEK_V4_NON_REASONING_EXTRA_BODY: dict[str, dict[str, str]] = {
    "thinking": {"type": "disabled"}
}

_NON_REASONING_HINT_HOSTS = {
    "api.ppinfra.com",
}
_DEEPSEEK_V4_THINKING_HOSTS = {
    "api.deepseek.com",
    # Verified against the user's current Parallel Computing Cloud gateway with
    # both ordinary text and a named function call. Keep this allowlist narrow:
    # an arbitrary OpenAI-compatible reseller may not forward DeepSeek fields.
    "llmapi.paratera.com",
}

ReasoningStrategy = Literal["auto", "fast", "deep"]
ReasoningPurpose = Literal["control", "analysis"]
EffectiveReasoningMode = Literal["fast", "deep", "provider_default"]
ModelIntegrationStatus = Literal[
    "tuned_flash",
    "tuned_pro",
    "compatibility_trial",
]
CompilerTransportPreference = Literal[
    "function_tool",
    "validated_json_text",
    "auto",
]


@dataclass(frozen=True)
class ProviderReasoningDirective:
    """Portable description of the reasoning request PocketLab can safely send."""

    effective_mode: EffectiveReasoningMode
    reasoning_effort: Literal["low", "high", "max"] | None = None
    extra_body: dict[str, Any] | None = None

    def model_settings_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by Agents SDK ``ModelSettings``."""

        result: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            result["reasoning"] = {"effort": self.reasoning_effort}
        if self.extra_body is not None:
            result["extra_body"] = self.extra_body
        return result

    def chat_completions_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by the OpenAI-compatible Python client."""

        result: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            result["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            result["extra_body"] = self.extra_body
        return result


@dataclass(frozen=True)
class PocketLabModelIntegration:
    """Name-scoped integration promises that PocketLab has actually exercised."""

    status: ModelIntegrationStatus
    compiler_transport: CompilerTransportPreference


def _normalized_model_key(model_name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (model_name or "").casefold())


def pocketlab_model_integration(model_name: str | None) -> PocketLabModelIntegration:
    """Return the narrow model-specific contract PocketLab is willing to claim.

    Flash has repeatedly satisfied the bounded function-tool compiler contract.
    Pro is reliable through server-validated JSON, while its extra tool turns can
    make the same compiler both slow and contract-invalid. Unknown models remain
    available for portable compatibility trials without an integration claim.
    """

    normalized = _normalized_model_key(model_name)
    if "deepseekv4flash" in normalized:
        return PocketLabModelIntegration(
            status="tuned_flash",
            compiler_transport="function_tool",
        )
    if "deepseekv4pro" in normalized:
        return PocketLabModelIntegration(
            status="tuned_pro",
            compiler_transport="validated_json_text",
        )
    return PocketLabModelIntegration(
        status="compatibility_trial",
        compiler_transport="auto",
    )

def _is_deepseek_v4_model(model_name: str | None) -> bool:
    normalized = _normalized_model_key(model_name)
    return "deepseekv4" in normalized


def normalize_reasoning_strategy(value: str | None) -> ReasoningStrategy:
    normalized = (value or "auto").strip().casefold()
    if normalized not in {"auto", "fast", "deep"}:
        raise ValueError("推理策略必须是 auto、fast 或 deep。")
    return cast(ReasoningStrategy, normalized)


def reasoning_metadata_from_model_settings(
    model_settings: object | None,
) -> tuple[EffectiveReasoningMode | None, str | None]:
    """Extract privacy-safe reasoning metadata from Agents SDK settings."""

    if model_settings is None:
        return None, None
    extra_body = getattr(model_settings, "extra_body", None)
    if isinstance(extra_body, Mapping):
        thinking = extra_body.get("thinking")
        if isinstance(thinking, Mapping):
            thinking_type = str(thinking.get("type") or "").casefold()
            if thinking_type == "enabled":
                mode: EffectiveReasoningMode | None = "deep"
            elif thinking_type == "disabled":
                mode = "fast"
            else:
                mode = None
        elif isinstance(extra_body.get("enable_thinking"), bool):
            mode = "deep" if extra_body["enable_thinking"] else "fast"
        else:
            mode = None
    else:
        mode = None

    reasoning = getattr(model_settings, "reasoning", None)
    if isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
    else:
        effort = getattr(reasoning, "effort", None)
    effort_text = str(effort) if effort is not None else None
    if mode is None and effort_text:
        mode = "deep" if effort_text in {"high", "max", "xhigh"} else "fast"
    return mode, effort_text


def provider_reasoning_directive(
    base_url: str,
    model_name: str | None = None,
    *,
    strategy: ReasoningStrategy = "auto",
    purpose: ReasoningPurpose,
) -> ProviderReasoningDirective:
    """Select a narrow, provider-safe reasoning mode for one model operation.

    Control-plane operations include routing, schema compilation and tool
    selection.  They stay fast even under ``deep`` because long hidden reasoning
    there adds latency and can make multi-turn tool replay provider-specific.
    Analysis operations contain the actual evidence interpretation and report
    synthesis.  ``auto`` requests high effort there, while ``deep`` requests the
    provider's highest verified effort.  Unknown gateways receive no private
    fields and retain their own default behaviour.
    """

    selected_strategy = normalize_reasoning_strategy(strategy)
    hostname = (urlsplit(base_url).hostname or "").casefold().rstrip(".")
    wants_deep_analysis = purpose == "analysis" and selected_strategy != "fast"

    if hostname in _NON_REASONING_HINT_HOSTS:
        if wants_deep_analysis:
            return ProviderReasoningDirective(
                effective_mode="deep",
                extra_body={"enable_thinking": True},
            )
        return ProviderReasoningDirective(
            effective_mode="fast",
            extra_body=dict(PPIO_NON_REASONING_EXTRA_BODY),
        )

    if hostname in _DEEPSEEK_V4_THINKING_HOSTS and _is_deepseek_v4_model(model_name):
        if wants_deep_analysis:
            effort: Literal["high", "max"] = (
                "max" if selected_strategy == "deep" else "high"
            )
            return ProviderReasoningDirective(
                effective_mode="deep",
                reasoning_effort=effort,
                extra_body={"thinking": {"type": "enabled"}},
            )
        return ProviderReasoningDirective(
            effective_mode="fast",
            extra_body={
                "thinking": dict(DEEPSEEK_V4_NON_REASONING_EXTRA_BODY["thinking"])
            },
        )

    if hostname == "api.openai.com" and (model_name or "").casefold().startswith("gpt-5"):
        if wants_deep_analysis:
            return ProviderReasoningDirective(
                effective_mode="deep",
                reasoning_effort="max" if selected_strategy == "deep" else "high",
            )
        return ProviderReasoningDirective(
            effective_mode="fast",
            reasoning_effort="low",
        )

    return ProviderReasoningDirective(effective_mode="provider_default")


def provider_non_reasoning_extra_body(
    base_url: str,
    model_name: str | None = None,
) -> dict[str, Any] | None:
    """Return a vendor hint only for endpoints whose contract is known to accept it.

    ``extra_body`` is not part of the portable Chat Completions contract.  Sending a
    DeepSeek/PPIO-specific field to every OpenAI-compatible gateway makes otherwise
    valid profiles fail with an unknown-parameter response.
    """

    return provider_reasoning_directive(
        base_url,
        model_name,
        strategy="fast",
        purpose="control",
    ).extra_body
