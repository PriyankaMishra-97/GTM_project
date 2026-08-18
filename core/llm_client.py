"""LLM client hierarchy - the only place the system talks to a model.

Role in architecture:

    LLMClient (ABC)          shared, provider-agnostic behaviour:
      ├─ chat()                free text
      ├─ chat_json()           schema-constrained + ONE validated repair retry
      ├─ preflight()           startup check
      └─ _complete()           ABSTRACT - the single provider-specific hook
           │
           └─ OllamaClient    local Ollama HTTP (/api/chat)

Everything above `_complete` - determinism options, the JSON repair loop, the
salvage parser, preflight semantics - lives in the base class, so a new provider
is one subclass with one method. `gtm_copilot_gemini` is exactly that.

In:  prompt strings + optional pydantic model.
Out: str, or a validated pydantic instance; `LLMJSONError` / `LLMUnavailable` on
     failure so callers degrade gracefully instead of crashing a turn.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from core import config
from core.trace import Trace

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """The provider is unreachable, unauthenticated, or the model is missing."""


class LLMJSONError(RuntimeError):
    """The model failed twice to produce JSON matching the requested schema.

    Callers convert this into a graceful fallback (Router -> ASK; others -> an
    error recorded in the trace), never into a stack trace for the user.
    """


class LLMClient(ABC):
    """Provider-agnostic LLM facade with two model tiers."""

    #: Human-readable provider name, used in status output and error messages.
    provider: str = "abstract"

    def __init__(self, router_model: str | None = None, answer_model: str | None = None) -> None:
        self.router_model = router_model or config.ROUTER_MODEL
        self.answer_model = answer_model or config.ANSWER_MODEL

    # ------------------------------------------------------------------ #
    # provider hook - the ONLY method a subclass must implement
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Return raw text for one completion.

        `messages` is an alternating user/assistant list (the repair loop appends
        to it). `system` is passed separately because providers disagree on
        whether the system prompt is a message or a top-level field.
        """

    # ------------------------------------------------------------------ #
    # startup checks - subclasses override `available_models`
    # ------------------------------------------------------------------ #
    @abstractmethod
    def available_models(self) -> list[str]:
        """Model identifiers this client can currently reach."""

    def preflight(self) -> None:
        """Verify both tiers are usable; raise LLMUnavailable with a fix hint."""
        have = set(self.available_models())
        normalised = {m.split(":")[0] for m in have} | have
        missing = [
            m for m in (self.router_model, self.answer_model)
            if m not in have and m not in normalised
        ]
        if missing:
            raise LLMUnavailable(self._missing_model_hint(missing))

    def _missing_model_hint(self, missing: list[str]) -> str:
        return "Missing model(s): " + ", ".join(missing)

    def is_ready(self) -> bool:
        """Non-throwing preflight, for tests and the UI sidebar."""
        try:
            self.preflight()
            return True
        except LLMUnavailable:
            return False

    def status(self) -> dict[str, Any]:
        """Sidebar-friendly description of this client. Never raises."""
        info: dict[str, Any] = {
            "provider": self.provider,
            "router_model": self.router_model,
            "answer_model": self.answer_model,
            "ready": False,
            "models": [],
        }
        try:
            info["models"] = self.available_models()
            self.preflight()
            info["ready"] = True
        except LLMUnavailable as exc:
            info["error"] = str(exc)
        return info

    # ------------------------------------------------------------------ #
    # public API - shared by every provider
    # ------------------------------------------------------------------ #
    def chat(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        trace: Trace | None = None,
        stage: str | None = None,
    ) -> str:
        """Free-text completion."""
        model = model or self.answer_model
        raw = self._complete(model, system, [{"role": "user", "content": user}])
        text = raw.strip()
        if trace is not None:
            trace.add_llm_call(
                {
                    "stage": stage,
                    "model": model,
                    "system": system,
                    "user": user,
                    "attempts": [{"response": raw, "error": None}],
                    "ok": True,
                }
            )
        return text

    def chat_json(
        self,
        system: str,
        user: str,
        schema: Type[T],
        *,
        model: str | None = None,
        trace: Trace | None = None,
        stage: str | None = None,
    ) -> T:
        """Schema-constrained completion validated into `schema`.

        One repair retry: the validation error is appended to the conversation so
        the model sees exactly what it got wrong. Capped at one because a second
        recovery is rare and each attempt spends seconds of the latency budget -
        better to fail fast into the caller's fallback.
        """
        model = model or self.answer_model
        json_schema = schema.model_json_schema()
        messages: list[dict[str, str]] = [{"role": "user", "content": user}]

        attempts: list[dict[str, Any]] = []
        last_error = ""
        result: T | None = None
        for attempt in (1, 2):
            raw = self._complete(model, system, messages, json_schema=json_schema)
            try:
                result = schema.model_validate_json(extract_json(raw))
                attempts.append({"response": raw, "error": None})
                break
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                attempts.append({"response": raw, "error": last_error})
                if attempt == 2:
                    break
                messages += [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "That output was invalid. Error:\n"
                            f"{last_error}\n\n"
                            "Return ONLY valid JSON matching the schema. No prose."
                        ),
                    },
                ]

        if trace is not None:
            trace.add_llm_call(
                {
                    "stage": stage,
                    "model": model,
                    "system": system,
                    "user": user,
                    "attempts": attempts,
                    "ok": result is not None,
                }
            )

        if result is not None:
            return result
        raise LLMJSONError(f"{model} produced invalid JSON twice: {last_error}")


class OllamaClient(LLMClient):
    """Local Ollama over plain HTTP. No LangChain - every step stays inspectable."""

    provider = "ollama"

    def __init__(
        self,
        host: str | None = None,
        router_model: str | None = None,
        answer_model: str | None = None,
    ) -> None:
        super().__init__(router_model=router_model, answer_model=answer_model)
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self._session = requests.Session()

    def available_models(self) -> list[str]:
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?\n"
                f"{config.REQUIRED_ENV_HINT}"
            ) from exc
        return [m["name"] for m in response.json().get("models", [])]

    def _missing_model_hint(self, missing: list[str]) -> str:
        cmds = "\n".join(f"    ollama pull {m}" for m in missing)
        return "Missing Ollama model(s): " + ", ".join(missing) + "\nRun:\n" + cmds

    def _complete(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": config.LLM_KEEP_ALIVE,
            "options": {
                # Determinism triad: temperature 0 + fixed seed + greedy top_k.
                "temperature": config.LLM_TEMPERATURE,
                "seed": config.LLM_SEED,
                "top_k": 1,
                "top_p": 1.0,
                "num_predict": config.LLM_MAX_TOKENS,
                "num_ctx": config.LLM_NUM_CTX,
            },
        }
        if json_schema is not None:
            # Ollama's structured-output mode constrains decoding to the schema,
            # removing ~all "here is your JSON:" preamble failures.
            payload["format"] = json_schema
        try:
            response = self._session.post(
                f"{self.host}/api/chat", json=payload, timeout=config.LLM_TIMEOUT_S
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailable(f"Ollama call failed ({model}): {exc}") from exc
        return response.json().get("message", {}).get("content", "")


def extract_json(raw: str) -> str:
    """Salvage a JSON object from a response wrapped in prose or code fences.

    Structured-output modes usually make this a no-op, but smaller models still
    occasionally emit a ```json fence.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.lstrip("`")
        text = text[4:] if text.lower().startswith("json") else text
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


_DEFAULT_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide singleton (keeps one HTTP session warm)."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = OllamaClient()
    return _DEFAULT_CLIENT


def set_client(client: LLMClient) -> None:
    """Override the singleton - used by tests and by alternate front-ends."""
    global _DEFAULT_CLIENT
    _DEFAULT_CLIENT = client
