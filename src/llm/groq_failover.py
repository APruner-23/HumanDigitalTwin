from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from langchain_groq import ChatGroq


class GroqFailover:
    """
    Wrapper per ChatGroq con switch automatico tra chiavi API al raggiungimento del rate limit.
    Supporta:
    - invoke(...)
    - with_structured_output(...).invoke(...)
    - bind_tools(...).invoke(...)
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        api_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model = model_name
        self.temperature = temperature
        self.kwargs = kwargs

        self.api_keys = api_keys or self._load_keys_from_env()
        if not self.api_keys:
            raise ValueError("Nessuna chiave API Groq trovata.")

        self.current_key_index = 0
        self._client = self._build_client(self.api_keys[self.current_key_index])

    def _load_keys_from_env(self) -> List[str]:
        keys = [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY2"),
            os.getenv("GROQ_API_KEY3"),
        ]
        return [k.strip() for k in keys if k and k.strip()]

    def _build_client(self, api_key: str) -> ChatGroq:
        return ChatGroq(
            groq_api_key=api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            **self.kwargs,
        )

    def _is_rotation_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()

        patterns = [
            "rate limit",
            "rate_limit",
            "rate_limit_exceeded",
            "quota",
            "quota exceeded",
            "insufficient_quota",
            "tokens per day",
            "daily limit",
            "usage limit",
            "resource exhausted",
            "too many requests",
            "429",
        ]

        return any(p in msg for p in patterns)

    def _rotate_key(self) -> bool:
        if self.current_key_index >= len(self.api_keys) - 1:
            return False

        self.current_key_index += 1
        self._client = self._build_client(self.api_keys[self.current_key_index])
        print(f"[GROQ FAILOVER] Switched to {self.active_key_label()}")
        return True

    def _run_with_failover(self, fn: Callable[[ChatGroq], Any]) -> Any:
        last_exc: Optional[Exception] = None

        while True:
            try:
                return fn(self._client)
            except Exception as exc:
                last_exc = exc

                if not self._is_rotation_error(exc):
                    raise

                rotated = self._rotate_key()
                if not rotated:
                    raise RuntimeError(
                        f"Tutte le chiavi API Groq sono esaurite o limitate. Ultimo errore: {exc}"
                    ) from exc

    def invoke(self, messages: Any) -> Any:
        return self._run_with_failover(lambda client: client.invoke(messages))

    def with_structured_output(self, schema: Any, method: str = "json_mode") -> "GroqStructuredFailover":
        return GroqStructuredFailover(self, schema=schema, method=method)

    def bind_tools(self, tools: Any) -> "GroqToolsFailover":
        return GroqToolsFailover(self, tools=tools)

    def active_key_label(self) -> str:
        return f"GROQ_API_KEY{'' if self.current_key_index == 0 else self.current_key_index + 1}"


class GroqStructuredFailover:
    def __init__(self, parent: GroqFailover, schema: Any, method: str = "json_mode"):
        self.parent = parent
        self.schema = schema
        self.method = method

    def invoke(self, messages: Any) -> Any:
        return self.parent._run_with_failover(
            lambda client: client.with_structured_output(self.schema, method=self.method).invoke(messages)
        )


class GroqToolsFailover:
    def __init__(self, parent: GroqFailover, tools: Any):
        self.parent = parent
        self.tools = tools

    def invoke(self, messages: Any) -> Any:
        return self.parent._run_with_failover(
            lambda client: client.bind_tools(self.tools).invoke(messages)
        )