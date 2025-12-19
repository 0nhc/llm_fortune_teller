from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI


class QwenInterface:
    """
    Thin client wrapper for Alibaba Qwen's OpenAI-compatible Chat Completions endpoint
    (DashScope compatible-mode).

    Features:
      - Uses the OpenAI Python SDK against the DashScope-compatible base URL
      - Maintains a local conversation history (system/user/assistant)
      - Exposes a reset() method to clear session state

    Notes:
      - Web search tooling is not supported in this interface; the flag is accepted for API parity.
      - Some Qwen endpoints accept extra parameters such as "enable_thinking".
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3-max-preview",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        enable_thinking: bool = True,
    ) -> None:
        """
        Args:
            api_key: DashScope/Qwen API key.
            model_name: Qwen model identifier.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate for a single completion.
            system_prompt: Optional system prompt inserted at session start.
            base_url: DashScope compatible-mode base URL.
            enable_thinking: If True, requests the model to enable internal reasoning mode when supported.
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._enable_thinking = bool(enable_thinking)

        self._system_prompt = system_prompt
        self._messages: List[Dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def reset(self) -> None:
        """Clear conversation history and re-apply the system prompt (if provided)."""
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})

    @staticmethod
    def _normalize_user_text(prompt_elements: List[Any]) -> str:
        """
        Convert a list of prompt elements into a single user message string.

        If a single structured object (dict/list) is provided, it is stringified.
        Otherwise, elements are stringified and concatenated with spaces.
        """
        if len(prompt_elements) == 1 and isinstance(prompt_elements[0], (dict, list)):
            return str(prompt_elements[0])
        return " ".join(str(p) for p in prompt_elements)

    def ask(self, prompt_elements: List[Any], use_web_search: bool = True) -> str:
        """
        Send a user message and return the assistant reply text.

        Args:
            prompt_elements: Prompt parts to be joined into a single message.
            use_web_search: Accepted for interface compatibility, ignored (not supported here).

        Returns:
            The assistant's message content as a string.
        """
        user_text = self._normalize_user_text(prompt_elements)
        self._messages.append({"role": "user", "content": user_text})

        extra_body: Optional[Dict[str, Any]] = None
        if self._enable_thinking:
            extra_body = {"enable_thinking": True}

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            extra_body=extra_body,
        )

        msg = response.choices[0].message
        content = msg.content or ""

        self._messages.append({"role": "assistant", "content": content})
        return content
