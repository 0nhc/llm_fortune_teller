from __future__ import annotations

from typing import Any, Dict, List, Optional

from anthropic import Anthropic


class ClaudeInterface:
    """
    Minimal wrapper around Anthropic's Messages API.

    Supports:
      - Maintaining a conversation history (system/user/assistant)
      - Passing a list of prompt elements (joined into a single user message)
      - Optional managed web search tool usage (when supported by the selected model/account)
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-5",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        enable_web_search_tool: bool = True,
        web_search_tool_type: str = "web_search_20250305",
    ) -> None:
        """
        Args:
            api_key: Anthropic API key.
            model_name: Claude model identifier.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate for a single response.
            system_prompt: Optional system prompt applied at the start of a session.
            enable_web_search_tool: If True, configures a managed web search tool descriptor.
            web_search_tool_type: Tool type string for the managed web search tool (versioned by Anthropic).
        """
        self._client = Anthropic(api_key=api_key)
        self._model_name = model_name
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)

        self._system_prompt = system_prompt
        self._messages: List[Dict[str, str]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

        self._web_search_tool: Optional[Dict[str, str]] = None
        if enable_web_search_tool:
            self._web_search_tool = {"name": "web_search", "type": web_search_tool_type}

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

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """
        Extract concatenated text blocks from an Anthropic Messages API response.

        Returns:
            Concatenated text if present, otherwise an empty string.
        """
        text_chunks: List[str] = []
        content = getattr(resp, "content", None) or []
        for block in content:
            if getattr(block, "type", None) == "text":
                text_val = getattr(block, "text", None)
                if text_val:
                    text_chunks.append(text_val)
        return "".join(text_chunks)

    def ask(self, prompt_elements: List[Any], use_web_search: bool = True) -> str:
        """
        Send a user message and return Claude's reply as plain text.

        Args:
            prompt_elements: A list of prompt parts that will be joined into one user message.
            use_web_search: If True, request Anthropic's managed web search tool (when configured).

        Returns:
            Claude's reply text. If no text is returned, a short error message is provided.
        """
        user_text = self._normalize_user_text(prompt_elements)
        self._messages.append({"role": "user", "content": user_text})

        kwargs: Dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": self._messages,
        }

        if use_web_search and self._web_search_tool is not None:
            kwargs["tools"] = [self._web_search_tool]
            kwargs["tool_choice"] = {"type": "auto"}

        resp = self._client.messages.create(**kwargs)
        reply_text = self._extract_text(resp)

        self._messages.append({"role": "assistant", "content": reply_text})

        return reply_text if reply_text else "[Claude error: empty response]"
