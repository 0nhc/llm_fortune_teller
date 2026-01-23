from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI


class KimiInterface:
    """
    Thin client wrapper for Moonshot AI's Kimi OpenAI-compatible Chat Completions endpoint.

    Features:
      - Uses the OpenAI Python SDK against Kimi's API base URL
      - Maintains a local conversation history (system/user/assistant)
      - Supports web search via builtin $web_search function
      - Exposes a reset() method to clear session state

    Notes:
      - Kimi API is OpenAI-compatible but has some differences:
        * Temperature range is [0, 1] (not [0, 2])
        * When temperature=0, n must be 1
      - Web search is supported via $web_search builtin_function tool.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "kimi-k2-turbo-preview",
        temperature: float = 0.6,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
    ) -> None:
        """
        Args:
            api_key: Moonshot AI/Kimi API key.
            model_name: Kimi model identifier.
            temperature: Sampling temperature for generation (range: [0, 1]).
            max_tokens: Maximum tokens to generate for a single completion.
            system_prompt: Optional system prompt inserted at session start.
            base_url: Kimi API base URL for the OpenAI-compatible endpoint.
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)

        self._system_prompt = system_prompt
        self._messages: List[Dict[str, Any]] = []
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

    def _handle_web_search_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """
        Handle $web_search builtin_function tool call.

        For Kimi's builtin $web_search, we simply return the arguments as-is.
        The actual search is performed by Kimi's backend.

        Args:
            tool_call: The tool_call object from the API response.

        Returns:
            The tool result (arguments returned as-is for $web_search).
        """
        arguments = json.loads(tool_call.function.arguments)
        return arguments

    def ask(self, prompt_elements: List[Any], use_web_search: bool = True) -> str:
        """
        Send a user message and return the assistant reply text.

        Args:
            prompt_elements: Prompt parts to be joined into a single message.
            use_web_search: If True, enable web search via $web_search builtin_function.

        Returns:
            The assistant's message content as a string.
        """
        user_text = self._normalize_user_text(prompt_elements)
        self._messages.append({"role": "user", "content": user_text})

        # Prepare tools if web search is enabled
        tools: Optional[List[Dict[str, Any]]] = None
        if use_web_search:
            tools = [
                {
                    "type": "builtin_function",
                    "function": {
                        "name": "$web_search",
                    },
                }
            ]

        # Handle tool_calls loop for web search
        finish_reason: Optional[str] = None
        while finish_reason is None or finish_reason == "tool_calls":
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=self._messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                tools=tools,
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message

            if finish_reason == "tool_calls":
                # Add assistant message with tool_calls to history
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls or []
                        ],
                    }
                )

                # Execute each tool call
                for tool_call in message.tool_calls or []:
                    tool_call_name = tool_call.function.name

                    if tool_call_name == "$web_search":
                        tool_result = self._handle_web_search_tool_call(tool_call)
                    else:
                        # Unknown tool - return error message
                        tool_result = {
                            "error": f"Unknown tool: {tool_call_name}",
                            "arguments": json.loads(tool_call.function.arguments),
                        }

                    # Add tool result to messages
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call_name,
                            "content": json.dumps(tool_result),
                        }
                    )
            else:
                # Normal response - add to history and return
                content = message.content or ""
                self._messages.append({"role": "assistant", "content": content})
                return content

        # Fallback (should not reach here in normal flow)
        return message.content or ""
