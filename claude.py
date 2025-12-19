# claude_interface.py
from typing import List, Any, Optional

from anthropic import Anthropic


class ClaudeInterface:
    """
    Simple wrapper around Anthropic's Messages API, with an interface
    similar to your ChatGPTInterface / GeminiInterface / DeepSeekInterface.

    - Maintains its own conversation history.
    - Exposes a `reset()` method to clear history.
    - `ask()` takes a list of prompt elements, joins them into one string,
      and (optionally) enables Claude's managed web search tool.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-5",  # Claude 4.5 Sonnet (see official docs)
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        # Anthropic Python client
        self._client = Anthropic(api_key=api_key)
        self._model_name = model_name

        # Basic generation config
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Optional system message
        self._system_prompt = system_prompt

        # Internal conversation history, Anthropic-style messages
        # Each item: {"role": "system" | "user" | "assistant", "content": str}
        self._messages: List[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

        # Pre-defined web search tool config (managed by Anthropic)
        # See: Web search tool docs in Claude 4.5
        self._web_search_tool = {
            "name": "web_search",
            "type": "web_search_20250305",
        }

    def reset(self):
        """
        Clear conversation history.
        This is equivalent to starting a completely new chat session.
        """
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})

    def _build_user_text(self, prompt_elements: List[Any]) -> str:
        """
        Convert a list of prompt elements into a single string.
        Mirrors the pattern you already use in other interfaces.
        """
        if len(prompt_elements) == 1 and isinstance(prompt_elements[0], (dict, list)):
            return str(prompt_elements[0])
        parts = [str(p) for p in prompt_elements]
        return " ".join(parts)

    def ask(
        self,
        prompt_elements: List[Any],
        use_web_search: bool = True,
    ) -> str:
        """
        Send a user message and get Claude's reply as plain text.

        - `prompt_elements`: similar to your other interfaces; they will be joined.
        - `use_web_search`: when True, enables Anthropic's managed Web Search tool.
          Claude will decide when to actually call it; you do not need to
          implement any external HTTP calls yourself.
        """
        user_text = self._build_user_text(prompt_elements)

        # Append user message to internal history
        self._messages.append({"role": "user", "content": user_text})

        # Build base arguments for Messages API
        kwargs = dict(
            model=self._model_name,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=self._messages,
        )

        # Enable Anthropic's managed Web Search tool if requested
        if use_web_search:
            kwargs["tools"] = [self._web_search_tool]
            # Let Claude automatically decide when to invoke the tool
            kwargs["tool_choice"] = {"type": "auto"}

        # Call Anthropic Messages API
        resp = self._client.messages.create(**kwargs)

        # Collect all text blocks from the response content
        text_chunks: List[str] = []
        for block in resp.content:
            # For plain conversation, we only care about text blocks
            if getattr(block, "type", None) == "text":
                # `block.text` holds the actual text
                text_val = getattr(block, "text", None)
                if text_val:
                    text_chunks.append(text_val)

        reply_text = "".join(text_chunks) if text_chunks else ""

        # Append assistant reply to history
        # (we only store the concatenated text, not the full structured blocks)
        self._messages.append({"role": "assistant", "content": reply_text})

        # Fallback in case something unexpected happens
        if not reply_text:
            return "[Claude error: no text in response]"
        return reply_text
