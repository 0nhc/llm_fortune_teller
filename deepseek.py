from typing import List, Any, Optional
from openai import OpenAI


class DeepSeekInterface:
    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-reasoner",
        temperature: float = 0.0,
        max_tokens: int = 12000,
        system_prompt: Optional[str] = None,
    ):
        # DeepSeek: use their base_url
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",  # or "https://api.deepseek.com/v1"
        )
        self._model_name = model_name
        # DeepSeek uses `max_tokens` on chat.completions
        self._generation_config = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # store conversation history ourselves (chat.completions doesn't have previous_response_id)
        self._system_prompt = system_prompt
        self._messages: List[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    def reset(self):
        """Clear conversation history."""
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})

    def _build_user_text(self, prompt_elements: List[Any]) -> str:
        if len(prompt_elements) == 1 and isinstance(prompt_elements[0], (dict, list)):
            # if you actually wanted structured messages here, you can handle that,
            # but for now we just stringify like your original code
            return str(prompt_elements[0])
        text_parts = [str(p) for p in prompt_elements]
        return " ".join(text_parts)

    def ask(self,
            prompt_elements: List[Any],
            use_web_search: bool = True) -> str:
        # DeepSeek doesn't support OpenAI's `web_search_preview` tool,
        # so we just ignore `use_web_search`.
        user_text = self._build_user_text(prompt_elements)

        # append user message to history
        self._messages.append({"role": "user", "content": user_text})

        # DeepSeek reasoning model:
        #   - expects max_tokens
        #   - ignores temperature/top_p/etc for deepseek-reasoner (per docs)
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=self._messages,
            max_tokens=self._generation_config["max_tokens"],
            extra_body={"thinking": {"type": "enabled"}}
            # temperature is accepted but ignored by deepseek-reasoner; safe to pass or omit
        )

        msg = response.choices[0].message

        # For deepseek-reasoner, you *also* get msg.reasoning_content if you want CoT
        # reasoning_content = getattr(msg, "reasoning_content", None)
        content = msg.content or ""

        # append assistant reply to history
        self._messages.append({"role": "assistant", "content": content})

        return content
