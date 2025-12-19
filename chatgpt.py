from typing import List, Any, Optional
from openai import OpenAI


class ChatGPTInterface:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-5.1",
        temperature: float = 0.0,
        max_tokens: int = 2048
    ):
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        self._last_response_id: Optional[str] = None

    def reset(self):
        self._last_response_id = None

    def ask(self,
            prompt_elements: List[Any],
            use_web_search: bool = True) -> str:
        if len(prompt_elements) == 1 and isinstance(prompt_elements[0], (dict, list)):
            api_input = prompt_elements[0]
        else:
            text_parts = [str(p) for p in prompt_elements]
            api_input = " ".join(text_parts)

        kwargs = dict(
            model=self._model_name,
            reasoning={"effort": "medium"},
            input=api_input,
            max_output_tokens=self._generation_config["max_output_tokens"],
        )

        # 注意：不是所有模型都支持 web_search_preview，要看官方 docs
        if use_web_search:
            kwargs["tools"] = [{"type": "web_search_preview"}]

        if self._last_response_id is not None:
            kwargs["previous_response_id"] = self._last_response_id

        response = self._client.responses.create(**kwargs)

        self._last_response_id = getattr(response, "id", None)

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        if getattr(response, "output", None):
            first_msg = response.output[0]
            content = getattr(first_msg, "content", None)
            if content and len(content) > 0:
                first_block = content[0]
                text = getattr(first_block, "text", None)
                if text is not None:
                    return text

        return str(response)
