from __future__ import annotations

import io
from typing import Any, List, Optional

from google import genai
from google.genai import types
from PIL import Image


class GeminiInterface:
    """
    Wrapper for Google's Gemini GenerateContent API.

    Features:
      - Accepts mixed inputs: text and PIL images
      - Maintains a simple in-memory conversation history
      - Optionally enables Google's managed web search grounding tool
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        enable_search_tool: bool = True,
    ) -> None:
        """
        Args:
            api_key: Google Gemini API key.
            model_name: Gemini model identifier.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum number of tokens to generate for a single response.
            enable_search_tool: If True, configures the Google Search grounding tool.
        """
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self._search_tool: Optional[types.Tool] = None
        if enable_search_tool:
            self._search_tool = types.Tool(google_search=types.GoogleSearch())

        self._history: List[types.Content] = []

    def reset(self) -> None:
        """Clear conversation history (equivalent to starting a new session)."""
        self._history.clear()

    def _build_config(self, use_web_search: bool) -> types.GenerateContentConfig:
        """
        Create a per-request GenerateContentConfig.

        Args:
            use_web_search: If True, attach the web search tool (when configured).

        Returns:
            A configured GenerateContentConfig instance.
        """
        cfg_kwargs = {
            "temperature": self._temperature,
            "max_output_tokens": self.max_tokens,
        }
        if use_web_search and self._search_tool is not None:
            cfg_kwargs["tools"] = [self._search_tool]
        return types.GenerateContentConfig(**cfg_kwargs)

    @staticmethod
    def _image_to_part(img: Image.Image, mime_type: str = "image/png") -> types.Part:
        """
        Convert a PIL image into a Gemini Part.

        Args:
            img: PIL Image.
            mime_type: MIME type for encoding; defaults to PNG.

        Returns:
            types.Part containing the encoded image bytes.
        """
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return types.Part.from_bytes(data=buf.read(), mime_type=mime_type)

    @staticmethod
    def _coerce_text_parts(prompt_elements: List[Any]) -> List[str]:
        """Stringify all non-image elements."""
        return [str(x) for x in prompt_elements if not isinstance(x, Image.Image)]

    def ask(self, prompt_elements: List[Any], use_web_search: bool = True) -> str:
        """
        Send a user message and return Gemini's text response.

        Args:
            prompt_elements: A list of elements. Each element may be:
              - str (or any object convertible to str), or
              - PIL.Image.Image
              Mixed text+image inputs are supported.
            use_web_search: If True, enable web search grounding (when configured).

        Returns:
            The model's response text, or a short error string if no text is present.
        """
        parts: List[types.Part] = []

        # Gather images first, then insert a single text part (if any) at the front.
        for element in prompt_elements:
            if isinstance(element, Image.Image):
                parts.append(self._image_to_part(element))

        text_parts = self._coerce_text_parts(prompt_elements)
        if text_parts:
            user_text = " ".join(text_parts)
            parts.insert(0, types.Part.from_text(text=user_text))

        if not parts:
            parts.append(types.Part.from_text(text=""))

        user_msg = types.UserContent(parts=parts)
        contents: List[types.Content] = [*self._history, user_msg]

        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=self._build_config(use_web_search=use_web_search),
        )

        # Update history (best effort).
        self._history.append(user_msg)
        if getattr(resp, "candidates", None) and resp.candidates and resp.candidates[0].content:
            self._history.append(resp.candidates[0].content)

        # Prefer resp.text if present.
        text = getattr(resp, "text", None)
        if text:
            return text

        # Fallback: concatenate text fields from returned parts.
        if getattr(resp, "candidates", None) and resp.candidates and resp.candidates[0].content:
            cand_parts = resp.candidates[0].content.parts or []
            chunks: List[str] = []
            for p in cand_parts:
                t = getattr(p, "text", None)
                if t:
                    chunks.append(t)
            if chunks:
                return "".join(chunks)

        return "[Gemini error: empty response]"
