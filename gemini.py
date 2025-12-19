# gemini.py
from typing import List, Any
from google import genai
from google.genai import types
from PIL import Image
import io


class GeminiInterface:
    """
    Interface for interacting with Google's Gemini API.
    
    Supports text and image inputs, conversation history, and optional web search.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",   # Change this to use a different model
        temperature: float = 0.0,
        max_tokens: int = 2048
    ):
        """
        Initialize the Gemini interface.
        
        Args:
            api_key: Google Gemini API key
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature (0.0 for deterministic output)
            max_tokens: Maximum number of tokens in the response
        """
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

        # Store basic config scalars separately to avoid strange fields from model_dump
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Web Search tool
        # https://ai.google.dev/gemini-api/docs/grounding?hl=zh-cn 
        self._search_tool = types.Tool(google_search=types.GoogleSearch())

        # Simple conversation history: maintain Content list ourselves
        self._history: list[types.Content] = []

    def reset(self):
        """
        Manually reset conversation history, equivalent to starting a new conversation.
        """
        self._history.clear()

    def _build_config(self, use_web_search: bool) -> types.GenerateContentConfig:
        """
        Build GenerateContentConfig uniformly to avoid repeatedly passing tools.
        
        Args:
            use_web_search: Whether to enable web search functionality
            
        Returns:
            Configured GenerateContentConfig instance
        """
        if use_web_search:
            return types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                tools=[self._search_tool],
            )
        else:
            return types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )

    def ask(self,
            prompt_elements: List[Any],
            use_web_search: bool = True) -> str:
        """
        Send a user message and return Gemini's text response.
        
        Args:
            prompt_elements: List of strings and/or PIL.Image objects. Supports mixed text and images.
            use_web_search: Whether to enable web search for this request
            
        Returns:
            Text response from Gemini
            
        Example:
            >>> interface = GeminiInterface(api_key="...")
            >>> response = interface.ask(["What is this?", Image.open("image.png")])
        """
        # Process prompt_elements, distinguishing between text and images
        parts = []
        text_parts = []
        
        for element in prompt_elements:
            if isinstance(element, Image.Image):
                # If it's a PIL Image, convert to bytes and create image part
                img_bytes = io.BytesIO()
                element.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                parts.append(types.Part.from_bytes(
                    data=img_bytes.read(),
                    mime_type='image/png'
                ))
            else:
                # Collect text parts first
                text_parts.append(str(element))
        
        # If there's text, add text part
        if text_parts:
            user_text = " ".join(text_parts)
            parts.insert(0, types.Part.from_text(text=user_text))
        
        # If no parts, create an empty text part
        if not parts:
            parts.append(types.Part.from_text(text=""))

        # Construct user message for this round
        user_msg = types.UserContent(parts=parts)

        # Send history + current message together to the model
        contents: list[types.Content] = [*self._history, user_msg]

        # Decide whether to use search for this round
        config = self._build_config(use_web_search=use_web_search)

        # Call Gemini
        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )

        # Add this round's user & model response to history
        self._history.append(user_msg)
        if resp.candidates and resp.candidates[0].content:
            self._history.append(resp.candidates[0].content)

        # Prefer resp.text
        if getattr(resp, "text", None):
            return resp.text

        # Fallback: concatenate all text from parts
        if resp.candidates and resp.candidates[0].content:
            parts = resp.candidates[0].content.parts or []
            chunks = []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    chunks.append(t)
            if chunks:
                return "".join(chunks)

        return "[Gemini error: no text in response]"
