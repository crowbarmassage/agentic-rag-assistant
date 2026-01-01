"""Google Gemini LLM provider implementation."""

import json
from typing import Optional, Type
import google.generativeai as genai
from pydantic import BaseModel

from .base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation."""

    SUPPORTED_MODELS = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model or self.DEFAULT_MODEL
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model_name} not supported. Choose from: {self.SUPPORTED_MODELS}")

        if api_key:
            genai.configure(api_key=api_key)
        # Otherwise uses GOOGLE_API_KEY env var

        self.model = genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate text completion."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )

        # Handle usage metadata safely
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
            }

        return LLMResponse(
            content=response.text,
            model=self.model_name,
            usage=usage,
            raw_response=response
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseModel:
        """Generate structured output."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = f"""
Respond with valid JSON conforming to this schema:
{schema_json}

Output ONLY the JSON object, no markdown or additional text.
"""
        full_prompt = f"{system_prompt or ''}\n\n{schema_instruction}\n\n{prompt}".strip()

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json"
        )

        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )

        return response_model.model_validate_json(response.text)

    @property
    def provider_name(self) -> str:
        return "gemini"
