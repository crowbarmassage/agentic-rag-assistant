"""Groq LLM provider implementation."""

import json
from typing import Optional, Type
from groq import Groq
from pydantic import BaseModel

from .base import BaseLLMProvider, LLMResponse


class GroqProvider(BaseLLMProvider):
    """Groq API provider implementation."""

    SUPPORTED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model} not supported. Choose from: {self.SUPPORTED_MODELS}")
        self.client = Groq(api_key=api_key)  # Uses GROQ_API_KEY env var if None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            raw_response=response
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> BaseModel:
        """Generate structured output using JSON mode."""
        messages = []

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = f"""
Respond with valid JSON conforming to this schema:
{schema_json}

Output ONLY the JSON object, no markdown formatting.
"""

        effective_system = ((system_prompt or "") + "\n\n" + schema_instruction).strip()
        messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        return response_model.model_validate_json(raw_content)

    @property
    def provider_name(self) -> str:
        return "groq"
