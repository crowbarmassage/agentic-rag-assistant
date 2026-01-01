"""OpenAI LLM provider implementation."""

import json
from typing import Optional, Type
from openai import OpenAI
from pydantic import BaseModel

from .base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation."""

    SUPPORTED_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {self.model} not supported. Choose from: {self.SUPPORTED_MODELS}")
        self.client = OpenAI(api_key=api_key)  # Uses OPENAI_API_KEY env var if None

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

        # Build schema instruction
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = f"""
You must respond with valid JSON that conforms to this schema:
{schema_json}

Respond ONLY with the JSON object, no additional text or markdown formatting.
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
        return "openai"
