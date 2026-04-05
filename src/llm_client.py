"""
LLM Client for interacting with Ollama, Anthropic, and optionally OpenAI.
"""

import json
import os
import time
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    success: bool
    error: Optional[str] = None


class OllamaClient:
    """Client for Ollama local LLM server."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except requests.exceptions.RequestException:
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library."""
        print(f"Pulling model {model_name}... (this may take a while)")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minutes for large models
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        print(f"  {data['status']}", end="\r")
            print(f"\nModel {model_name} pulled successfully!")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        format: str = "json"  # Request JSON output
    ) -> LLMResponse:
        """Generate a response from the model."""
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    "format": format,
                },
                timeout=120  # 2 minutes timeout
            )

            latency = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                return LLMResponse(
                    text=data.get("response", ""),
                    model=model,
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                    latency_seconds=latency,
                    success=True,
                )
            else:
                return LLMResponse(
                    text="",
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_seconds=latency,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

        except requests.exceptions.Timeout:
            return LLMResponse(
                text="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                success=False,
                error="Request timed out",
            )
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                text="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                success=False,
                error=str(e),
            )


class OpenAIClient:
    """Client for OpenAI API (optional)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.api_key is not None and len(self.api_key) > 0

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Generate a response from OpenAI."""
        if not self.is_available():
            return LLMResponse(
                text="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=0,
                success=False,
                error="OpenAI API key not configured",
            )

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )

            latency = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    latency_seconds=latency,
                    success=True,
                )
            else:
                return LLMResponse(
                    text="",
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_seconds=latency,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

        except requests.exceptions.RequestException as e:
            return LLMResponse(
                text="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                success=False,
                error=str(e),
            )


class AnthropicClient:
    """Client for Anthropic Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1"

    def is_available(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.api_key is not None and len(self.api_key) > 0

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Generate a response from Anthropic Claude."""
        if not self.is_available():
            return LLMResponse(
                text="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=0,
                success=False,
                error="Anthropic API key not configured. Set ANTHROPIC_API_KEY env var.",
            )

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )

            latency = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                text = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text += block.get("text", "")
                usage = data.get("usage", {})
                return LLMResponse(
                    text=text,
                    model=model,
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    latency_seconds=latency,
                    success=True,
                )
            else:
                return LLMResponse(
                    text="",
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_seconds=latency,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

        except requests.exceptions.RequestException as e:
            return LLMResponse(
                text="",
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_seconds=time.time() - start_time,
                success=False,
                error=str(e),
            )


def create_client(provider: str, **kwargs):
    """Factory function to create the appropriate client."""
    if provider == "ollama":
        return OllamaClient(base_url=kwargs.get("base_url", "http://localhost:11434"))
    elif provider == "openai":
        return OpenAIClient(api_key=kwargs.get("api_key"))
    elif provider == "anthropic":
        return AnthropicClient(api_key=kwargs.get("api_key"))
    else:
        raise ValueError(f"Unknown provider: {provider}. Available: ollama, openai, anthropic")


# Quick test function
def test_ollama_connection():
    """Test if Ollama is running and list available models."""
    client = OllamaClient()

    if not client.is_available():
        print("Ollama is not running. Start it with: ollama serve")
        return False

    print("Ollama is running!")
    models = client.list_models()

    if models:
        print(f"Available models: {models}")
    else:
        print("No models installed. Pull one with: ollama pull llama3.2:1b")

    return True


if __name__ == "__main__":
    test_ollama_connection()
