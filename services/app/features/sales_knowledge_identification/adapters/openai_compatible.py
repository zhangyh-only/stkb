from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from app.core.logging import get_logger

from ..models import ModelCompletion, ModelRequest

logger = get_logger("model")


class ModelGatewayError(RuntimeError):
    """模型服务调用失败，错误信息不包含鉴权凭据。"""


class OpenAICompatibleGateway:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: int,
        enable_thinking: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def complete(self, request: ModelRequest) -> ModelCompletion:
        started = perf_counter()
        logger.info(
            "model_call.started provider=%s model=%s document_package_id=%s",
            self.provider,
            self.model,
            request.document_package_id,
        )
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_output_tokens,
                    "response_format": {"type": "json_object"},
                    "enable_thinking": self.enable_thinking,
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            logger.exception(
                "model_call.failed provider=%s model=%s document_package_id=%s duration_ms=%d",
                self.provider,
                self.model,
                request.document_package_id,
                round((perf_counter() - started) * 1000),
            )
            raise ModelGatewayError(f"model request failed: {error}") from error

        try:
            choice = payload["choices"][0]
            message_content = choice["message"]["content"]
            content = _normalize_message_content(message_content)
            usage = payload.get("usage") or {}
        except (KeyError, IndexError, TypeError) as error:
            logger.exception(
                "model_call.invalid_response provider=%s model=%s document_package_id=%s "
                "duration_ms=%d",
                self.provider,
                self.model,
                request.document_package_id,
                round((perf_counter() - started) * 1000),
            )
            raise ModelGatewayError("model response does not match Chat Completions") from error

        completion = ModelCompletion(
            provider=self.provider,
            model=payload.get("model") or self.model,
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason"),
        )
        logger.info(
            "model_call.completed provider=%s model=%s document_package_id=%s "
            "duration_ms=%d prompt_tokens=%d completion_tokens=%d finish_reason=%s",
            completion.provider,
            completion.model,
            request.document_package_id,
            round((perf_counter() - started) * 1000),
            completion.prompt_tokens,
            completion.completion_tokens,
            completion.finish_reason,
        )
        return completion


def _normalize_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_parts = [
            item.get("text", "")
            for item in value
            if isinstance(item, dict) and item.get("type") in {None, "text"}
        ]
        content = "".join(text_parts)
        if content:
            return content
    raise ModelGatewayError("model response content is not text")
