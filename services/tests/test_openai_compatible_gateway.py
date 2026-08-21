import json

import httpx

from app.features.sales_knowledge_identification.adapters.openai_compatible import (
    OpenAICompatibleGateway,
)
from app.features.sales_knowledge_identification.models import ModelRequest


def test_gateway_calls_chat_completions_with_json_output_contract() -> None:
    captured_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(
            200,
            json={
                "model": "qwen-plus",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"candidates":[],"weakSignals":[],"unresolvedItems":[]}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 101,
                    "completion_tokens": 22,
                    "total_tokens": 123,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gateway = OpenAICompatibleGateway(
        provider="dashscope",
        base_url="https://example.test/compatible-mode/v1",
        api_key="secret-key",
        model="qwen-plus",
        temperature=0.1,
        max_output_tokens=8000,
        timeout_seconds=30,
        client=client,
    )

    completion = gateway.complete(
        ModelRequest(
            document_package_id="DP-TEST",
            system_prompt="只输出 JSON",
            user_prompt="识别这份资料",
        )
    )

    assert captured_request is not None
    assert str(captured_request.url) == ("https://example.test/compatible-mode/v1/chat/completions")
    assert captured_request.headers["authorization"] == "Bearer secret-key"
    body = json.loads(captured_request.content)
    assert body["model"] == "qwen-plus"
    assert body["response_format"] == {"type": "json_object"}
    assert body["enable_thinking"] is False
    assert body["messages"][0] == {"role": "system", "content": "只输出 JSON"}
    assert completion.content.startswith('{"candidates"')
    assert completion.prompt_tokens == 101
    assert completion.completion_tokens == 22
    assert completion.finish_reason == "stop"
