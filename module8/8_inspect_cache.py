import asyncio
import httpx

URL = "http://localhost:8000/v1/chat/completions"

async def send_request(request_id):
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Request {request_id}: Explain how "
                    "PagedAttention works in detail. "
                    "Include examples and technical explanations."
                )
            }
        ],
        "temperature": 0,
        "max_tokens": 1000,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()

        print(
            f"Request {request_id}: "
            f"prompt={result['usage']['prompt_tokens']}, "
            f"completion={result['usage']['completion_tokens']}"
        )

async def main():
    await asyncio.gather(
        *(send_request(i) for i in range(1, 5))
    )

asyncio.run(main())