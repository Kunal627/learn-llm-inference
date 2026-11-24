import requests

url = "http://localhost:8080/v1/chat/completions"

payload = {
    "model": "/models/Qwen3-8B-Q4_K_M.gguf",
    "messages": [
        {
            "role": "user",
            "content": "Explain FP16 and BF16 in two sentences."
        }
    ],
    "temperature": 0,
    "max_tokens": 256,
    "stream": False,
    "chat_template_kwargs": {
        "enable_thinking": False
    }
}

response = requests.post(
    url,
    json=payload,
    timeout=300
)

print("Status:", response.status_code)
print(response.text)