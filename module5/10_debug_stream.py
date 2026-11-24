
import requests
import json

url = "http://localhost:8080/v1/chat/completions"

payload = {
    "model": "/models/Qwen3-8B-Q4_K_M.gguf",
    "messages": [
        {
            "role": "user",
            "content": "Explain FP16 in two sentences."
        }
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": True
}

response = requests.post(
    url,
    json=payload,
    stream=True,
    timeout=120
)

print("HTTP STATUS:", response.status_code)
print("CONTENT TYPE:", response.headers.get("Content-Type"))

response.raise_for_status()

for i, line in enumerate(response.iter_lines(decode_unicode=True)):

    if not line:
        continue

    print(f"\nRAW LINE {i}: {repr(line)}")

    if isinstance(line, bytes):
        line = line.decode("utf-8")

    if line.startswith("data:"):
        data = line[len("data:"):].strip()

        if data == "[DONE]":
            print("STREAM FINISHED")
            break

        try:
            chunk = json.loads(data)

            print("JSON:", json.dumps(chunk, indent=2))

        except json.JSONDecodeError:
            print("Could not parse JSON")

    if i >= 30:
        break

response.close()
