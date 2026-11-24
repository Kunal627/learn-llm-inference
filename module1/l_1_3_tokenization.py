
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"

# Load tokenizer only
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully")
print("Vocabulary size:", tokenizer.vocab_size)

# Input text
text = "What is artificial intelligence?"

# Convert text to token IDs
token_ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

print("\nOriginal text:")
print(text)

print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))

# Convert IDs back to text
decoded_text = tokenizer.decode(token_ids)

print("\nDecoded text:")
print(decoded_text)

# Inspect individual tokens
print("\nIndividual tokens:")
print(tokenizer.convert_ids_to_tokens(token_ids))


sentences = [
    "Hello",
    "Hello world",
    "Artificial intelligence is fascinating.",
    "The quick brown fox jumps over the lazy dog.",
    "Pneumonoultramicroscopicsilicovolcanoconiosis",
    "AI inference engineering using PyTorch and CUDA"
]

print("\nTOKENIZATION COMPARISON")
print("-" * 75)

for sentence in sentences:

    ids = tokenizer.encode(
        sentence,
        add_special_tokens=False
    )

    tokens = tokenizer.convert_ids_to_tokens(ids)

    print("\nText:", sentence)
    print("Token count:", len(ids))
    print("Tokens:", tokens)



messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What is artificial intelligence?"
    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(formatted_prompt)



# Convert the chat template directly into token IDs

chat_token_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True
)

print("Chat token IDs:", chat_token_ids)
print("Total tokens:", len(chat_token_ids))
print("chat token size:", len(chat_token_ids))


sentences = [
    "AI is useful.",
    "Artificial intelligence is useful.",
    "Artificial intelligence is useful for healthcare.",
    "Artificial intelligence is useful for medical image analysis.",
    "Artificial intelligence is useful for medical image analysis and clinical research."
]

print(f"{'Words':<10}{'Tokens':<10}{'Characters':<12}")
print("-" * 32)

for text in sentences:
    ids = tokenizer.encode(text, add_special_tokens=False)

    words = len(text.split())
    tokens = len(ids)
    chars = len(text)

    print(f"{words:<10}{tokens:<10}{chars:<12}")
