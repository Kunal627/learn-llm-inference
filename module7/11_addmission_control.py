
import math

BLOCK_SIZE = 16
TOTAL_BLOCKS = 12
MAX_ACTIVE_SEQUENCES = 3

free_blocks = TOTAL_BLOCKS
active_requests = {}
waiting_requests = []


def required_blocks(prompt_tokens, output_tokens):
    total_tokens = prompt_tokens + output_tokens
    return math.ceil(total_tokens / BLOCK_SIZE)


def admit_request(request):
    global free_blocks

    request_id = request["id"]

    blocks = required_blocks(
        request["prompt"],
        request["output"]
    )

    if len(active_requests) >= MAX_ACTIVE_SEQUENCES:
        print(f"{request_id}: Rejected from admission - sequence limit")
        waiting_requests.append(request)
        return False

    if blocks > free_blocks:
        print(
            f"{request_id}: Waiting - needs {blocks} blocks, "
            f"only {free_blocks} available"
        )
        waiting_requests.append(request)
        return False

    active_requests[request_id] = blocks

    global_free = blocks

    free_blocks -= global_free

    print(
        f"{request_id}: Admitted - allocated {blocks} blocks"
    )

    return True


def release_request(request_id):
    global free_blocks

    blocks = active_requests.pop(request_id)

    free_blocks += blocks

    print(
        f"\n{request_id}: Completed - released {blocks} blocks"
    )


def show_status():
    print("\n" + "=" * 45)
    print("SCHEDULER STATUS")
    print("=" * 45)

    print("Active:", active_requests)
    print("Waiting:", [r["id"] for r in waiting_requests])
    print("Free blocks:", free_blocks)


# Requests
requests = [
    {"id": "A", "prompt": 32, "output": 32},
    {"id": "B", "prompt": 48, "output": 32},
    {"id": "C", "prompt": 64, "output": 32},
]

for request in requests:
    admit_request(request)

show_status()

# A completes
release_request("A")

show_status()
