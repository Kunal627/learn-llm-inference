
from dataclasses import dataclass
from collections import deque
import statistics


# ============================================================
# 1. REQUEST DEFINITION
# ============================================================

@dataclass
class Request:

    request_id: str
    arrival_time: int
    prompt_tokens: int
    output_tokens: int

    remaining_tokens: int = 0
    generated_tokens: int = 0

    first_token_time: int | None = None
    completion_time: int | None = None

    status: str = "waiting"

    def __post_init__(self):
        self.remaining_tokens = self.output_tokens


# ============================================================
# 2. CREATE REQUESTS
# ============================================================

requests = [

    Request("A", 0, 100, 3),
    Request("B", 0, 200, 6),
    Request("C", 2, 150, 4),
    Request("D", 4, 80, 3),
    Request("E", 5, 120, 5),
    Request("F", 7, 90, 2),
    Request("G", 8, 300, 7),
    Request("H", 10, 110, 4),

]


# ============================================================
# 3. SCHEDULER CONFIGURATION
# ============================================================

MAX_ACTIVE_SEQUENCES = 2

SIMULATION_TIME = 0

waiting_queue = deque()

active_requests = []

completed_requests = []

event_log = []

max_active_observed = 0


# ============================================================
# 4. ARRIVAL HANDLING
# ============================================================

def process_arrivals(current_time):

    for request in requests:

        if (
            request.arrival_time == current_time
            and request.status == "waiting"
        ):

            waiting_queue.append(request)

            event_log.append(
                f"Time {current_time}: "
                f"Request {request.request_id} arrived"
            )


# ============================================================
# 5. ADMISSION CONTROL
# ============================================================

def admit_requests(current_time):

    while (
        len(active_requests) < MAX_ACTIVE_SEQUENCES
        and waiting_queue
    ):

        request = waiting_queue.popleft()

        request.status = "active"

        active_requests.append(request)

        event_log.append(
            f"Time {current_time}: "
            f"Request {request.request_id} admitted"
        )


# ============================================================
# 6. DECODE ITERATION
# ============================================================

def execute_decode_iteration(current_time):

    finished = []

    for request in active_requests:

        # Generate one token.
        request.generated_tokens += 1

        request.remaining_tokens -= 1

        # Record time to first token.
        if request.first_token_time is None:

            request.first_token_time = current_time + 1

        # Check whether generation is complete.
        if request.remaining_tokens == 0:

            request.status = "completed"

            request.completion_time = current_time + 1

            completed_requests.append(request)

            finished.append(request)

            event_log.append(
                f"Time {current_time + 1}: "
                f"Request {request.request_id} completed"
            )

    # Remove completed requests.
    for request in finished:

        active_requests.remove(request)


# ============================================================
# 7. MAIN SCHEDULING LOOP
# ============================================================

while len(completed_requests) < len(requests):

    # 1. New arrivals.
    process_arrivals(SIMULATION_TIME)

    # 2. Fill available active slots.
    admit_requests(SIMULATION_TIME)

    # 3. Track maximum concurrency.
    max_active_observed = max(
        max_active_observed,
        len(active_requests)
    )

    # 4. Execute one decode iteration.
    if active_requests:

        active_ids = [
            request.request_id
            for request in active_requests
        ]

        event_log.append(
            f"Time {SIMULATION_TIME}: "
            f"Decode active requests {active_ids}"
        )

        execute_decode_iteration(SIMULATION_TIME)

    # 5. Advance simulation.
    SIMULATION_TIME += 1

    # Safety check.
    if SIMULATION_TIME > 10000:
        raise RuntimeError("Simulation did not terminate")


# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

total_output_tokens = sum(
    r.output_tokens for r in requests
)

total_prompt_tokens = sum(
    r.prompt_tokens for r in requests
)

simulation_duration = SIMULATION_TIME

total_requests = len(requests)

requests_per_unit = (
    total_requests / simulation_duration
)

tokens_per_unit = (
    total_output_tokens / simulation_duration
)

latencies = [
    r.completion_time - r.arrival_time
    for r in requests
]

ttfts = [
    r.first_token_time - r.arrival_time
    for r in requests
]

average_latency = statistics.mean(latencies)

average_ttft = statistics.mean(ttfts)


# ============================================================
# 9. PRINT SCHEDULING TRACE
# ============================================================

print("=" * 75)
print("CONTINUOUS BATCHING SCHEDULER")
print("=" * 75)

for event in event_log:
    print(event)


# ============================================================
# 10. REQUEST SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("REQUEST SUMMARY")
print("=" * 75)

print(
    f"{'ID':<8}"
    f"{'Arrival':>10}"
    f"{'Prompt':>10}"
    f"{'Output':>10}"
    f"{'First token':>14}"
    f"{'Completion':>14}"
    f"{'Latency':>10}"
)

print("-" * 75)

for r in requests:

    print(
        f"{r.request_id:<8}"
        f"{r.arrival_time:>10}"
        f"{r.prompt_tokens:>10}"
        f"{r.output_tokens:>10}"
        f"{r.first_token_time:>14}"
        f"{r.completion_time:>14}"
        f"{r.completion_time-r.arrival_time:>10}"
    )


# ============================================================
# 11. FINAL METRICS
# ============================================================

print("\n" + "=" * 75)
print("SCHEDULER METRICS")
print("=" * 75)

print("Total requests:", total_requests)

print("Total prompt tokens:", total_prompt_tokens)

print("Total generated tokens:", total_output_tokens)

print("Simulation duration:", simulation_duration)

print("Maximum active sequences:", max_active_observed)

print(
    f"Requests per time unit: "
    f"{requests_per_unit:.3f}"
)

print(
    f"Generated tokens per time unit: "
    f"{tokens_per_unit:.3f}"
)

print(
    f"Average completion latency: "
    f"{average_latency:.3f}"
)

print(
    f"Average time to first token: "
    f"{average_ttft:.3f}"
)

print("\nSimulation completed.")
