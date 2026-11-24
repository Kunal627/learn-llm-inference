
import requests
import time
import json
import pandas as pd
import statistics

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_URL = "http://localhost:8080"

MODEL = "/models/Qwen3-8B-Q4_K_M.gguf"

TEMPERATURE = 0.0
MAX_TOKENS = 256

WARMUP_RUNS = 1
BENCHMARK_RUNS = 5

REQUEST_TIMEOUT = 300

RESULTS_FILE = "qwen3_baseline_results.csv"
SUMMARY_FILE = "qwen3_baseline_summary.csv"


# ============================================================
# 2. TEST PROMPTS
# ============================================================

PROMPTS = [

    """
    Explain the difference between FP16 and BF16
    in LLM inference. Keep your answer concise.
    """,

    """
    What is KV cache quantization?
    Explain how Q8_0 reduces memory usage compared
    with FP16. Keep your answer concise.
    """,

    """
    Extract the chemical information from this report entry:

    Extractable name: Bis(2-ethylhexyl) phthalate
    Molecular formula: C24H38O4
    CAS number: 117-81-7
    Maximum total water: 12.5 ug/g

    Return JSON with:
    chemical_name,
    molecular_formula,
    cas_number,
    max_total_water_ug_per_g.
    """,

    """
    Two reports contain the following chemical names:

    Report A: 2-Ethylhexyl phthalate
    Report B: Bis(2-ethylhexyl) benzene-1,2-dicarboxylate

    Explain whether these names could refer to the same chemical.

    Return a JSON object with:
    possible_match,
    reasoning.
    """
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_float(value):

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value, unit=""):

    if value is None:
        return "N/A"

    return f"{value:.3f}{unit}"


def check_server():

    url = f"{BASE_URL}/health"

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    print("LLM server health: OK")


# ============================================================
# 4. SINGLE STREAMING BENCHMARK
# ============================================================

def benchmark_once(prompt):

    url = f"{BASE_URL}/v1/chat/completions"

    payload = {

        "model": MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": TEMPERATURE,

        "max_tokens": MAX_TOKENS,

        "stream": True,

        # Ask llama.cpp to return token usage
        "stream_options": {
            "include_usage": True
        },

        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    start_time = time.perf_counter()

    first_generated_time = None
    first_answer_time = None

    reasoning_chunks = []
    answer_chunks = []

    usage = None
    finish_reason = None

    response = requests.post(
        url,
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    try:

        for line in response.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue

            if isinstance(line, bytes):
                line = line.decode("utf-8")

            if not line.startswith("data:"):
                continue

            data = line[len("data:"):].strip()

            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)

            except json.JSONDecodeError:
                continue

            # ------------------------------------------------
            # Token usage may arrive in a separate final chunk
            # ------------------------------------------------

            if chunk.get("usage") is not None:
                usage = chunk["usage"]

            choices = chunk.get("choices", [])

            if not choices:
                continue

            choice = choices[0]

            delta = choice.get("delta", {})

            reasoning = delta.get("reasoning_content") or ""

            content = delta.get("content") or ""

            current_time = time.perf_counter()

            # ------------------------------------------------
            # First generated token: reasoning OR answer
            # ------------------------------------------------

            if reasoning or content:

                if first_generated_time is None:

                    first_generated_time = current_time

            # ------------------------------------------------
            # Final answer content
            # ------------------------------------------------

            if content:

                if first_answer_time is None:

                    first_answer_time = current_time

                answer_chunks.append(content)

            # ------------------------------------------------
            # Reasoning content
            # ------------------------------------------------

            if reasoning:

                reasoning_chunks.append(reasoning)

            # ------------------------------------------------
            # Finish reason
            # ------------------------------------------------

            if choice.get("finish_reason"):

                finish_reason = choice["finish_reason"]

    finally:

        response.close()

    end_time = time.perf_counter()

    # ========================================================
    # 5. CALCULATE METRICS
    # ========================================================

    total_time = end_time - start_time

    reasoning_text = "".join(reasoning_chunks)

    answer_text = "".join(answer_chunks)

    ttft = None

    if first_generated_time is not None:

        ttft = first_generated_time - start_time

    answer_ttft = None

    if first_answer_time is not None:

        answer_ttft = first_answer_time - start_time

    # Exact completion token count, if supplied
    completion_tokens = None
    prompt_tokens = None
    total_tokens = None

    if usage:

        completion_tokens = usage.get("completion_tokens")

        prompt_tokens = usage.get("prompt_tokens")

        total_tokens = usage.get("total_tokens")

    # --------------------------------------------------------
    # TPOT: average time per subsequent generated token
    # --------------------------------------------------------

    tpot = None

    if (
        completion_tokens is not None
        and completion_tokens > 1
        and ttft is not None
    ):

        tpot = (
            total_time - ttft
        ) / (completion_tokens - 1)

    # --------------------------------------------------------
    # Overall generation throughput
    # --------------------------------------------------------

    tokens_per_second = None

    if (
        completion_tokens is not None
        and total_time > 0
    ):

        tokens_per_second = (
            completion_tokens / total_time
        )

    # ========================================================
    # 6. RETURN RESULTS
    # ========================================================

    return {

        "ttft_seconds": ttft,

        "answer_ttft_seconds": answer_ttft,

        "total_seconds": total_time,

        "prompt_tokens": prompt_tokens,

        "completion_tokens": completion_tokens,

        "total_tokens": total_tokens,

        "tpot_seconds": tpot,

        "tokens_per_second": tokens_per_second,

        "reasoning": reasoning_text,

        "output": answer_text,

        "completion_chars": len(answer_text),

        "finish_reason": finish_reason
    }


# ============================================================
# 7. RUN BENCHMARK
# ============================================================

def run_benchmark():

    check_server()

    results = []

    total_executions = (
        len(PROMPTS) *
        (WARMUP_RUNS + BENCHMARK_RUNS)
    )

    execution_count = 0

    for prompt_index, prompt in enumerate(PROMPTS):

        print("\n" + "=" * 65)

        print(
            f"PROMPT {prompt_index + 1}/{len(PROMPTS)}"
        )

        print("=" * 65)

        for run in range(
            WARMUP_RUNS + BENCHMARK_RUNS
        ):

            execution_count += 1

            print(
                f"\nExecution {execution_count}/{total_executions}"
            )

            result = benchmark_once(prompt)

            # ------------------------------------------------
            # Warmup
            # ------------------------------------------------

            if run < WARMUP_RUNS:

                print("Warmup completed")

                continue

            # ------------------------------------------------
            # Store measurements
            # ------------------------------------------------

            row = {

                "prompt": prompt_index + 1,

                "run": run,

                "ttft_seconds": result["ttft_seconds"],

                "answer_ttft_seconds": result[
                    "answer_ttft_seconds"
                ],

                "total_seconds": result["total_seconds"],

                "prompt_tokens": result["prompt_tokens"],

                "completion_tokens": result[
                    "completion_tokens"
                ],

                "total_tokens": result["total_tokens"],

                "tpot_seconds": result["tpot_seconds"],

                "tokens_per_second": result[
                    "tokens_per_second"
                ],

                "completion_chars": result[
                    "completion_chars"
                ],

                "finish_reason": result["finish_reason"],

                "output": result["output"],

                "reasoning": result["reasoning"]
            }

            results.append(row)

            # ------------------------------------------------
            # Print measurements
            # ------------------------------------------------

            print(
                "TTFT:",
                format_metric(
                    result["ttft_seconds"], "s"
                )
            )

            print(
                "Answer TTFT:",
                format_metric(
                    result["answer_ttft_seconds"], "s"
                )
            )

            print(
                "Total:",
                format_metric(
                    result["total_seconds"], "s"
                )
            )

            print(
                "Completion tokens:",
                result["completion_tokens"]
            )

            print(
                "TPOT:",
                format_metric(
                    result["tpot_seconds"], "s"
                )
            )

            print(
                "Tokens/sec:",
                format_metric(
                    result["tokens_per_second"]
                )
            )

            print(
                "Answer characters:",
                result["completion_chars"]
            )

            print(
                "Finish reason:",
                result["finish_reason"]
            )

            print("\nGenerated answer:")

            print(result["output"][:500])

    return results


# ============================================================
# 8. SAVE RAW RESULTS
# ============================================================

def save_results(results):

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print("\nRaw results saved:", RESULTS_FILE)

    return df


# ============================================================
# 9. GENERATE SUMMARY
# ============================================================

def generate_summary(df):

    metric_columns = [

        "ttft_seconds",

        "answer_ttft_seconds",

        "total_seconds",

        "tpot_seconds",

        "tokens_per_second"
    ]

    summary = (
        df.groupby("prompt")[metric_columns]
        .agg(["mean", "median", "min", "max"])
    )

    summary.to_csv(SUMMARY_FILE)

    print("\n" + "=" * 65)

    print("BENCHMARK SUMMARY")

    print("=" * 65)

    print(summary.to_string())

    print("\nSummary saved:", SUMMARY_FILE)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    print("\nOVERALL MEDIAN METRICS")

    for metric in metric_columns:

        values = df[metric].dropna()

        if len(values) == 0:

            print(f"{metric}: N/A")

        else:

            print(
                f"{metric}: "
                f"{statistics.median(values):.4f}"
            )


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print("QWEN3 QUANTIZATION BASELINE BENCHMARK")

    print("=" * 65)

    print("Model:", MODEL)

    print("Temperature:", TEMPERATURE)

    print("Maximum output tokens:", MAX_TOKENS)

    print("Warmup runs:", WARMUP_RUNS)

    print("Measured runs:", BENCHMARK_RUNS)

    print("Number of prompts:", len(PROMPTS))

    print("=" * 65)

    results = run_benchmark()

    df = save_results(results)

    generate_summary(df)

    print("\nBenchmark completed successfully.")
