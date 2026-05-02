"""
run_groq.py — Experiment runner for Groq-hosted models.
Runs the full benchmark against Llama 3.3 70B via Groq API.

Usage:
  python scripts/run_groq.py --api-key YOUR_KEY
  python scripts/run_groq.py --api-key YOUR_KEY --model llama-3.3-70b-versatile
  python scripts/run_groq.py --api-key YOUR_KEY --benchmark data/benchmark_complete.json
  python scripts/run_groq.py --api-key YOUR_KEY --dry-run  (test with first 3 samples)

Requires: pip install groq
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

try:
    from groq import Groq
except ImportError:
    print("ERROR: groq package not installed. Run: pip install groq")
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

MODELS = [
    "llama-3.3-70b-versatile",   # Llama 3.3 70B — largest model, run via Groq
]

DEFAULT_BENCHMARK = "data/benchmark_complete.json"
RESULTS_DIR = "results"
RATE_LIMIT_DELAY = 2.5  # seconds between calls (Groq free tier: 30 RPM)


# ============================================================
# Helpers
# ============================================================

def strip_think_tags(text):
    """Remove <think>...</think> reasoning blocks from model outputs."""
    if text is None:
        return ""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def build_messages(sample):
    """
    Build the chat messages list from a benchmark sample.
    Handles all 7 categories + benign samples uniformly.

    For samples WITH external_context (Categories A-D, some benign):
      - System message = sample's system_prompt
      - User message   = "Context: {external_context}\n\nQuestion: {user_query}"

    For samples WITHOUT external_context (Categories E-G from PromptSleuth, some benign):
      - System message = sample's system_prompt
      - User message   = sample's user_query (which contains the injection directly)
    """
    messages = [{"role": "system", "content": sample["system_prompt"]}]

    user_msg = sample["user_query"]
    ext_ctx = sample.get("external_context")
    if ext_ctx:
        user_msg = f"Context: {ext_ctx}\n\nQuestion: {user_msg}"

    messages.append({"role": "user", "content": user_msg})
    return messages


def call_groq(client, sample, model_name):
    """Send a single benchmark sample to the Groq API and return the output."""
    messages = build_messages(sample)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,       # Deterministic for reproducibility
        max_tokens=1024,
    )

    output = response.choices[0].message.content
    output = strip_think_tags(output)
    return output


def make_result_record(sample, model_name, model_output):
    """
    Build a result record that preserves ALL original sample fields
    plus adds model output and metadata. This ensures downstream
    analysis scripts can access injection_position, context_length,
    injection_language, severity, etc.
    """
    record = {**sample}  # Copy all original fields
    record["model"] = model_name
    record["model_output"] = model_output
    record["timestamp"] = datetime.now().isoformat()
    return record


def safe_filename(model_name):
    """Convert model name to safe filename."""
    return model_name.replace(".", "_").replace("/", "_").replace("-", "_")


# ============================================================
# Main runner
# ============================================================

def run_benchmark(client, benchmark, model_name, results_dir, dry_run=False):
    """Run the full benchmark for a single model."""
    samples = benchmark if not dry_run else benchmark[:3]
    total = len(samples)
    model_safe = safe_filename(model_name)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Samples: {total} {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*60}")

    results = []
    errors = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        print(f"  [{i+1:>3d}/{total}] {sample['id']:10s} | {sample['category']:30s}", end="", flush=True)

        try:
            output = call_groq(client, sample, model_name)
            record = make_result_record(sample, model_name, output)
            results.append(record)
            # Truncate output for display
            display = output[:80].replace("\n", " ") + ("..." if len(output) > 80 else "")
            print(f" | OK: {display}")

        except Exception as e:
            error_msg = f"ERROR: {type(e).__name__}: {e}"
            record = make_result_record(sample, model_name, error_msg)
            results.append(record)
            errors += 1
            print(f" | {error_msg[:60]}")

            # If rate limited, wait longer
            if "rate" in str(e).lower() or "429" in str(e):
                print("  >> Rate limited. Waiting 60 seconds...")
                time.sleep(60)

        time.sleep(RATE_LIMIT_DELAY)

    elapsed = time.time() - start_time

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{model_safe}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Done! {total} samples in {elapsed:.0f}s ({errors} errors)")
    print(f"  Saved to: {out_path}")

    return results


# ============================================================
# Entry point
# ============================================================

def main():
    global RATE_LIMIT_DELAY

    parser = argparse.ArgumentParser(description="Run benchmark on Groq-hosted models")
    parser.add_argument("--api-key", required=True, help="Your Groq API key")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Path to benchmark JSON")
    parser.add_argument("--model", default=None, help="Run only this model (default: both)")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true", help="Test with first 3 samples only")
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY, help="Seconds between API calls")
    args = parser.parse_args()

    RATE_LIMIT_DELAY = args.delay

    # Load benchmark
    print(f"Loading benchmark from: {args.benchmark}")
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} samples")

    # Print category distribution
    cat_dist = {}
    for s in benchmark:
        cat_dist[s["category"]] = cat_dist.get(s["category"], 0) + 1
    print("\nCategory distribution:")
    for cat, count in sorted(cat_dist.items()):
        print(f"  {cat:35s}: {count}")

    # Initialize client
    client = Groq(api_key=args.api_key)

    # Select models
    models = [args.model] if args.model else MODELS

    # Run each model
    all_results = {}
    for model_name in models:
        results = run_benchmark(client, benchmark, model_name, args.results_dir, args.dry_run)
        all_results[model_name] = results

    print(f"\n{'='*60}")
    print(f"  ALL DONE! Results saved to: {args.results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()