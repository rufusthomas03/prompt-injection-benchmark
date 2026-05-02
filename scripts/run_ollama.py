"""
run_ollama.py — Experiment runner for local Ollama models.
Runs the full benchmark against a locally-hosted model via Ollama.

Models used in this project (run each separately with --model):
  llama3.1:8b   — Llama 3.1 8B
  llama3.2:3b   — Llama 3.2 3B
  llama3.2:1b   — Llama 3.2 1B

Usage:
  python scripts/run_ollama.py --model llama3.1:8b
  python scripts/run_ollama.py --model llama3.2:3b
  python scripts/run_ollama.py --model llama3.2:1b
  python scripts/run_ollama.py --model llama3.1:8b --dry-run

Prerequisites:
  1. Install Ollama: https://ollama.com/download
  2. Pull the models:
       ollama pull llama3.1:8b
       ollama pull llama3.2:3b
       ollama pull llama3.2:1b
  3. Install Python library: pip install ollama

Note: With 16 GB RAM and no GPU, expect ~10-30 seconds per response.
      For 172 samples, total runtime is ~30-85 minutes per model.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

try:
    import ollama
except ImportError:
    print("ERROR: ollama package not installed. Run: pip install ollama")
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_BENCHMARK = "data/benchmark_complete.json"
RESULTS_DIR = "results"


# ============================================================
# Helpers
# ============================================================

def build_messages(sample):
    """
    Build the chat messages list from a benchmark sample.
    Same logic as run_groq.py for consistency.
    """
    messages = [{"role": "system", "content": sample["system_prompt"]}]

    user_msg = sample["user_query"]
    ext_ctx = sample.get("external_context")
    if ext_ctx:
        user_msg = f"Context: {ext_ctx}\n\nQuestion: {user_msg}"

    messages.append({"role": "user", "content": user_msg})
    return messages


def call_ollama(sample, model_name):
    """Send a single benchmark sample to Ollama and return the output."""
    messages = build_messages(sample)

    response = ollama.chat(
        model=model_name,
        messages=messages,
    )

    return response["message"]["content"]


def make_result_record(sample, model_name, model_output):
    """Build a result record preserving all original fields."""
    record = {**sample}
    record["model"] = f"{model_name}-local"
    record["model_output"] = model_output
    record["timestamp"] = datetime.now().isoformat()
    return record


def safe_filename(model_name):
    return model_name.replace(".", "_").replace("/", "_").replace("-", "_").replace(":", "_")


# ============================================================
# Main runner
# ============================================================

def run_benchmark(benchmark, model_name, results_dir, dry_run=False):
    samples = benchmark if not dry_run else benchmark[:3]
    total = len(samples)
    model_safe = safe_filename(model_name)

    print(f"\n{'='*60}")
    print(f"  Model: {model_name} (local via Ollama)")
    print(f"  Samples: {total} {'(DRY RUN)' if dry_run else ''}")
    print(f"  Expected time: ~{total * 20 // 60}-{total * 30 // 60} minutes")
    print(f"{'='*60}")

    results = []
    errors = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        sample_start = time.time()
        print(f"  [{i+1:>3d}/{total}] {sample['id']:10s} | {sample['category']:30s}", end="", flush=True)

        try:
            output = call_ollama(sample, model_name)
            record = make_result_record(sample, model_name, output)
            results.append(record)
            elapsed_sample = time.time() - sample_start
            display = output[:60].replace("\n", " ") + ("..." if len(output) > 60 else "")
            print(f" | {elapsed_sample:.1f}s | {display}")

        except Exception as e:
            error_msg = f"ERROR: {type(e).__name__}: {e}"
            record = make_result_record(sample, model_name, error_msg)
            results.append(record)
            errors += 1
            print(f" | {error_msg[:60]}")

        # Save checkpoint every 25 samples
        if (i + 1) % 25 == 0:
            checkpoint_path = os.path.join(results_dir, f"{model_safe}_checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            elapsed_total = time.time() - start_time
            remaining = (total - i - 1) * (elapsed_total / (i + 1))
            print(f"\n  >> Checkpoint saved ({i+1}/{total}). "
                  f"Elapsed: {elapsed_total/60:.1f}min. "
                  f"Est. remaining: {remaining/60:.1f}min.\n")

    elapsed = time.time() - start_time

    # Save final results
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{model_safe}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Done! {total} samples in {elapsed/60:.1f} minutes ({errors} errors)")
    print(f"  Saved to: {out_path}")

    # Clean up checkpoint
    checkpoint_path = os.path.join(results_dir, f"{model_safe}_checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return results


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run benchmark on local Ollama model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Path to benchmark JSON")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Test with first 3 samples")
    args = parser.parse_args()

    # Verify Ollama is running
    try:
        models = ollama.list()
        available = [m.model for m in models.models] if hasattr(models, 'models') else []
        print(f"Ollama is running. Available models: {available}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama. Is it running?\n  {e}")
        print("Start it with: ollama serve")
        sys.exit(1)

    # Load benchmark
    print(f"\nLoading benchmark from: {args.benchmark}")
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} samples")

    # Run
    run_benchmark(benchmark, args.model, args.results_dir, args.dry_run)

    print(f"\n{'='*60}")
    print(f"  DONE! Results saved to: {args.results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()