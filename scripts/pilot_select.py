"""
pilot_select.py — Select 12 pilot samples (3 per novel category A-D).

Selection strategy for the 4 key questions:

  Category A (3 samples) — Q1: Does injection position affect success?
    CPD-001: beginning + short context
    CPD-013: middle   + medium context
    CPD-025: end      + long context

  Category B (3 samples) — Q2: Do models decode Base64?
    OBF-001: base64 encoding
    OBF-006: ROT13 encoding
    OBF-010: HTML comment injection

  Category C (3 samples) — Q3: Do models follow non-English injections?
    CRL-001: high-resource language (French)
    CRL-006: low-resource language  (Zulu)
    CRL-011: script-switching       (Arabic)

  Category D (3 samples) — Q4: Do models process metadata fields?
    META-001: email subject line
    META-006: filename injection
    META-011: calendar event description

Usage:
  python scripts/pilot_select.py
  python scripts/pilot_select.py --benchmark data/benchmark_complete.json
  python scripts/pilot_select.py --output data/pilot_samples.json
"""

import argparse
import json
import os
import sys


# ============================================================
# Pilot sample IDs — chosen to maximize sub-type diversity
# ============================================================

PILOT_IDS = [
    # Category A: Contextual Positioning — tests position × length
    "CPD-001",   # beginning + short  → direct answer to Q1 (position effect)
    "CPD-013",   # middle   + medium  → middle position baseline
    "CPD-025",   # end      + long    → hardest: buried at end in long context

    # Category B: Obfuscation & Encoding — tests decoding ability
    "OBF-001",   # base64 encoding    → direct answer to Q2 (Base64 decoding)
    "OBF-006",   # ROT13 encoding     → alternative cipher
    "OBF-010",   # HTML comments      → markup-based hiding

    # Category C: Cross-Lingual — tests language comprehension
    "CRL-001",   # French (high-resource)    → direct answer to Q3
    "CRL-006",   # Zulu   (low-resource)     → safety gap hypothesis
    "CRL-011",   # Arabic (script-switching) → non-Latin script

    # Category D: Metadata Injection — tests field processing
    "META-001",  # email subject line   → direct answer to Q4
    "META-006",  # filename injection   → file system metadata
    "META-011",  # calendar description → calendar/event metadata
]


def select_pilot_samples(benchmark_path, output_path):
    """Extract pilot samples from the full benchmark."""

    with open(benchmark_path, encoding="utf-8") as f:
        benchmark = json.load(f)

    # Build lookup by ID
    by_id = {s["id"]: s for s in benchmark}

    # Select pilot samples
    pilot = []
    missing = []
    for pid in PILOT_IDS:
        if pid in by_id:
            pilot.append(by_id[pid])
        else:
            missing.append(pid)

    if missing:
        print(f"WARNING: {len(missing)} sample(s) not found: {missing}")
        sys.exit(1)

    # Save pilot subset
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pilot, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"Pilot samples selected: {len(pilot)}")
    print(f"\n{'ID':10s} | {'Category':25s} | {'Subcategory':40s} | {'Severity':8s}")
    print("-" * 95)
    for s in pilot:
        print(f"{s['id']:10s} | {s['category']:25s} | {s['subcategory']:40s} | {s['severity']:8s}")

    print(f"\n  Key question mapping:")
    print(f"    Q1 (position effect?):    CPD-001, CPD-013, CPD-025")
    print(f"    Q2 (Base64 decoding?):    OBF-001, OBF-006, OBF-010")
    print(f"    Q3 (non-English follow?): CRL-001, CRL-006, CRL-011")
    print(f"    Q4 (metadata process?):   META-001, META-006, META-011")
    print(f"\nSaved to: {output_path}")

    return pilot


def main():
    parser = argparse.ArgumentParser(description="Select 12 pilot samples")
    parser.add_argument("--benchmark", default="data/benchmark_complete.json",
                        help="Path to full benchmark JSON")
    parser.add_argument("--output", default="data/pilot_samples.json",
                        help="Output path for pilot subset")
    args = parser.parse_args()

    select_pilot_samples(args.benchmark, args.output)


if __name__ == "__main__":
    main()