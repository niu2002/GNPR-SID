import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a subset of JSON records for smoke tests.")
    parser.add_argument("--input", required=True, help="Path to input JSON list")
    parser.add_argument("--output", required=True, help="Path to output JSON list")
    parser.add_argument("--limit", type=int, required=True, help="Maximum number of records to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    rng = random.Random(args.seed)
    sampled = list(records)
    rng.shuffle(sampled)
    sampled = sampled[: min(args.limit, len(sampled))]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(args.output)
    print(f"Selected {len(sampled)} / {len(records)} records")


if __name__ == "__main__":
    main()
