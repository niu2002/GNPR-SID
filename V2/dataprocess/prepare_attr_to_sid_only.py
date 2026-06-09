import argparse
import json
import os
from typing import List, Dict


ATTR_TO_SID_INSTRUCTION = "Given a POI attributes, describe its semantic code."


def parse_args():
    parser = argparse.ArgumentParser(description="Filter alignment data to attr->sid only for GNPR-SID V2.")
    parser.add_argument("--train-input", required=True, help="Path to train_align.json")
    parser.add_argument("--valid-input", required=True, help="Path to valid_align.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write filtered datasets")
    parser.add_argument("--train-limit", type=int, default=0, help="Optional train sample limit, 0 means keep all")
    parser.add_argument("--valid-limit", type=int, default=0, help="Optional valid sample limit, 0 means keep all")
    return parser.parse_args()


def load_json(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_attr_to_sid(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [record for record in records if record.get("instruction") == ATTR_TO_SID_INSTRUCTION]


def maybe_limit(records: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit and limit > 0:
        return records[:limit]
    return records


def write_json(path: str, records: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_records = maybe_limit(filter_attr_to_sid(load_json(args.train_input)), args.train_limit)
    valid_records = maybe_limit(filter_attr_to_sid(load_json(args.valid_input)), args.valid_limit)

    train_path = os.path.join(args.output_dir, "train_attr_to_sid.json")
    valid_path = os.path.join(args.output_dir, "valid_attr_to_sid.json")
    write_json(train_path, train_records)
    write_json(valid_path, valid_records)

    print(train_path)
    print(valid_path)
    print(f"Train attr->sid samples: {len(train_records)}")
    print(f"Valid attr->sid samples: {len(valid_records)}")


if __name__ == "__main__":
    main()
