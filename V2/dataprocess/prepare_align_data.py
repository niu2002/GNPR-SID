import argparse
import json
import os
import random
from ast import literal_eval

import pandas as pd


def code_to_tag(code_list):
    letters = "abcdefghijklmnopqrstuvwxyz"
    if len(code_list) > len(letters):
        raise ValueError(f"SID length {len(code_list)} exceeds supported tag alphabet size.")
    return "".join(f"<{letters[i]}_{value}>" for i, value in enumerate(code_list))


def build_mapping(poi_info_path, sid_csv_path):
    poi_info = pd.read_csv(poi_info_path)
    poi_codes = pd.read_csv(sid_csv_path)
    poi_codes["sid"] = poi_codes["sid"].apply(literal_eval)

    merged = poi_info.merge(poi_codes, on="pid", how="inner")
    mapping = {}
    for _, row in merged.iterrows():
        code_key = str(list(tuple(row["sid"])))
        mapping[code_key] = {
            "category": row["category"],
            "region": row["region"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "visit_time_and_count": row["visit_time_and_count"],
        }
    return mapping


def build_alignment_dataset(mapping):
    dataset = []
    for code_key, meta in mapping.items():
        code_list = json.loads(code_key.replace("'", '"'))
        tag = code_to_tag(code_list)

        attr_text = (
            "{"
            f"Category: {meta['category']}; "
            f"Region: {meta['region']}; "
            f"Latitude: {meta['latitude']}; "
            f"Longitude: {meta['longitude']}; "
            f"Visit_time_and_count: {meta['visit_time_and_count']}"
            "}"
        )

        dataset.append(
            {
                "instruction": "Given a POI attributes, describe its semantic code.",
                "input": f"Can you based on the attributes {attr_text} to predict the POI semantic code?",
                "output": tag,
            }
        )
        dataset.append(
            {
                "instruction": "Given a semantic code, describe its POI attributes.",
                "input": f"Can you describe the POI semantic code {tag} attributes?",
                "output": attr_text,
            }
        )
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SID alignment datasets for GNPR-SID V2.")
    parser.add_argument("--poi-info", required=True, help="Path to poi_info.csv")
    parser.add_argument("--sid-csv", required=True, help="Path to sid.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write mapping and alignment JSON files")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mapping = build_mapping(args.poi_info, args.sid_csv)
    mapping_path = os.path.join(args.output_dir, "semantic_code_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    dataset = build_alignment_dataset(mapping)
    dataset_path = os.path.join(args.output_dir, "semantic_instruction_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    rng = random.Random(args.seed)
    rng.shuffle(dataset)
    valid_size = int(len(dataset) * args.valid_ratio)
    valid_data = dataset[:valid_size]
    train_data = dataset[valid_size:]

    train_path = os.path.join(args.output_dir, "train_align.json")
    valid_path = os.path.join(args.output_dir, "valid_align.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(valid_path, "w", encoding="utf-8") as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    print(mapping_path)
    print(dataset_path)
    print(train_path)
    print(valid_path)
    print(f"Alignment dataset size: {len(dataset)}")
    print(f"Train size: {len(train_data)} | Valid size: {len(valid_data)}")


if __name__ == "__main__":
    main()
