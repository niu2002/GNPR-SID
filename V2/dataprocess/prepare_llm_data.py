import argparse
import ast
import json
import os
from ast import literal_eval

import pandas as pd


INSTRUCTION = (
    "Here is a record of a user's POI accesses, your task is based on the history "
    "to predict the POI that the user is likely to access at the specified time."
)


def encode_item(item):
    labels = "abcdefghijklmnopqrstuvwxyz"
    if len(item) > len(labels):
        raise ValueError(f"SID length {len(item)} exceeds supported tag alphabet size.")
    return "".join(f"<{labels[i]}_{val}>" for i, val in enumerate(item))


def load_pid_to_code(codebook_path):
    df_code = pd.read_csv(codebook_path)
    pid2code = {}
    for _, row in df_code.iterrows():
        pid = row["pid"]
        sid_str = row["sid"]
        item_list = literal_eval(sid_str)
        pid2code[pid] = encode_item(item_list)
    return pid2code


def convert_csv_to_llm_json_sid(
    input_file,
    output_file,
    pid2code,
    instruction=INSTRUCTION,
    keep_last_k_per_user=None,
    skip_unknown=True,
):
    df = pd.read_csv(input_file)
    if keep_last_k_per_user is not None:
        df = df.groupby("UserId", group_keys=False).tail(keep_last_k_per_user).reset_index(drop=True)

    samples = []
    skipped_rows = 0
    for row in df.itertuples(index=False):
        uid = row.UserId
        try:
            poi_seq = ast.literal_eval(row.sequence_PoiId)
            time_seq = ast.literal_eval(row.sequence_UTCTimeOffset)
        except Exception:
            skipped_rows += 1
            continue

        if len(poi_seq) < 2 or len(poi_seq) != len(time_seq):
            skipped_rows += 1
            continue

        history_pids = poi_seq[:-1]
        history_times = time_seq[:-1]
        target_time = time_seq[-1]
        target_pid = poi_seq[-1]

        history_codes = []
        bad_flag = False
        for pid in history_pids:
            if pid in pid2code:
                history_codes.append(pid2code[pid])
            elif skip_unknown:
                bad_flag = True
                break
            else:
                history_codes.append(f"<unk_{pid}>")

        if bad_flag:
            skipped_rows += 1
            continue

        if target_pid in pid2code:
            target_code = pid2code[target_pid]
        elif skip_unknown:
            skipped_rows += 1
            continue
        else:
            target_code = f"<unk_{target_pid}>"

        history_text = ", ".join(
            f"{t} visited {p}" for t, p in zip(history_times, history_codes)
        )
        input_text = (
            f"User_{uid} checkin history: {history_text}.\n"
            f"When {target_time} user_{uid} is likely to visit:"
        )

        samples.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": target_code,
            }
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(samples)} samples to {output_file}")
    print(f"Skipped {skipped_rows} rows")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LLM fine-tuning JSON data for GNPR-SID V2.")
    parser.add_argument("--sid-csv", required=True, help="Path to sid.csv")
    parser.add_argument("--train-seq", required=True, help="Path to train_poi_sequence.csv")
    parser.add_argument("--val-seq", required=True, help="Path to validation_poi_sequence.csv")
    parser.add_argument("--test-seq", required=True, help="Path to test_poi_sequence.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to write llm_train/val/test.json")
    parser.add_argument("--train-last-k", type=int, default=5, help="Keep last K train samples per user")
    parser.add_argument("--skip-unknown", type=str, default="true", help="Whether to skip POIs without SID")
    return parser.parse_args()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
    args = parse_args()
    skip_unknown = str2bool(args.skip_unknown)
    pid2code = load_pid_to_code(args.sid_csv)

    train_out = os.path.join(args.output_dir, "llm_train.json")
    val_out = os.path.join(args.output_dir, "llm_val.json")
    test_out = os.path.join(args.output_dir, "llm_test.json")

    convert_csv_to_llm_json_sid(
        args.train_seq,
        train_out,
        pid2code=pid2code,
        keep_last_k_per_user=args.train_last_k,
        skip_unknown=skip_unknown,
    )
    convert_csv_to_llm_json_sid(
        args.val_seq,
        val_out,
        pid2code=pid2code,
        skip_unknown=skip_unknown,
    )
    convert_csv_to_llm_json_sid(
        args.test_seq,
        test_out,
        pid2code=pid2code,
        skip_unknown=skip_unknown,
    )

    print(train_out)
    print(val_out)
    print(test_out)


if __name__ == "__main__":
    main()
