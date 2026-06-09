import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Validate GNPR-SID V2 LLM inputs before training.")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--train-dataset", required=True, help="Path to training json")
    parser.add_argument("--valid-dataset", required=True, help="Path to validation json")
    parser.add_argument("--max-preview-chars", type=int, default=500, help="Preview text length")
    parser.add_argument("--local-files-only", action="store_true", help="Only load local tokenizer files")
    return parser.parse_args()


def format_record(record):
    return (
        f"### Instruction:\n{record['instruction'].strip()}\n\n"
        f"### Input:\n{record['input'].strip()}\n\n"
        f"### Response:\n{record['output'].strip()}<|eot_id|>"
    )


def main():
    args = parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    train_data = load_dataset("json", data_files=args.train_dataset)["train"]
    valid_data = load_dataset("json", data_files=args.valid_dataset)["train"]

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Base model path exists: {os.path.exists(args.base_model)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample = train_data[0]
    print("Sample keys:", list(sample.keys()))
    formatted = format_record(sample)
    print("Formatted preview:")
    print(formatted[: args.max_preview_chars] + ("..." if len(formatted) > args.max_preview_chars else ""))

    tokenized = tokenizer(formatted, return_tensors="pt")
    print("Tokenized length:", tokenized["input_ids"].shape[1])
    print("Last 20 tokens:", tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0][-20:]))

    print("Validation passed.")


if __name__ == "__main__":
    main()
