import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple


SID_PATTERN = re.compile(r"<([a-z])_(\d+)>")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate next-SID generation quality for GNPR-SID V2.")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter directory")
    parser.add_argument("--dataset", required=True, help="Path to llm_val.json or llm_test.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write predictions and metrics")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation max new tokens")
    parser.add_argument("--torch-dtype", default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true", help="Only load local files")
    return parser.parse_args()


def resolve_torch_dtype(torch_module, dtype_name):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch_module, dtype_name)


def build_prompt(record: Dict[str, str]) -> str:
    return (
        f"### Instruction:\n{record['instruction'].strip()}\n\n"
        f"### Input:\n{record['input'].strip()}\n\n"
        f"### Response:\n"
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def parse_sid(text: str) -> List[Tuple[str, int]]:
    return [(letter, int(value)) for letter, value in SID_PATTERN.findall(text)]


def extract_first_sid(text: str) -> str:
    matches = SID_PATTERN.findall(text)
    if not matches:
        return ""
    return "".join(f"<{letter}_{value}>" for letter, value in matches)


def is_valid_sid(text: str) -> bool:
    cleaned = text.strip()
    matches = SID_PATTERN.findall(cleaned)
    return bool(matches) and "".join(f"<{l}_{v}>" for l, v in matches) == cleaned


def sid_component_scores(gold_sid: str, pred_sid: str) -> Dict[str, float]:
    gold = dict(parse_sid(gold_sid))
    pred = dict(parse_sid(pred_sid))
    labels = sorted(set(gold.keys()) | set(pred.keys()))
    return {
        label: 1.0 if (label in gold and label in pred and gold[label] == pred[label]) else 0.0
        for label in labels
    }


def sample_records(records: List[Dict[str, str]], sample_size: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    copied = list(records)
    rng.shuffle(copied)
    return copied[: min(sample_size, len(copied))]


def batched(items: List[Dict[str, str]], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield items[idx: idx + batch_size]


def main():
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset, "r", encoding="utf-8") as f:
        records = json.load(f)

    eval_records = sample_records(records, args.sample_size, args.seed)
    if not eval_records:
        raise ValueError("No evaluation records were selected.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=resolve_torch_dtype(torch, args.torch_dtype),
        local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    predictions = []
    for batch in batched(eval_records, args.batch_size):
        prompts = [build_prompt(record) for record in batch]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
        for idx, record in enumerate(batch):
            generated_ids = generated[idx][prompt_lengths[idx]:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            gold_text = record["output"].strip()
            extracted_sid = extract_first_sid(pred_text)
            predictions.append(
                {
                    "instruction": record["instruction"],
                    "input": record["input"],
                    "gold_output": gold_text,
                    "pred_output": pred_text,
                    "extracted_sid": extracted_sid,
                    "is_exact_match": normalize_text(pred_text) == normalize_text(gold_text),
                }
            )

    metrics: Dict[str, object] = {
        "total_samples": len(predictions),
        "sample_size_requested": args.sample_size,
        "exact_match": sum(p["is_exact_match"] for p in predictions) / len(predictions),
        "valid_sid_rate": sum(is_valid_sid(p["pred_output"]) for p in predictions) / len(predictions),
        "extractable_sid_rate": sum(bool(p["extracted_sid"]) for p in predictions) / len(predictions),
        "extracted_sid_exact_match": sum(p["extracted_sid"] == p["gold_output"] for p in predictions) / len(predictions),
    }

    component_hits = {}
    component_counts = {}
    for pred in predictions:
        sid_for_score = pred["pred_output"] if is_valid_sid(pred["pred_output"]) else pred["extracted_sid"]
        scores = sid_component_scores(pred["gold_output"], sid_for_score)
        for label, score in scores.items():
            component_hits[label] = component_hits.get(label, 0.0) + score
            component_counts[label] = component_counts.get(label, 0) + 1
    metrics["sid_component_accuracy"] = {
        label: component_hits[label] / component_counts[label]
        for label in sorted(component_counts.keys())
    }

    predictions_path = os.path.join(args.output_dir, "next_sid_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(args.output_dir, "next_sid_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(predictions_path)
    print(metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
