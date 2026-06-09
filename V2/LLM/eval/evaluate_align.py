import argparse
import ast
import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple


SID_PATTERN = re.compile(r"<([a-z])_(\d+)>")
CATEGORY_PATTERN = re.compile(r"Category:\s*([^;{}]+)")
REGION_PATTERN = re.compile(r"Region:\s*([^;{}]+)")
LAT_PATTERN = re.compile(r"Latitude:\s*([-+]?\d+(?:\.\d+)?)")
LON_PATTERN = re.compile(r"Longitude:\s*([-+]?\d+(?:\.\d+)?)")
VISIT_PATTERN = re.compile(r"Visit_time_and_count:\s*(\{.*\})")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SID alignment generation quality for GNPR-SID V2.")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter directory")
    parser.add_argument("--dataset", required=True, help="Path to alignment evaluation json")
    parser.add_argument("--output-dir", required=True, help="Directory to write predictions and metrics")
    parser.add_argument("--samples-per-task", type=int, default=50, help="Number of samples per task to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation max new tokens")
    parser.add_argument("--torch-dtype", default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true", help="Only load local files")
    parser.add_argument("--latlon-tolerance", type=float, default=1e-3, help="Latitude/longitude tolerance")
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


def detect_task_type(record: Dict[str, str]) -> str:
    instruction = record["instruction"].lower()
    if "given a poi attributes" in instruction:
        return "attr_to_sid"
    if "given a semantic code" in instruction:
        return "sid_to_attr"
    return "unknown"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def parse_sid(text: str) -> List[Tuple[str, int]]:
    return [(letter, int(value)) for letter, value in SID_PATTERN.findall(text)]


def is_valid_sid(text: str) -> bool:
    cleaned = text.strip()
    matches = SID_PATTERN.findall(cleaned)
    return bool(matches) and "".join(f"<{l}_{v}>" for l, v in matches) == cleaned


def sid_component_scores(gold_sid: str, pred_sid: str) -> Dict[str, Optional[float]]:
    gold = dict(parse_sid(gold_sid))
    pred = dict(parse_sid(pred_sid))
    labels = sorted(set(gold.keys()) | set(pred.keys()))
    scores = {}
    for label in labels:
        if label not in gold or label not in pred:
            scores[label] = 0.0
        else:
            scores[label] = 1.0 if gold[label] == pred[label] else 0.0
    return scores


def parse_attr_output(text: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    category = CATEGORY_PATTERN.search(text)
    region = REGION_PATTERN.search(text)
    lat = LAT_PATTERN.search(text)
    lon = LON_PATTERN.search(text)
    visit = VISIT_PATTERN.search(text)

    if category:
        result["category"] = category.group(1).strip()
    if region:
        result["region"] = region.group(1).strip()
    if lat:
        result["latitude"] = float(lat.group(1))
    if lon:
        result["longitude"] = float(lon.group(1))
    if visit:
        raw_visit = visit.group(1).strip()
        result["visit_time_and_count_raw"] = raw_visit
        try:
            result["visit_time_and_count"] = ast.literal_eval(raw_visit)
        except Exception:
            pass
    return result


def float_close(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, abs_tol=tol)


def sample_records(records: List[Dict[str, str]], samples_per_task: int, seed: int) -> List[Dict[str, str]]:
    grouped = {"attr_to_sid": [], "sid_to_attr": []}
    for record in records:
        task_type = detect_task_type(record)
        if task_type in grouped:
            grouped[task_type].append(record)

    rng = random.Random(seed)
    sampled = []
    for task_type, group in grouped.items():
        rng.shuffle(group)
        take = min(samples_per_task, len(group))
        for item in group[:take]:
            copied = dict(item)
            copied["task_type"] = task_type
            sampled.append(copied)

    rng.shuffle(sampled)
    return sampled


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

    eval_records = sample_records(records, args.samples_per_task, args.seed)
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
            predictions.append(
                {
                    "task_type": record["task_type"],
                    "instruction": record["instruction"],
                    "input": record["input"],
                    "gold_output": gold_text,
                    "pred_output": pred_text,
                    "is_exact_match": normalize_text(pred_text) == normalize_text(gold_text),
                }
            )

    metrics: Dict[str, object] = {
        "total_samples": len(predictions),
        "samples_per_task_requested": args.samples_per_task,
        "attr_to_sid_count": sum(p["task_type"] == "attr_to_sid" for p in predictions),
        "sid_to_attr_count": sum(p["task_type"] == "sid_to_attr" for p in predictions),
    }

    attr_preds = [p for p in predictions if p["task_type"] == "attr_to_sid"]
    sid_preds = [p for p in predictions if p["task_type"] == "sid_to_attr"]

    if attr_preds:
        metrics["attr_to_sid_exact_match"] = sum(p["is_exact_match"] for p in attr_preds) / len(attr_preds)
        metrics["valid_sid_rate"] = sum(is_valid_sid(p["pred_output"]) for p in attr_preds) / len(attr_preds)

        component_hits = {}
        component_counts = {}
        for pred in attr_preds:
            scores = sid_component_scores(pred["gold_output"], pred["pred_output"])
            for label, score in scores.items():
                component_hits[label] = component_hits.get(label, 0.0) + (score or 0.0)
                component_counts[label] = component_counts.get(label, 0) + 1
        metrics["sid_component_accuracy"] = {
            label: (component_hits[label] / component_counts[label])
            for label in sorted(component_counts.keys())
        }

    if sid_preds:
        metrics["sid_to_attr_exact_match"] = sum(p["is_exact_match"] for p in sid_preds) / len(sid_preds)

        category_hits = 0
        region_hits = 0
        lat_hits = 0
        lon_hits = 0
        latlon_hits = 0
        visit_presence_hits = 0
        visit_parse_hits = 0

        for pred in sid_preds:
            gold_attr = parse_attr_output(pred["gold_output"])
            pred_attr = parse_attr_output(pred["pred_output"])

            if gold_attr.get("category") == pred_attr.get("category"):
                category_hits += 1
            if gold_attr.get("region") == pred_attr.get("region"):
                region_hits += 1
            if float_close(gold_attr.get("latitude"), pred_attr.get("latitude"), args.latlon_tolerance):
                lat_hits += 1
            if float_close(gold_attr.get("longitude"), pred_attr.get("longitude"), args.latlon_tolerance):
                lon_hits += 1
            if (
                float_close(gold_attr.get("latitude"), pred_attr.get("latitude"), args.latlon_tolerance)
                and float_close(gold_attr.get("longitude"), pred_attr.get("longitude"), args.latlon_tolerance)
            ):
                latlon_hits += 1
            if "visit_time_and_count_raw" in pred_attr:
                visit_presence_hits += 1
            if "visit_time_and_count" in pred_attr:
                visit_parse_hits += 1

        denom = len(sid_preds)
        metrics["sid_to_attr_field_match"] = {
            "category_acc": category_hits / denom,
            "region_acc": region_hits / denom,
            "latitude_match_rate": lat_hits / denom,
            "longitude_match_rate": lon_hits / denom,
            "latlon_match_rate": latlon_hits / denom,
            "visit_field_presence_rate": visit_presence_hits / denom,
            "visit_field_parse_rate": visit_parse_hits / denom,
        }

    predictions_path = os.path.join(args.output_dir, "align_eval_predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(args.output_dir, "align_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(predictions_path)
    print(metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
