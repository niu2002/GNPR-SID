import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into a base LLM for GNPR-SID V2.")
    parser.add_argument("--base-model-path", required=True, help="Base model path")
    parser.add_argument("--lora-model-path", required=True, help="LoRA checkpoint path")
    parser.add_argument("--output-dir", required=True, help="Merged model output directory")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true", help="Only load local model/tokenizer files")
    return parser.parse_args()


def resolve_torch_dtype(torch_module, dtype_name):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch_module, dtype_name)


def merge(base_model_path, lora_model_path, output_dir, torch_dtype="bfloat16", local_files_only=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("MERGING LORA MODELS INTO BASE MODEL")
    print("=" * 40)

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
    if not os.path.exists(lora_model_path):
        raise FileNotFoundError(f"LoRA model path does not exist: {lora_model_path}")

    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    try:
        print(f"Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=resolve_torch_dtype(torch, torch_dtype),
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=local_files_only,
        )
        print(f"Base model loaded | Vocab: {len(tokenizer)}")
        print("*" * 40)

        print(f"Loading and merging lora model from: {lora_model_path}")
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            local_files_only=local_files_only,
        )
        merged_model = lora_model.merge_and_unload()
        del base_model, lora_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Lora model merged successfully")
        print("*" * 40)

        print(f"Saving final merged model to: {output_dir}")
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")
        print("*" * 40)
    except Exception:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        raise


if __name__ == "__main__":
    args = parse_args()
    merge(
        args.base_model_path,
        args.lora_model_path,
        args.output_dir,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
