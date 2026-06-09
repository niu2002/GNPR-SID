import argparse
import os
import random


def format_fn(batch):
    out = []
    for ins, inp, resp in zip(batch["instruction"], batch["input"], batch["output"]):
        text = (
            f"### Instruction:\n{ins.strip()}\n\n"
            f"### Input:\n{inp.strip()}\n\n"
            f"### Response:\n{resp.strip()}<|eot_id|>"
        )
        out.append(text)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Train SID-to-LLM alignment adapter for GNPR-SID V2.")
    parser.add_argument("--base-model", required=True, help="Base model path or HF/modelscope identifier")
    parser.add_argument("--train-dataset", required=True, help="Path to train_align.json")
    parser.add_argument("--valid-dataset", required=True, help="Path to valid_align.json")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="sid_alignment")
    parser.add_argument("--wandb-name", default="sid_mapping")
    parser.add_argument("--report-to", default="none", help="Training report target, e.g. none or wandb")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--local-files-only", action="store_true", help="Only load local model/tokenizer files")
    return parser.parse_args()


def resolve_torch_dtype(torch_module, dtype_name):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch_module, dtype_name)


def train(args):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    random.seed(args.seed)
    os.environ["WANDB_PROJECT"] = args.wandb_project

    train_data = load_dataset("json", data_files=args.train_dataset)["train"]
    val_data = load_dataset("json", data_files=args.valid_dataset)["train"]

    print("Example data sample:")
    print(train_data[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=resolve_torch_dtype(torch, args.torch_dtype),
        local_files_only=args.local_files_only,
    )
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["embed_tokens"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForCompletionOnlyLM(
        response_template="### Response:\n",
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        logging_steps=5,
        warmup_steps=180,
        bf16=(args.torch_dtype == "bfloat16"),
        fp16=(args.torch_dtype == "float16"),
        run_name=args.wandb_name,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        tokenizer=tokenizer,
        formatting_func=format_fn,
        max_seq_length=args.cutoff_len,
        data_collator=collator,
    )

    example = format_fn(train_data[:1])
    print("Example formatted prompt:")
    print(example[0][:500] + "...")
    input_ids = tokenizer(example[0], return_tensors="pt")["input_ids"]
    print("Tokenized length:", input_ids.shape[1])

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training finished!")


if __name__ == "__main__":
    train(parse_args())
