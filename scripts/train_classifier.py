"""
QLoRA fine-tuning for RegClassifier using Unsloth + trl SFTTrainer.

Prerequisites:
    uv sync --group dev --group finetune
    GPU with >=6 GB VRAM (16 GB recommended)

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --model google/gemma-3-4b-it --epochs 2

Output:
    outputs/reg_classifier/   — LoRA adapter checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def format_example(ex: dict) -> str:
    """Convert instruction/response dict into the chat template format."""
    return (
        f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{ex['response']}<|im_end|>"
    )


def load_split(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main(model_name: str, epochs: int, output_dir: Path) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,          # auto-detect float16/bfloat16
        load_in_4bit=True,   # QLoRA 4-bit quantisation
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                 # LoRA rank — higher = more parameters = better but slower
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,        # scaling factor: effective_lr = lora_alpha / r * lr
        lora_dropout=0.05,    # regularisation
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves VRAM
    )

    train_data = load_split(Path("evals/finetune/train.json"))
    val_data = load_split(Path("evals/finetune/val.json"))

    train_dataset = Dataset.from_list([{"text": format_example(ex)} for ex in train_data])
    val_dataset = Dataset.from_list([{"text": format_example(ex)} for ex in val_data])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=2048,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,   # effective batch = 4*4 = 16
            num_train_epochs=epochs,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            output_dir=str(output_dir),
            report_to="none",               # no W&B; watch logs locally
        ),
    )

    print("Training started...")
    trainer.train()
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"Checkpoint saved to {output_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/reg_classifier")
    args = parser.parse_args()
    main(args.model, args.epochs, Path(args.output_dir))
