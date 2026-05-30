from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


DEFAULT_MODEL_NAME = "VinAI/bartpho-syllable"
DEFAULT_OUTPUT_DIR = "models/bartpho_sentence"
SOURCE_COLUMN = "gloss_sequence"
TARGET_COLUMN = "vietnamese_sentence"
PREFIX = "gloss: "


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dataset: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if SOURCE_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Dataset phải có 2 cột: {SOURCE_COLUMN}, {TARGET_COLUMN}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    df = df[[SOURCE_COLUMN, TARGET_COLUMN]].copy()
    df[SOURCE_COLUMN] = df[SOURCE_COLUMN].astype(str).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()

    df = df[
        (df[SOURCE_COLUMN] != "")
        & (df[TARGET_COLUMN] != "")
        & (df[SOURCE_COLUMN].str.lower() != "nan")
        & (df[TARGET_COLUMN].str.lower() != "nan")
    ]

    df = df.drop_duplicates().reset_index(drop=True)

    return df


def load_train_val_test(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.train and args.val and args.test:
        train_df = read_dataset(args.train)
        val_df = read_dataset(args.val)
        test_df = read_dataset(args.test)

        return train_df, val_df, test_df

    if not args.full:
        raise ValueError(
            "Bạn cần truyền --train --val --test hoặc truyền --full."
        )

    full_path = Path(args.full)

    if not full_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file full dataset: {full_path}")

    full_df = pd.read_csv(full_path, encoding="utf-8-sig")

    if SOURCE_COLUMN not in full_df.columns or TARGET_COLUMN not in full_df.columns:
        raise ValueError(
            f"Dataset phải có 2 cột: {SOURCE_COLUMN}, {TARGET_COLUMN}. "
            f"Các cột hiện có: {list(full_df.columns)}"
        )

    full_df[SOURCE_COLUMN] = full_df[SOURCE_COLUMN].astype(str).str.strip()
    full_df[TARGET_COLUMN] = full_df[TARGET_COLUMN].astype(str).str.strip()

    full_df = full_df[
        (full_df[SOURCE_COLUMN] != "")
        & (full_df[TARGET_COLUMN] != "")
        & (full_df[SOURCE_COLUMN].str.lower() != "nan")
        & (full_df[TARGET_COLUMN].str.lower() != "nan")
    ]

    full_df = full_df.drop_duplicates().reset_index(drop=True)

    if "split" in full_df.columns:
        train_df = full_df[full_df["split"].astype(str).str.lower() == "train"]
        val_df = full_df[
            full_df["split"].astype(str).str.lower().isin(["val", "valid", "validation"])
        ]
        test_df = full_df[full_df["split"].astype(str).str.lower() == "test"]

        train_df = train_df[[SOURCE_COLUMN, TARGET_COLUMN]].reset_index(drop=True)
        val_df = val_df[[SOURCE_COLUMN, TARGET_COLUMN]].reset_index(drop=True)
        test_df = test_df[[SOURCE_COLUMN, TARGET_COLUMN]].reset_index(drop=True)

        if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
            return train_df, val_df, test_df

    temp_df, test_df = train_test_split(
        full_df[[SOURCE_COLUMN, TARGET_COLUMN]],
        test_size=0.10,
        random_state=args.seed,
        shuffle=True,
    )

    train_df, val_df = train_test_split(
        temp_df,
        test_size=0.11,
        random_state=args.seed,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def build_preprocess_function(tokenizer, max_source_length: int, max_target_length: int):
    def preprocess_function(batch: Dict[str, list]) -> Dict[str, list]:
        sources = [
            PREFIX + str(text).strip()
            for text in batch[SOURCE_COLUMN]
        ]

        targets = [
            str(text).strip()
            for text in batch[TARGET_COLUMN]
        ]

        model_inputs = tokenizer(
            sources,
            max_length=max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return preprocess_function


def generate_examples(
    model,
    tokenizer,
    df: pd.DataFrame,
    output_path: str | Path,
    max_source_length: int,
    max_target_length: int,
    num_examples: int = 30,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    sample_df = df.head(num_examples).copy()

    lines = []

    model.eval()

    for _, row in sample_df.iterrows():
        gloss = str(row[SOURCE_COLUMN]).strip()
        target = str(row[TARGET_COLUMN]).strip()

        source_text = PREFIX + gloss

        inputs = tokenizer(
            source_text,
            return_tensors="pt",
            max_length=max_source_length,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        prediction = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        lines.append(f"Gloss: {gloss}")
        lines.append(f"Target: {target}")
        lines.append(f"Predict: {prediction}")
        lines.append("-" * 80)

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BARTpho cho bài toán gloss sequence -> câu tiếng Việt."
    )

    parser.add_argument("--train", default="data/sentence_pairs_train_simple.csv")
    parser.add_argument("--val", default="data/sentence_pairs_val_simple.csv")
    parser.add_argument("--test", default="data/sentence_pairs_test_simple.csv")
    parser.add_argument("--full", default=None)

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)

    parser.add_argument("--max-source-length", type=int, default=96)
    parser.add_argument("--max-target-length", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAIN BARTPHO GLOSS -> VIETNAMESE SENTENCE")
    print("=" * 80)
    print("Model:", args.model_name)
    print("Output:", output_dir)

    train_df, val_df, test_df = load_train_val_test(args)

    print("\nDataset:")
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)

    print("\nExample:")
    print(train_df.head(5))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=False,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_dataset = dataframe_to_dataset(train_df)
    val_dataset = dataframe_to_dataset(val_df)
    test_dataset = dataframe_to_dataset(test_df)

    preprocess_function = build_preprocess_function(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    fp16 = torch.cuda.is_available()

    print("\nDevice:")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: Không có. Train sẽ chậm hơn trên CPU.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStart training...")
    train_result = trainer.train()

    print("\nEvaluate on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)

    print("\nEvaluate on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    print("\nSaving final model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "train": train_result.metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "num_train_samples": len(train_df),
        "num_val_samples": len(val_df),
        "num_test_samples": len(test_df),
        "source_column": SOURCE_COLUMN,
        "target_column": TARGET_COLUMN,
        "prefix": PREFIX,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }

    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    print("\nGenerate sample predictions...")
    generate_examples(
        model=trainer.model,
        tokenizer=tokenizer,
        df=test_df,
        output_path=output_dir / "sample_predictions.txt",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_examples=40,
    )

    print("\nDone.")
    print("Model saved to:", output_dir)
    print("Metrics:", output_dir / "training_metrics.json")
    print("Sample predictions:", output_dir / "sample_predictions.txt")


if __name__ == "__main__":
    main()
