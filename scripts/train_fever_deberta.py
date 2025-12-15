#!/usr/bin/env python3
'''Train DeBERTa-v3-base on the FEVER dataset using Hugging Face Transformers.'''
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fine-tune DeBERTa-v3-base on FEVER',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset-name', type=str, default='pietrolesci/nli_fever', help='HF dataset repo id to load')
    parser.add_argument('--train-split', type=str, default='train', help='Name of the training split')
    parser.add_argument('--validation-split', type=str, default='dev', help='Name of the validation split')
    parser.add_argument('--test-split', type=str, default='test', help='Name of the held-out test split')
    parser.add_argument('--premise-column', type=str, default='premise', help='Column containing evidence/premise text')
    parser.add_argument('--hypothesis-column', type=str, default='hypothesis', help='Column containing the claim/hypothesis text')
    parser.add_argument('--model-name', type=str, default='microsoft/deberta-v3-base', help='Model checkpoint to fine-tune')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--output-dir', type=str, default='outputs/deberta_fever', help='Directory to store checkpoints and logs')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--warmup-ratio', type=float, default=0.06, help='Warmup ratio for the LR scheduler')
    parser.add_argument('--num-train-epochs', type=float, default=3.0, help='Number of training epochs')
    parser.add_argument('--per-device-train-batch-size', type=int, default=8, help='Per-device train batch size')
    parser.add_argument('--per-device-eval-batch-size', type=int, default=16, help='Per-device eval batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--logging-steps', type=int, default=50, help='Log every N update steps')
    parser.add_argument('--save-total-limit', type=int, default=3, help='Maximum checkpoints to keep')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 mixed precision')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for reproducibility')
    parser.add_argument('--max-train-samples', type=int, help='Optional limit on number of training samples')
    parser.add_argument('--max-eval-samples', type=int, help='Optional limit on number of validation samples')
    parser.add_argument('--max-test-samples', type=int, help='Optional limit on number of test samples')
    parser.add_argument('--resume-from-checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--report-to', type=str, nargs='*', default=['tensorboard'], help='Integrations to report metrics to')
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    )


def shorten_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), max_samples)))


def load_and_tokenize_datasets(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> Tuple[Dict[str, Dataset], Sequence[str]]:
    LOGGER.info('Loading dataset %s', args.dataset_name)
    raw_datasets: DatasetDict = load_dataset(args.dataset_name)

    for split in (args.train_split, args.validation_split, args.test_split):
        if split not in raw_datasets:
            raise ValueError(f'Split "{split}" not found in dataset. Available splits: {list(raw_datasets.keys())}')

    column_names = raw_datasets[args.train_split].column_names
    missing_columns = {args.premise_column, args.hypothesis_column, 'label'} - set(column_names)
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    if args.max_train_samples:
        raw_datasets[args.train_split] = shorten_dataset(raw_datasets[args.train_split], args.max_train_samples)
    if args.max_eval_samples:
        raw_datasets[args.validation_split] = shorten_dataset(raw_datasets[args.validation_split], args.max_eval_samples)
    if args.max_test_samples:
        raw_datasets[args.test_split] = shorten_dataset(raw_datasets[args.test_split], args.max_test_samples)

    label_feature = raw_datasets[args.train_split].features.get('label')
    if isinstance(label_feature, ClassLabel) and label_feature.names:
        label_names = list(label_feature.names)
    else:
        unique_labels = sorted(set(raw_datasets[args.train_split]['label']))
        label_names = [str(label) for label in unique_labels]

    def tokenize_function(examples):
        return tokenizer(
            examples[args.premise_column],
            examples[args.hypothesis_column],
            truncation=True,
            max_length=args.max_length,
        )

    keep_columns = {args.premise_column, args.hypothesis_column, 'label'}
    remove_columns = [col for col in column_names if col not in keep_columns]

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns,
        desc='Tokenizing dataset',
    )

    LOGGER.info(
        'Dataset sizes -> train: %d | eval: %d | test: %d',
        len(tokenized_datasets[args.train_split]),
        len(tokenized_datasets[args.validation_split]),
        len(tokenized_datasets[args.test_split]),
    )

    return (
        {
            'train': tokenized_datasets[args.train_split],
            'eval': tokenized_datasets[args.validation_split],
            'test': tokenized_datasets[args.test_split],
        },
        label_names,
    )


def build_trainer(args: argparse.Namespace):
    configure_logging()
    LOGGER.info('Torch version: %s | CUDA available: %s', torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        LOGGER.info('CUDA device: %s', torch.cuda.get_device_name(0))
    else:
        LOGGER.warning('CUDA not detected, training will fall back to CPU')

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    datasets_dict, label_names = load_and_tokenize_datasets(args, tokenizer)

    num_labels = len(label_names)
    id2label = {idx: name for idx, name in enumerate(label_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metrics = {
            'accuracy': accuracy_metric.compute(predictions=predictions, references=labels)['accuracy'],
            'f1_macro': f1_metric.compute(predictions=predictions, references=labels, average='macro')['f1'],
            'f1_weighted': f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1'],
            'precision_macro': precision_metric.compute(predictions=predictions, references=labels, average='macro')['precision'],
            'recall_macro': recall_metric.compute(predictions=predictions, references=labels, average='macro')['recall'],
        }
        return metrics

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets_dict['train'],
        eval_dataset=datasets_dict['eval'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer, datasets_dict


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer, datasets_dict = build_trainer(args)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()

    train_metrics = train_result.metrics
    train_metrics['train_samples'] = len(datasets_dict['train'])
    trainer.log_metrics('train', train_metrics)
    trainer.save_metrics('train', train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    eval_metrics['eval_samples'] = len(datasets_dict['eval'])
    trainer.log_metrics('eval', eval_metrics)
    trainer.save_metrics('eval', eval_metrics)

    test_metrics = trainer.evaluate(datasets_dict['test'], metric_key_prefix='test')
    test_metrics['test_samples'] = len(datasets_dict['test'])
    trainer.log_metrics('test', test_metrics)
    trainer.save_metrics('test', test_metrics)

    summary_path = output_path / 'summary_metrics.json'
    with summary_path.open('w', encoding='utf-8') as fp:
        json.dump({'train': train_metrics, 'eval': eval_metrics, 'test': test_metrics}, fp, indent=2)

    LOGGER.info('Training complete. Metrics saved to %s', summary_path)


if __name__ == '__main__':
    main()
