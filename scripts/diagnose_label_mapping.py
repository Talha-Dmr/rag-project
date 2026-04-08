#!/usr/bin/env python3
"""Diagnostic script to verify and debug label mapping between training and inference.

Checks:
1. Config.json id2label vs training label ordering
2. Known premise/hypothesis test pairs
3. FEVER gold data accuracy under both mapping schemes
4. FinReg real data predictions under both mappings
"""

import json
import sys
import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Suppress transformers logging noise
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.hallucination_detector import HallucinationDetector
from src.training.data.base_converter import BaseConverter

MODEL_PATH = PROJECT_ROOT / "electra_deberta" / "final_fever_deberta_v3_base_model"
FEVER_TEST_PATH = PROJECT_ROOT / "data" / "training" / "nli_dataset_fever_pair" / "test.jsonl"
FINREG_PATH = PROJECT_ROOT / "data" / "domain_finreg" / "questions_finreg_conflict.jsonl"

# Training-time mapping (from BaseConverter / FeverConverter)
TRAINING_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

# --- Helpers ---

def load_raw_model(model_path: Path, device: str = "cpu"):
    """Load model and tokenizer directly, bypassing HallucinationDetector."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    model.to(device)
    return model, tokenizer


def predict_raw(model, tokenizer, premise: str, hypothesis: str, device: str = "cpu"):
    """Return raw logits, softmax probs, and predicted index."""
    inputs = tokenizer(premise, hypothesis, max_length=256, padding=True,
                       truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = F.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    return logits, probs, pred_idx


LABEL_ALIASES = HallucinationDetector.LABEL_ALIASES


def normalize_label(name: str):
    """Apply the same normalization as HallucinationDetector._normalize_label_name."""
    norm = str(name or "").strip().lower().replace("-", " ").replace("_", " ")
    norm = " ".join(norm.split())
    return LABEL_ALIASES.get(norm)


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


# --- Step 1: Verify the swap ---

def verify_swap(model_path: Path):
    print_section("STEP 1: Verify label mapping swap")

    # Read config.json directly
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    id2label_raw = config.get("id2label", {})
    print(f"\nconfig.json id2label:")
    for idx_str, name in sorted(id2label_raw.items(), key=lambda x: int(x[0])):
        canonical = normalize_label(name)
        print(f"  index {idx_str}: \"{name}\" -> canonical: \"{canonical}\"")

    print(f"\nTraining-time label order (BaseConverter):")
    for idx, name in sorted(TRAINING_MAP.items()):
        print(f"  index {idx}: {name}")

    # Show what _configure_label_mapping produces
    canonical_to_index = {}
    for raw_idx, raw_name in id2label_raw.items():
        idx = int(raw_idx)
        canonical = normalize_label(raw_name)
        if canonical:
            canonical_to_index[canonical] = idx

    print(f"\nHallucinationDetector._configure_label_mapping result:")
    for name in ["entailment", "neutral", "contradiction"]:
        idx = canonical_to_index.get(name, "?")
        training_idx = {v: k for k, v in TRAINING_MAP.items()}[name]
        match = "OK" if idx == training_idx else "MISMATCH"
        print(f"  {name}: runtime index {idx}, training index {training_idx}  [{match}]")

    # Check for mismatch
    mismatches = []
    for name in ["entailment", "neutral", "contradiction"]:
        runtime_idx = canonical_to_index.get(name)
        training_idx = {v: k for k, v in TRAINING_MAP.items()}[name]
        if runtime_idx != training_idx:
            mismatches.append(f"{name}: runtime={runtime_idx} training={training_idx}")

    if mismatches:
        print(f"\n*** SWAP CONFIRMED ***")
        for m in mismatches:
            print(f"  {m}")
    else:
        print(f"\n*** NO MISMATCH DETECTED ***")

    return canonical_to_index


# --- Step 2: Test with known pairs ---

def test_known_pairs(model_path: Path):
    print_section("STEP 2: Known test pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_raw_model(model_path, device)

    pairs = [
        ("entailment",    "The sky is blue.", "The sky is blue."),
        ("contradiction", "The sky is blue.", "The sky is red."),
        ("neutral",       "The sky is blue.", "The capital of France is Paris."),
    ]

    print(f"\n{'Expected':<15} {'Idx':>3} {'Under config.json':>20} {'Under training':>18}")
    print("-" * 70)

    for expected, premise, hypothesis in pairs:
        logits, probs, pred_idx = predict_raw(model, tokenizer, premise, hypothesis, device)

        # Interpret under config.json mapping (what runtime does now)
        raw_id2label = dict(model.config.id2label)
        raw_name = raw_id2label.get(str(pred_idx), raw_id2label.get(pred_idx, "?"))
        config_label = normalize_label(raw_name) or raw_name

        # Interpret under training mapping
        training_label = TRAINING_MAP.get(pred_idx, "?")

        print(f"{expected:<15} {pred_idx:>3} {config_label:>20} {training_label:>18}")
        print(f"  logits: [{', '.join(f'{l:.3f}' for l in logits.tolist())}]")
        print(f"  probs:  [{', '.join(f'{p:.3f}' for p in probs.tolist())}]")
        print()


# --- Step 3: Test on FEVER gold data ---

def test_fever_gold(model_path: Path, limit: int = 50):
    print_section(f"STEP 3: FEVER gold data accuracy ({limit} examples)")

    if not FEVER_TEST_PATH.exists():
        print(f"  FEVER test data not found: {FEVER_TEST_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_raw_model(model_path, device)

    examples = []
    with open(FEVER_TEST_PATH, encoding="utf-8") as f:
        for line in f:
            if len(examples) >= limit:
                break
            examples.append(json.loads(line))

    # Config.json mapping: build index -> canonical from config
    config_id2label = {}
    for idx_str, name in model.config.id2label.items():
        config_id2label[int(idx_str)] = normalize_label(name)

    correct_config = 0
    correct_training = 0
    total = 0
    pred_counts_config = Counter()
    pred_counts_training = Counter()
    gold_counts = Counter()

    for ex in examples:
        gold_label = ex["label"]
        gold_name = TRAINING_MAP[gold_label]
        gold_counts[gold_name] += 1

        _, probs, pred_idx = predict_raw(model, tokenizer, ex["premise"], ex["hypothesis"], device)

        config_pred = config_id2label.get(pred_idx, "?")
        training_pred = TRAINING_MAP.get(pred_idx, "?")

        pred_counts_config[config_pred] += 1
        pred_counts_training[training_pred] += 1

        if config_pred == gold_name:
            correct_config += 1
        if training_pred == gold_name:
            correct_training += 1
        total += 1

    acc_config = correct_config / total * 100
    acc_training = correct_training / total * 100

    print(f"\nGold label distribution:")
    for name in ["entailment", "neutral", "contradiction"]:
        print(f"  {name}: {gold_counts[name]}")

    print(f"\nPrediction distribution (config.json mapping):")
    for name in ["entailment", "neutral", "contradiction"]:
        print(f"  {name}: {pred_counts_config.get(name, 0)}")

    print(f"\nPrediction distribution (training mapping):")
    for name in ["entailment", "neutral", "contradiction"]:
        print(f"  {name}: {pred_counts_training.get(name, 0)}")

    print(f"\nAccuracy under config.json mapping: {correct_config}/{total} = {acc_config:.1f}%")
    print(f"Accuracy under training mapping:    {correct_training}/{total} = {acc_training:.1f}%")

    better = "training" if acc_training > acc_config else "config.json"
    print(f"\nBetter mapping: {better}")


# --- Step 4: Test on FinReg real data ---

def test_finreg(model_path: Path, limit: int = 5):
    print_section(f"STEP 4: FinReg predictions ({limit} pairs)")

    if not FINREG_PATH.exists():
        print(f"  FinReg data not found: {FINREG_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_raw_model(model_path, device)

    config_id2label = {}
    for idx_str, name in model.config.id2label.items():
        config_id2label[int(idx_str)] = normalize_label(name)

    # Load FinReg questions
    questions = []
    with open(FINREG_PATH, encoding="utf-8") as f:
        for line in f:
            if len(questions) >= limit:
                break
            questions.append(json.loads(line))

    # Use detector for full pipeline (with context from vector DB)
    # But for diagnostic, just show raw model output with question as hypothesis
    print(f"\nShowing raw model predictions on question text pairs.")
    print(f"(Premise = question, Hypothesis = question — tests neutral bias)\n")

    for q in questions:
        text = q.get("query", q.get("question", ""))
        _, probs, pred_idx = predict_raw(model, tokenizer, text, text, device)

        config_label = config_id2label.get(pred_idx, "?")
        training_label = TRAINING_MAP.get(pred_idx, "?")

        print(f"Q: {text[:80]}...")
        print(f"  pred_idx={pred_idx}  config.json -> {config_label}  training -> {training_label}")
        print(f"  probs: [{', '.join(f'{p:.3f}' for p in probs.tolist())}]")
        print()


# --- Main ---

def main():
    print("FEVER Hallucination Detector — Label Mapping Diagnostic")
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Also test via HallucinationDetector wrapper
    print_section("HallucinationDetector wrapper mapping")
    detector = HallucinationDetector(str(MODEL_PATH), device="cpu")
    print(f"  label_to_index: {detector.label_to_index}")
    print(f"  index_to_label: {detector.index_to_label}")

    # Step 1
    verify_swap(MODEL_PATH)

    # Step 2
    test_known_pairs(MODEL_PATH)

    # Step 3
    test_fever_gold(MODEL_PATH, limit=50)

    # Step 4
    test_finreg(MODEL_PATH, limit=5)


if __name__ == "__main__":
    main()
