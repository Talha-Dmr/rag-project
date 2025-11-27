#!/usr/bin/env python3
"""
Test script to verify ambiguity datasets are properly downloaded and can be loaded.
This script checks WiC, AmbigQA, ASQA, CLAMBER, and CondAmbigQA-2K datasets.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_wic_dataset(base_path: Path) -> Dict[str, Any]:
    """Load WiC (Word-in-Context) dataset."""
    print("\n" + "="*80)
    print("Testing WiC Dataset")
    print("="*80)

    results = {}
    splits = ['train', 'dev']

    for split in splits:
        data_file = base_path / split / f"{split}.data.txt"
        gold_file = base_path / split / f"{split}.gold.txt"

        if not data_file.exists() or not gold_file.exists():
            print(f"❌ {split} split files not found")
            continue

        # Read data and gold labels
        with open(data_file, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()

        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_lines = f.readlines()

        examples = []
        for data_line, gold_line in zip(data_lines, gold_lines):
            parts = data_line.strip().split('\t')
            if len(parts) >= 5:
                examples.append({
                    'target_word': parts[0],
                    'pos_tag': parts[1],
                    'indices': parts[2],
                    'sentence1': parts[3],
                    'sentence2': parts[4],
                    'label': gold_line.strip()
                })

        results[split] = examples
        print(f"\n✓ {split.upper()} split: {len(examples)} examples")

        # Show first example
        if examples:
            print(f"\nSample example from {split}:")
            ex = examples[0]
            print(f"  Target word: {ex['target_word']}")
            print(f"  Sentence 1: {ex['sentence1'][:80]}...")
            print(f"  Sentence 2: {ex['sentence2'][:80]}...")
            print(f"  Same meaning: {ex['label']}")

    total = sum(len(v) for v in results.values())
    print(f"\n📊 Total WiC examples: {total}")
    return results


def load_ambigqa_dataset(base_path: Path) -> Dict[str, Any]:
    """Load AmbigQA dataset."""
    print("\n" + "="*80)
    print("Testing AmbigQA Dataset")
    print("="*80)

    results = {}
    files = {
        'train': base_path / 'train_light.json',
        'dev': base_path / 'dev_light.json'
    }

    for split, file_path in files.items():
        if not file_path.exists():
            print(f"❌ {split} file not found: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results[split] = data
        print(f"\n✓ {split.upper()} split: {len(data)} examples")

        # Show first example
        if data:
            ex = data[0]
            print(f"\nSample example from {split}:")
            print(f"  ID: {ex.get('id', 'N/A')}")
            print(f"  Question: {ex.get('question', 'N/A')[:100]}...")

            # Check annotation structure
            annotations = ex.get('annotations', [])
            if annotations:
                ann = annotations[0]
                ann_type = ann.get('type', 'N/A')
                print(f"  Annotation type: {ann_type}")

                if ann_type == 'multipleQAs':
                    qa_pairs = ann.get('qaPairs', [])
                    print(f"  Number of QA pairs: {len(qa_pairs)}")
                    if qa_pairs:
                        print(f"  First disambiguated Q: {qa_pairs[0].get('question', 'N/A')[:80]}...")
                        print(f"  First answer: {qa_pairs[0].get('answer', 'N/A')}")
                elif ann_type == 'singleAnswer':
                    print(f"  Answer: {ann.get('answer', 'N/A')}")

    total = sum(len(v) for v in results.values())
    print(f"\n📊 Total AmbigQA examples: {total}")
    return results


def load_clamber_dataset(base_path: Path) -> List[Dict[str, Any]]:
    """Load CLAMBER benchmark dataset."""
    print("\n" + "="*80)
    print("Testing CLAMBER Dataset")
    print("="*80)

    file_path = base_path / 'clamber_benchmark.jsonl'

    if not file_path.exists():
        print(f"❌ CLAMBER file not found: {file_path}")
        return []

    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # CLAMBER uses double-encoded JSON (JSON string containing JSON)
                json_str = json.loads(line)
                json_obj = json.loads(json_str)
                examples.append(json_obj)

    print(f"\n✓ Loaded {len(examples)} examples")

    # Analyze categories and subclasses
    categories = {}
    subclasses = {}
    for ex in examples:
        cat = ex.get('category', 'unknown')
        subclass = ex.get('subclass', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        subclasses[subclass] = subclasses.get(subclass, 0) + 1

    print(f"\n📊 Category distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print(f"\n📊 Subclass distribution:")
    for subclass, count in sorted(subclasses.items()):
        print(f"  {subclass}: {count}")

    # Show first example
    if examples:
        ex = examples[0]
        print(f"\nSample example:")
        print(f"  Question: {ex.get('question', 'N/A')[:100]}...")
        print(f"  Category: {ex.get('category', 'N/A')}")
        print(f"  Subclass: {ex.get('subclass', 'N/A')}")
        print(f"  Requires clarification: {ex.get('require_clarification', 'N/A')}")
        clarifying_q = ex.get('clarifying_question', '')
        if clarifying_q:
            print(f"  Clarifying question: {clarifying_q[:100]}...")

    print(f"\n📊 Total CLAMBER examples: {len(examples)}")
    return examples


def load_asqa_dataset(base_path: Path) -> Dict[str, Any]:
    """Load ASQA (Ambiguous Long-Form QA) dataset."""
    print("\n" + "="*80)
    print("Testing ASQA Dataset")
    print("="*80)

    file_path = base_path / 'dataset' / 'ASQA.json'

    if not file_path.exists():
        print(f"❌ ASQA file not found: {file_path}")
        print("💡 To download: gsutil -m cp -r gs://gresearch/ASQA/data/ASQA.json data/ambiguity_datasets/03_asqa/dataset/")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ASQA has 'train' and 'dev' keys, each containing a dict of examples
    results = {}
    for split in ['train', 'dev']:
        if split not in data:
            print(f"❌ {split} split not found in ASQA data")
            continue

        split_data = data[split]
        results[split] = split_data
        print(f"\n✓ {split.upper()} split: {len(split_data)} examples")

        # Show first example
        if split_data:
            sample_id = list(split_data.keys())[0]
            ex = split_data[sample_id]
            print(f"\nSample example from {split}:")
            print(f"  ID: {sample_id}")
            print(f"  Ambiguous question: {ex.get('ambiguous_question', 'N/A')[:100]}...")

            qa_pairs = ex.get('qa_pairs', [])
            print(f"  Number of QA pairs: {len(qa_pairs)}")

            if qa_pairs:
                first_qa = qa_pairs[0]
                print(f"  First disambiguated Q: {first_qa.get('question', 'N/A')[:80]}...")
                print(f"  First short answers: {first_qa.get('short_answers', 'N/A')}")

            annotations = ex.get('annotations', [])
            if annotations:
                print(f"  Number of long-form annotations: {len(annotations)}")
                first_ann = annotations[0]
                long_answer = first_ann.get('long_answer', '')
                if long_answer:
                    print(f"  Long answer preview: {long_answer[:100]}...")

    total = sum(len(v) for v in results.values())
    print(f"\n📊 Total ASQA examples: {total}")
    return results


def load_condambigqa_dataset(base_path: Path) -> List[Dict[str, Any]]:
    """Load CondAmbigQA-2K dataset."""
    print("\n" + "="*80)
    print("Testing CondAmbigQA-2K Dataset")
    print("="*80)

    file_path = base_path / 'train.json'

    if not file_path.exists():
        print(f"❌ CondAmbigQA-2K file not found: {file_path}")
        print("💡 Download using: python download_condambigqa.py")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n✓ Loaded {len(data)} examples")

    # Analyze properties
    total_properties = 0
    total_contexts = 0

    for ex in data:
        properties = ex.get('properties', [])
        contexts = ex.get('ctxs', [])
        total_properties += len(properties)
        total_contexts += len(contexts)

    avg_properties = total_properties / len(data) if data else 0
    avg_contexts = total_contexts / len(data) if data else 0

    print(f"\n📊 Dataset statistics:")
    print(f"  Average properties per question: {avg_properties:.2f}")
    print(f"  Average contexts per question: {avg_contexts:.2f}")

    # Show first example
    if data:
        ex = data[0]
        print(f"\nSample example:")
        print(f"  ID: {ex.get('id', 'N/A')}")
        print(f"  Question: {ex.get('question', 'N/A')[:100]}...")
        print(f"  Number of properties: {len(ex.get('properties', []))}")
        print(f"  Number of contexts: {len(ex.get('ctxs', []))}")

        properties = ex.get('properties', [])
        if properties and len(properties) > 0:
            first_prop = properties[0]
            citations = first_prop.get('citations', [])
            if citations:
                print(f"  First citation preview: {citations[0].get('text', '')[:100]}...")

    print(f"\n📊 Total CondAmbigQA-2K examples: {len(data)}")
    return data


def main():
    """Main function to test all datasets."""
    print("\n" + "="*80)
    print("AMBIGUITY DATASETS VERIFICATION")
    print("="*80)

    base_dir = Path(__file__).parent / 'data' / 'ambiguity_datasets'

    if not base_dir.exists():
        print(f"❌ Datasets directory not found: {base_dir}")
        sys.exit(1)

    # Test each dataset
    wic_data = load_wic_dataset(base_dir / '01_wic')
    ambigqa_data = load_ambigqa_dataset(base_dir / '02_ambigqa')
    asqa_data = load_asqa_dataset(base_dir / '03_asqa')
    clamber_data = load_clamber_dataset(base_dir / '04_clamber')
    condambigqa_data = load_condambigqa_dataset(base_dir / '05_condambigqa')

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    wic_total = sum(len(v) for v in wic_data.values()) if wic_data else 0
    ambigqa_total = sum(len(v) for v in ambigqa_data.values()) if ambigqa_data else 0
    asqa_total = sum(len(v) for v in asqa_data.values()) if asqa_data else 0
    clamber_total = len(clamber_data) if clamber_data else 0
    condambigqa_total = len(condambigqa_data) if condambigqa_data else 0

    print(f"\n✓ WiC:          {wic_total:,} examples")
    print(f"✓ AmbigQA:      {ambigqa_total:,} examples")
    print(f"✓ ASQA:         {asqa_total:,} examples")
    print(f"✓ CLAMBER:      {clamber_total:,} examples")
    print(f"✓ CondAmbigQA:  {condambigqa_total:,} examples")
    print(f"\n📊 Total:       {wic_total + ambigqa_total + asqa_total + clamber_total + condambigqa_total:,} examples")

    print("\n✅ Dataset verification complete!\n")


if __name__ == "__main__":
    main()
