#!/usr/bin/env python3
"""
Download CondAmbigQA-2K dataset from HuggingFace.
This dataset contains 2000 examples of conditionally ambiguous questions.
"""

import json
from pathlib import Path
from datasets import load_dataset

def main():
    print("\n" + "="*80)
    print("Downloading CondAmbigQA-2K from HuggingFace")
    print("="*80)

    # Create output directory
    output_dir = Path(__file__).parent / 'data' / 'ambiguity_datasets' / '05_condambigqa'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Loading dataset from HuggingFace...")
    try:
        # Load dataset from HuggingFace
        ds = load_dataset("Apocalypse-AGI-DAO/CondAmbigQA-2K")

        print(f"✓ Dataset loaded successfully")
        print(f"\nDataset splits: {list(ds.keys())}")

        # Save each split to JSON
        for split_name, split_data in ds.items():
            output_file = output_dir / f"{split_name}.json"

            # Convert to list of dictionaries
            data_list = [example for example in split_data]

            # Save to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved {split_name} split: {len(data_list)} examples → {output_file}")

            # Show sample
            if data_list:
                print(f"\nSample from {split_name}:")
                sample = data_list[0]
                print(f"  Keys: {list(sample.keys())}")
                if 'question' in sample:
                    print(f"  Question: {sample['question'][:100]}...")

        print("\n" + "="*80)
        print("✅ CondAmbigQA-2K dataset downloaded successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Accept the dataset terms on HuggingFace")
        print("3. Check your internet connection")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
