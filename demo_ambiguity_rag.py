#!/usr/bin/env python3
"""
Demonstration script showing how to use ambiguity datasets with the RAG system.
This script loads AmbigQA examples and shows how they can be indexed and queried.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_ambigqa_sample(file_path: Path, max_examples: int = 20) -> List[Dict[str, Any]]:
    """Load a sample of AmbigQA examples."""
    print(f"\n📥 Loading AmbigQA sample from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Take first max_examples
    sample = data[:max_examples]
    print(f"✓ Loaded {len(sample)} examples")

    return sample


def convert_ambigqa_to_documents(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert AmbigQA examples to document format suitable for RAG indexing.

    For each ambiguous question, we create multiple documents:
    - One for the original ambiguous question
    - One for each disambiguated QA pair (if multipleQAs type)
    """
    documents = []

    for ex in examples:
        question_id = ex.get('id', 'unknown')
        ambiguous_question = ex.get('question', '')

        annotations = ex.get('annotations', [])
        for ann_idx, ann in enumerate(annotations):
            ann_type = ann.get('type', '')

            if ann_type == 'multipleQAs':
                # Create a document for each disambiguated QA pair
                qa_pairs = ann.get('qaPairs', [])

                for qa_idx, qa_pair in enumerate(qa_pairs):
                    disambiguated_q = qa_pair.get('question', '')
                    answers = qa_pair.get('answer', [])

                    # Create document text
                    content = f"""Question (Ambiguous): {ambiguous_question}

Disambiguated Question: {disambiguated_q}

Answer: {', '.join(answers) if isinstance(answers, list) else answers}

Note: This question is ambiguous. The original question "{ambiguous_question}" can be interpreted in multiple ways.
This is one specific interpretation."""

                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': 'AmbigQA',
                            'question_id': question_id,
                            'ambiguous_question': ambiguous_question,
                            'disambiguated_question': disambiguated_q,
                            'answers': answers,
                            'annotation_type': ann_type,
                            'annotation_idx': ann_idx,
                            'qa_pair_idx': qa_idx
                        }
                    })

            elif ann_type == 'singleAnswer':
                # Single answer - not ambiguous
                answers = ann.get('answer', [])

                content = f"""Question: {ambiguous_question}

Answer: {', '.join(answers) if isinstance(answers, list) else answers}

Note: This question has a single, unambiguous answer."""

                documents.append({
                    'content': content,
                    'metadata': {
                        'source': 'AmbigQA',
                        'question_id': question_id,
                        'question': ambiguous_question,
                        'answers': answers,
                        'annotation_type': ann_type,
                        'annotation_idx': ann_idx
                    }
                })

    return documents


def display_documents(documents: List[Dict[str, Any]], max_display: int = 3):
    """Display sample documents."""
    print(f"\n📄 Sample documents (showing {min(max_display, len(documents))} of {len(documents)}):")
    print("="*80)

    for idx, doc in enumerate(documents[:max_display], 1):
        print(f"\nDocument {idx}:")
        print("-" * 80)
        print(doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'])
        print(f"\nMetadata: {doc['metadata']['source']}, Type: {doc['metadata'].get('annotation_type', 'N/A')}")
        print("="*80)


def analyze_ambiguity_patterns(examples: List[Dict[str, Any]]):
    """Analyze patterns in the ambiguity dataset."""
    print("\n📊 Analyzing Ambiguity Patterns")
    print("="*80)

    total_examples = len(examples)
    single_answer = 0
    multiple_qas = 0
    total_disambiguations = 0

    for ex in examples:
        annotations = ex.get('annotations', [])
        for ann in annotations:
            ann_type = ann.get('type', '')
            if ann_type == 'singleAnswer':
                single_answer += 1
            elif ann_type == 'multipleQAs':
                multiple_qas += 1
                qa_pairs = ann.get('qaPairs', [])
                total_disambiguations += len(qa_pairs)

    print(f"\nTotal examples: {total_examples}")
    print(f"Single answer (unambiguous): {single_answer}")
    print(f"Multiple QAs (ambiguous): {multiple_qas}")
    print(f"Total disambiguations: {total_disambiguations}")

    if multiple_qas > 0:
        avg_disambiguations = total_disambiguations / multiple_qas
        print(f"Average disambiguations per ambiguous question: {avg_disambiguations:.2f}")

    print("\n" + "="*80)


def main():
    """Main demonstration function."""
    print("\n" + "="*80)
    print("AMBIGUITY DATASETS + RAG SYSTEM DEMONSTRATION")
    print("="*80)

    # Load AmbigQA sample
    base_dir = Path(__file__).parent / 'data' / 'ambiguity_datasets' / '02_ambigqa'
    train_file = base_dir / 'train_light.json'

    if not train_file.exists():
        print(f"❌ AmbigQA training file not found: {train_file}")
        sys.exit(1)

    # Load sample
    examples = load_ambigqa_sample(train_file, max_examples=20)

    # Analyze patterns
    analyze_ambiguity_patterns(examples)

    # Convert to documents
    print("\n📝 Converting to RAG-compatible documents...")
    documents = convert_ambigqa_to_documents(examples)
    print(f"✓ Created {len(documents)} documents from {len(examples)} examples")

    # Display sample documents
    display_documents(documents, max_display=3)

    # Show how to use with RAG
    print("\n💡 How to Use with RAG System:")
    print("="*80)
    print("""
To index these documents into your RAG system:

1. Save documents to a JSON file:

   import json
   with open('ambigqa_docs.json', 'w') as f:
       json.dump(documents, f, indent=2)

2. Use the DataManager to load:

   from src.dataset.data_manager import DataManager

   dm = DataManager()
   loaded_docs = dm.load_from_path('ambigqa_docs.json', loader_type='json')

3. Index into RAG pipeline:

   from src.rag.rag_pipeline import RAGPipeline

   pipeline = RAGPipeline('config/base_config.yaml')
   num_chunks = pipeline.index_documents('ambigqa_docs.json')

4. Query with ambiguous questions:

   result = pipeline.query("When did the Simpsons first air?")
   print(result['answer'])

   # The system will retrieve multiple disambiguations if available

Benefits of using ambiguity datasets:
- RAG system learns to handle multiple interpretations
- Improves robustness to ambiguous queries
- Provides context for disambiguation
- Helps LLM generate more comprehensive answers
    """)
    print("="*80)

    print("\n✅ Demonstration complete!\n")


if __name__ == "__main__":
    main()
