# Ambiguity Datasets

This directory contains datasets focused on **Aleatorik (inherent) ambiguity** in natural language questions. These datasets are valuable for training and evaluating RAG systems that need to handle ambiguous queries.

## What is Aleatorik Ambiguity?

**Aleatorik ambiguity** refers to inherent uncertainty in language that cannot be reduced with more data. Examples include:
- Words with multiple meanings (polysemy): "bank" → financial institution vs. river bank
- Questions with multiple valid interpretations: "When did the Simpsons first air?" → as shorts on Tracey Ullman Show (1987) vs. as a half-hour show (1989)

This is different from **Epistemik ambiguity**, which is knowledge gaps that can be resolved with more training data.

## Downloaded Datasets

### ✅ 1. WiC (Word-in-Context)
- **Location**: `01_wic/`
- **Size**: 732 KB
- **Examples**: 6,066
- **Format**: Tab-separated text files
- **Description**: Dataset for word sense disambiguation. Each example contains a target word used in two different sentences, with a label indicating if the word has the same meaning in both contexts.
- **Splits**: train (5,428), dev (638)
- **Example**:
  - Target: "carry"
  - Sentence 1: "You must carry your camping gear"
  - Sentence 2: "Sound carries well over water"
  - Same meaning: False

### ✅ 2. AmbigQA / AmbigNQ
- **Location**: `02_ambigqa/`
- **Size**: 18 MB
- **Examples**: 12,038
- **Format**: JSON
- **Description**: Questions from Natural Questions that are ambiguous and require multiple QA pairs for complete answers. Each ambiguous question includes disambiguated versions with specific answers.
- **Splits**: train_light (10,036), dev_light (2,002)
- **Example**:
  - Ambiguous: "When did the Simpsons first air?"
  - Disambiguation 1: "When did the Simpsons first air as shorts on Tracey Ullman Show?" → "April 19, 1987"
  - Disambiguation 2: "When did the Simpsons first air as a half-hour show?" → "December 17, 1989"

### ✅ 3. ASQA (Long-Form Answers for Ambiguous Questions)
- **Location**: `03_asqa/dataset/`
- **Size**: 14 MB
- **Examples**: 5,301
- **Format**: JSON
- **Description**: Long-form question answering dataset focusing on ambiguous factoid questions. Each question has multiple QA pairs and long-form answers.
- **Splits**: train (4,353), dev (948)
- **Example**:
  - Ambiguous: "When does the new bunk'd come out?"
  - QA Pair 1: "When does episode 42 of bunk'd come out?" → "May 24, 2017"
  - QA Pair 2: "When does episode 41 of bunk'd come out?" → "April 21, 2017"
  - Also includes long-form answers synthesizing all interpretations

### ✅ 4. CLAMBER
- **Location**: `04_clamber/`
- **Size**: 2.5 MB
- **Examples**: 3,202
- **Format**: JSONL (double-encoded JSON)
- **Description**: Benchmark for clarifying ambiguous questions across 8 ambiguity types.
- **Categories**:
  - FD (False Detection): 800 examples
  - LA (Lack of Answer): 800 examples
  - MC (Missing Context): 1,602 examples
- **Subclasses**: ICL, NK, co-reference, polysemy, what, when, where, whom
- **Example**:
  - Question: "Give me a list of good coffee shops?"
  - Clarifying: "What do you personally consider important in a coffee shop?"
  - Requires clarification: Yes

### ✅ 5. CondAmbigQA-2K
- **Location**: `05_condambigqa/`
- **Size**: 32 MB
- **Examples**: 2,000
- **Format**: JSON
- **Description**: Conditionally ambiguous questions with properties and contexts. Each question has multiple interpretations based on different conditions.
- **Source**: HuggingFace (requires login)
- **Splits**: train (2,000)
- **Features**:
  - Average 1.91 properties per question
  - Average 20 contexts per question
  - Rich citation information
- **Example**:
  - Question: "How long is a rainbow six siege game?"
  - Multiple properties with citations explaining different game modes and durations

## Dataset Statistics

| Dataset | Examples | Size | Format | Status |
|---------|----------|------|--------|--------|
| WiC | 6,066 | 732 KB | TXT | ✅ |
| AmbigQA | 12,038 | 18 MB | JSON | ✅ |
| ASQA | 5,301 | 14 MB | JSON | ✅ |
| CLAMBER | 3,202 | 2.5 MB | JSONL | ✅ |
| CondAmbigQA-2K | 2,000 | 32 MB | JSON | ✅ |
| **Total** | **28,607** | **~67 MB** | - | **5/5 ready** ✅ |

## Downloading Datasets

### CondAmbigQA-2K (Requires HuggingFace Login)

CondAmbigQA-2K requires a HuggingFace account. To download:

```bash
# 1. Login to HuggingFace (if not already logged in)
huggingface-cli login
# or: hf auth login

# 2. Download the dataset
python download_condambigqa.py
```

The script will automatically download and save the dataset to `data/ambiguity_datasets/05_condambigqa/`.

## Testing the Datasets

Run the verification script to test all 5 datasets:

```bash
python test_datasets.py
```

This script:
- ✅ Loads and validates each dataset (WiC, AmbigQA, ASQA, CLAMBER, CondAmbigQA-2K)
- ✅ Shows sample examples from each
- ✅ Reports statistics and distributions
- ✅ Checks data integrity
- ✅ Provides total count: **28,607 examples**

## Using with RAG System

See the demonstration script for detailed examples:

```bash
python demo_ambiguity_rag.py
```

### Quick Start

1. **Convert dataset to documents**:
   ```python
   from demo_ambiguity_rag import load_ambigqa_sample, convert_ambigqa_to_documents

   examples = load_ambigqa_sample('data/ambiguity_datasets/02_ambigqa/train_light.json')
   documents = convert_ambigqa_to_documents(examples)
   ```

2. **Save as JSON**:
   ```python
   import json
   with open('ambigqa_docs.json', 'w') as f:
       json.dump(documents, f, indent=2)
   ```

3. **Index into RAG**:
   ```python
   from src.rag.rag_pipeline import RAGPipeline

   pipeline = RAGPipeline('config/base_config.yaml')
   pipeline.index_documents('ambigqa_docs.json')
   ```

4. **Query**:
   ```python
   result = pipeline.query("When did the Simpsons first air?")
   print(result['answer'])
   ```

## Benefits for RAG Systems

Using ambiguity datasets provides several advantages:

1. **Improved Robustness**: System learns to handle questions with multiple valid interpretations
2. **Better Disambiguation**: RAG retrieves multiple relevant contexts for ambiguous queries
3. **Comprehensive Answers**: LLM can generate answers covering all interpretations
4. **Realistic Training**: Real-world questions are often ambiguous
5. **Evaluation Benchmark**: Test RAG system's ability to handle ambiguity

## Dataset Sources

- **WiC**: https://pilehvar.github.io/wic/
- **AmbigQA**: https://nlp.cs.washington.edu/ambigqa
- **ASQA**: https://github.com/google-research/language/tree/master/language/asqa
- **CLAMBER**: https://github.com/yuchenlin/clamber
- **CondAmbigQA-2K**: https://huggingface.co/datasets/Apocalypse-AGI-DAO/CondAmbigQA-2K

## Citation

If you use these datasets, please cite the original papers:

### WiC
```bibtex
@inproceedings{pilehvar-camacho-collados-2019-wic,
    title = "{WiC}: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations",
    author = "Pilehvar, Mohammad Taher and Camacho-Collados, Jose",
    booktitle = "NAACL-HLT",
    year = "2019"
}
```

### AmbigQA
```bibtex
@inproceedings{min2020ambigqa,
    title = {AmbigQA: Answering Ambiguous Open-domain Questions},
    author = {Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke},
    booktitle = {EMNLP},
    year = {2020}
}
```

### ASQA
```bibtex
@article{stelmakh2022asqa,
    title = {ASQA: Factoid Questions Meet Long-Form Answers},
    author = {Stelmakh, Ivan and Luan, Yi and Dhingra, Bhuwan and Chang, Ming-Wei},
    journal = {arXiv preprint arXiv:2204.06092},
    year = {2022}
}
```

### CLAMBER
```bibtex
@inproceedings{xu2023clamber,
    title = {CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models},
    author = {Xu, Tong and Zhang, Qingqing and Lin, Yuchenlin and others},
    year = {2023}
}
```
