# Reranking Implementation Documentation

## Overview

Reranking improves retrieval quality by re-scoring and reordering documents after initial retrieval. This two-stage approach balances efficiency (fast initial retrieval) with accuracy (precise reranking).

## Why Reranking?

Initial retrieval uses bi-encoders (separate query and document embeddings):
- **Pros**: Fast, efficient for large collections
- **Cons**: Limited semantic understanding of query-document interaction

Reranking uses more sophisticated models:
- **Pros**: Better accuracy, understands query-document relationships
- **Cons**: Slower, only practical for small candidate sets

**Best Practice**: Retrieve top-N (e.g., 50-100) with bi-encoder, then rerank to top-K (e.g., 5-10)

## Architecture

### Two-Stage Retrieval

```
Query → Embedder → Vector Store → Top-N Documents
                                        ↓
                                    Reranker
                                        ↓
                                   Top-K Documents
                                        ↓
                                       LLM
```

## Available Rerankers

### 1. Cross-Encoder Reranker (Primary)

**Best for**: High-quality reranking, question-answering

```python
from src.reranking import RerankerFactory

# Create cross-encoder reranker
reranker = RerankerFactory.create('cross_encoder', {
    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'top_k': 5,
    'device': 'cpu'
})

# Rerank documents
reranked_docs = reranker.rerank(
    query="What is machine learning?",
    documents=retrieved_docs,
    top_k=5
)

for doc in reranked_docs:
    print(f"Score: {doc['rerank_score']:.3f}")
    print(f"Content: {doc['content'][:100]}...")
```

**How It Works**:
1. Encodes query + document together
2. Produces relevance score directly
3. More accurate than bi-encoder similarity

**Recommended Models**:
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: Fast, good quality
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: Better quality, slower
- `cross-encoder/ms-marco-electra-base`: High quality

**Parameters**:
- `model_name`: HuggingFace cross-encoder model
- `top_k`: Number of documents to return (default: 5)
- `device`: 'cpu' or 'cuda'

**Configuration**:
```yaml
reranker:
  type: cross_encoder
  config:
    model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
    top_k: 5
    device: cpu
```

### 2. BM25 Reranker

**Best for**: Lexical search, keyword matching

```python
# Create BM25 reranker
reranker = RerankerFactory.create('bm25', {
    'top_k': 5
})

# Rerank documents
reranked_docs = reranker.rerank(
    query="machine learning algorithms",
    documents=retrieved_docs
)
```

**How It Works**:
1. Tokenizes query and documents
2. Scores based on term frequency and document length
3. Pure lexical matching (no ML model)

**Pros**:
- Very fast (no model loading)
- Good for keyword queries
- No GPU needed
- Works well with acronyms/rare terms

**Cons**:
- No semantic understanding
- Sensitive to exact word matching

**Configuration**:
```yaml
reranker:
  type: bm25
  config:
    top_k: 5
```

## Usage Patterns

### Basic Reranking

```python
from src.reranking import RerankerFactory

# Create reranker
reranker = RerankerFactory.create('cross_encoder')

# Get initial retrieval results
retrieved_docs = retriever.retrieve("What is RAG?", k=20)

# Rerank to top 5
reranked_docs = reranker.rerank(
    query="What is RAG?",
    documents=retrieved_docs,
    top_k=5
)
```

### Integrated with RAG Pipeline

```python
from src.rag import RAGPipeline, Retriever
from src.reranking import RerankerFactory

# Create components
retriever = Retriever(embedder, vector_store, k=20)
reranker = RerankerFactory.create('cross_encoder')

# Query with reranking
query = "Explain neural networks"

# 1. Initial retrieval (top 20)
docs = retriever.retrieve(query, k=20)

# 2. Rerank to top 5
reranked = reranker.rerank(query, docs, top_k=5)

# 3. Generate with reranked context
answer = llm.generate_with_context(
    query=query,
    context=[d['content'] for d in reranked]
)
```

### Comparing Rerankers

```python
from src.reranking import RerankerFactory

query = "What is deep learning?"
docs = retriever.retrieve(query, k=20)

# Cross-encoder reranking
ce_reranker = RerankerFactory.create('cross_encoder')
ce_results = ce_reranker.rerank(query, docs, top_k=5)

# BM25 reranking
bm25_reranker = RerankerFactory.create('bm25')
bm25_results = bm25_reranker.rerank(query, docs, top_k=5)

# Compare
print("Cross-Encoder Top Doc:")
print(ce_results[0]['content'][:200])

print("\nBM25 Top Doc:")
print(bm25_results[0]['content'][:200])
```

## Score Interpretation

### Cross-Encoder Scores

```python
reranked = reranker.rerank(query, docs)

for doc in reranked:
    print(f"Original score: {doc['original_score']:.3f}")
    print(f"Rerank score: {doc['rerank_score']:.3f}")
```

- **original_score**: Bi-encoder cosine similarity (0-1)
- **rerank_score**: Cross-encoder relevance score (unbounded, often -10 to +10)
- Higher rerank_score = more relevant

### BM25 Scores

- BM25 scores are unbounded positive numbers
- Higher score = better term matching
- Relative ranking matters more than absolute scores

## Performance Optimization

### 1. Tune Retrieval-Reranking Balance

```python
# Strategy A: Broad retrieval + aggressive reranking
docs = retriever.retrieve(query, k=50)  # Cast wide net
reranked = reranker.rerank(query, docs, top_k=3)  # Narrow down

# Strategy B: Focused retrieval + light reranking
docs = retriever.retrieve(query, k=10)  # Already good candidates
reranked = reranker.rerank(query, docs, top_k=5)  # Minor adjustments
```

### 2. Use GPU for Cross-Encoder

```python
reranker = RerankerFactory.create('cross_encoder', {
    'device': 'cuda',  # Much faster on GPU
    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
})
```

### 3. Choose Model Size

```python
# Fast (good for real-time)
reranker = RerankerFactory.create('cross_encoder', {
    'model_name': 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
})

# Balanced
reranker = RerankerFactory.create('cross_encoder', {
    'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
})

# High quality (slower)
reranker = RerankerFactory.create('cross_encoder', {
    'model_name': 'cross-encoder/ms-marco-electra-base'
})
```

## Integration Patterns

### Pattern 1: Reranking in RAGPipeline

Modify RAG pipeline to include reranking:

```python
class RAGPipelineWithReranking(RAGPipeline):
    def __init__(self, *args, reranker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = reranker

    def query(self, query_text, k=5, return_context=False):
        # Retrieve more documents initially
        docs = self.retriever.retrieve(query_text, k=k*4)

        # Rerank to final k
        if self.reranker:
            docs = self.reranker.rerank(query_text, docs, top_k=k)

        # Generate answer
        context_texts = [d['content'] for d in docs]
        answer = self.llm.generate_with_context(query_text, context_texts)

        return {'answer': answer, 'num_docs_retrieved': len(docs)}
```

### Pattern 2: Ensemble Reranking

Combine multiple reranking strategies:

```python
def ensemble_rerank(query, docs, top_k=5):
    # Get scores from both rerankers
    ce_reranker = RerankerFactory.create('cross_encoder')
    bm25_reranker = RerankerFactory.create('bm25')

    ce_docs = ce_reranker.rerank(query, docs, top_k=len(docs))
    bm25_docs = bm25_reranker.rerank(query, docs, top_k=len(docs))

    # Combine scores (weighted average)
    for i, doc in enumerate(docs):
        ce_score = ce_docs[i]['rerank_score']
        bm25_score = bm25_docs[i]['rerank_score']

        # Normalize and combine
        doc['ensemble_score'] = 0.7 * ce_score + 0.3 * bm25_score

    # Sort and return top_k
    docs.sort(key=lambda x: x['ensemble_score'], reverse=True)
    return docs[:top_k]
```

## Best Practices

### 1. Choose the Right Reranker

**Use Cross-Encoder when**:
- Quality is critical
- You have GPU available
- Working with natural language queries
- Need semantic understanding

**Use BM25 when**:
- Speed is critical
- Working with keyword queries
- Looking for exact term matches
- No GPU available

### 2. Optimize Retrieval Parameters

```python
# Good: Retrieve enough candidates for reranking
docs = retriever.retrieve(query, k=50)
reranked = reranker.rerank(query, docs, top_k=5)

# Bad: Too few candidates limits reranking benefit
docs = retriever.retrieve(query, k=5)
reranked = reranker.rerank(query, docs, top_k=5)  # No benefit
```

### 3. Monitor Score Distributions

```python
reranked = reranker.rerank(query, docs)

# Check score spread
scores = [d['rerank_score'] for d in reranked]
print(f"Score range: {min(scores):.2f} to {max(scores):.2f}")
print(f"Top score: {scores[0]:.2f}, 5th score: {scores[4]:.2f}")

# Large gap = confident results
# Small gap = ambiguous results
```

## Creating Custom Rerankers

```python
from src.core.base_classes import BaseReranker
from src.reranking.base_reranker import register_reranker

@register_reranker("custom")
class CustomReranker(BaseReranker):
    def __init__(self, config=None):
        super().__init__(config)
        # Initialize your model

    def rerank(self, query, documents, top_k=None):
        top_k = top_k or self.config.get('top_k', 5)

        # Your reranking logic
        for doc in documents:
            doc['rerank_score'] = your_scoring_function(query, doc)

        # Sort and return
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
```

## Troubleshooting

### Issue: Reranking is slow
**Solutions**:
- Use GPU: `device: 'cuda'`
- Use smaller model: TinyBERT variant
- Reduce initial retrieval k
- Use BM25 instead

### Issue: Reranking doesn't improve results
**Solutions**:
- Increase initial retrieval k (retrieve more candidates)
- Try different reranker (cross-encoder vs BM25)
- Check if query/documents match reranker training data
- Verify retrieval is working correctly first

### Issue: Out of memory during reranking
**Solutions**:
- Reduce initial retrieval k
- Process documents in batches
- Use smaller cross-encoder model
- Use CPU instead of GPU

## API Reference

### RerankerFactory

```python
RerankerFactory.register(name: str, reranker_class: Type)
RerankerFactory.create(reranker_type: str, config: Dict) -> BaseReranker
RerankerFactory.get_available_rerankers() -> List[str]
```

### BaseReranker

```python
rerank(query: str, documents: List[Dict], top_k: int) -> List[Dict]
```

### CrossEncoderReranker

```python
CrossEncoderReranker(config: Dict)
# config: model_name, top_k, device
```

### BM25Reranker

```python
BM25Reranker(config: Dict)
# config: top_k
```

## Examples

See `examples/reranking_examples.py` for complete working examples.
