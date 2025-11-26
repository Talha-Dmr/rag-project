"""
Step-by-Step RAG Example

This example shows how to manually construct and use each component:
- DataManager for loading
- Chunker for splitting
- Embedder for vectors
- VectorStore for storage
- Retriever for search
- Reranker for improving results
- LLM for generation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import DataManager
from src.chunking import ChunkerFactory
from src.embeddings import EmbedderFactory
from src.vector_stores import VectorStoreFactory
from src.rag import HuggingFaceLLM, Retriever
from src.reranking import RerankerFactory


def main():
    print("=" * 60)
    print("Step-by-Step RAG Components Example")
    print("=" * 60)

    # Step 1: Load Documents
    print("\n[Step 1] Loading Documents")
    data_manager = DataManager()

    # Create sample document
    sample_docs = [
        {
            'content': "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval with generation.",
            'metadata': {'source': 'example', 'id': 1}
        },
        {
            'content': "RAG improves language models by providing relevant context from a knowledge base.",
            'metadata': {'source': 'example', 'id': 2}
        }
    ]
    print(f"   Loaded {len(sample_docs)} sample documents")

    # Step 2: Chunking
    print("\n[Step 2] Chunking Documents")
    chunker = ChunkerFactory.create('fixed_size', {
        'chunk_size': 200,
        'chunk_overlap': 20
    })
    chunks = chunker.chunk_documents(sample_docs)
    print(f"   Created {len(chunks)} chunks")

    # Step 3: Embedding
    print("\n[Step 3] Generating Embeddings")
    embedder = EmbedderFactory.create('huggingface', {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',  # Smaller model for example
        'device': 'cpu'
    })
    texts = [chunk['content'] for chunk in chunks]
    embeddings = embedder.embed_batch(texts)
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {embedder.get_dimension()}")

    # Step 4: Store in Vector DB
    print("\n[Step 4] Storing in Vector Database")
    vector_store = VectorStoreFactory.create('chroma', {
        'persist_directory': './data/example_db',
        'collection_name': 'example'
    })
    metadatas = [chunk['metadata'] for chunk in chunks]
    vector_store.add_documents(texts, embeddings, metadatas)
    print(f"   Stored {vector_store.get_count()} documents")

    # Step 5: Retrieval
    print("\n[Step 5] Retrieving Relevant Documents")
    retriever = Retriever(embedder, vector_store, k=5)
    query = "What is RAG?"
    retrieved = retriever.retrieve(query)
    print(f"   Query: {query}")
    print(f"   Retrieved {len(retrieved)} documents")
    for i, doc in enumerate(retrieved[:2]):
        print(f"   Doc {i+1} (score: {doc['score']:.3f}): {doc['content'][:80]}...")

    # Step 6: Reranking
    print("\n[Step 6] Reranking Documents")
    reranker = RerankerFactory.create('bm25')  # Using BM25 for speed
    reranked = reranker.rerank(query, retrieved, top_k=3)
    print(f"   Reranked to top {len(reranked)} documents")
    for i, doc in enumerate(reranked):
        print(f"   Doc {i+1} (rerank score: {doc['rerank_score']:.3f})")

    # Step 7: Generation (Note: This requires downloading LLM)
    print("\n[Step 7] Generating Answer")
    print("   Note: LLM generation requires downloading models")
    print("   Skipping in this example...")
    # Uncomment to use LLM:
    # llm = HuggingFaceLLM({
    #     'model_name': 'microsoft/phi-2',
    #     'device': 'cpu',
    #     'max_new_tokens': 100
    # })
    # context = [doc['content'] for doc in reranked]
    # answer = llm.generate_with_context(query, context)
    # print(f"   Answer: {answer}")

    print("\n" + "=" * 60)
    print("Step-by-step example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
