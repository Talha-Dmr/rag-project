"""
Basic RAG Pipeline Example

This example demonstrates the complete RAG workflow:
1. Loading documents
2. Chunking
3. Embedding and indexing
4. Querying with retrieval
5. Generating answers
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_loader import load_config
from src.rag import RAGPipeline


def main():
    print("=" * 60)
    print("Modular RAG System - Basic Example")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config('base_config')
    print("   ✓ Configuration loaded")

    # Create RAG pipeline
    print("\n2. Initializing RAG pipeline...")
    print("   - Loading embedding model...")
    print("   - Loading LLM...")
    print("   - Initializing vector store...")
    rag = RAGPipeline.from_config(config)
    print("   ✓ RAG pipeline initialized")

    # Index documents (if you have documents to index)
    print("\n3. Document indexing...")
    print("   Note: Place your documents in a 'data/' directory")
    print("   Supported formats: PDF, TXT, JSON")

    # Example: Uncomment to index documents
    # try:
    #     num_chunks = rag.index_documents(
    #         source='data/',
    #         recursive=True,
    #         file_extensions=['.pdf', '.txt']
    #     )
    #     print(f"   ✓ Indexed {num_chunks} document chunks")
    # except FileNotFoundError:
    #     print("   ⚠ No documents found in 'data/' directory")

    # Query the system
    print("\n4. Querying the RAG system...")
    query = "What is retrieval-augmented generation?"
    print(f"   Query: {query}")

    try:
        result = rag.query(
            query_text=query,
            k=5,
            return_context=True
        )

        print(f"\n   Answer:")
        print(f"   {result['answer']}")
        print(f"\n   Retrieved {result['num_docs_retrieved']} documents")

        if 'context' in result and result['context']:
            print(f"\n   Top retrieved document:")
            top_doc = result['context'][0]
            print(f"   Score: {top_doc['score']:.3f}")
            print(f"   Content: {top_doc['content'][:200]}...")

    except Exception as e:
        print(f"   ⚠ Error querying: {e}")
        print("   Make sure you have indexed documents first!")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
