import sys
import os
import pytest

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from src.reranking.base_reranker import RerankerFactory
import src.reranking.rerankers
from src.core.logger import get_logger

logger = get_logger(__name__)

# Configuration
CONFIG = {
    "model_name_or_path": "Alibaba-NLP/gte-multilingual-reranker-base",
    "device": "cpu",
    "batch_size": 2
}

def test_mgte_integration():
    """Standard integration test"""
    print("\n=== Test 1: Standard Integration ===")
    reranker = RerankerFactory.create("mgte", CONFIG)
    
    query = "What is the capital of Turkey?"
    documents = [
        {"content": "Paris is the capital of France.", "metadata": {"id": 1}},
        {"content": "Ankara is the capital of Turkey.", "metadata": {"id": 2}},
        {"content": "Istanbul is the largest city in Turkey.", "metadata": {"id": 3}}
    ]
    
    results = reranker.rerank(query, documents, top_k=2)
    
    assert len(results) == 2
    assert "Ankara" in results[0]['content']
    print("✅ Standard test passed.")

def test_batch_processing():
    """Test batch processing with more documents than batch_size"""
    print("\n=== Test 2: Batch Processing ===")
    # batch_size is 2, we send 5 documents
    reranker = RerankerFactory.create("mgte", {"batch_size": 2, "device": "cpu"})
    
    docs = [{"content": f"Document content {i}", "metadata": {"id": i}} for i in range(5)]
    query = "test query"
    
    results = reranker.rerank(query, docs)
    
    assert len(results) == 5
    print("✅ Batch processing test passed.")

def test_long_context():
    """Test with long context inputs (near 8192 tokens)"""
    print("\n=== Test 3: Long Context Capability ===")
    reranker = RerankerFactory.create("mgte", CONFIG)
    
    # Create a very long document (approx 6000 words, likely > 7000 tokens)
    long_text = "word " * 6000
    docs = [{"content": long_text, "metadata": {"id": "long_doc"}}]
    
    # Should not crash
    results = reranker.rerank("query", docs)
    
    assert len(results) == 1
    assert results[0]['score'] is not None
    print("✅ Long context test passed.")

def test_multilingual_capabilities():
    """Test English query against Turkish document"""
    print("\n=== Test 4: Multilingual Capabilities ===")
    reranker = RerankerFactory.create("mgte", CONFIG)
    
    query = "What causes inflation?"
    docs = [
        {"content": "Enflasyon, dolaşımdaki para arzının artması sonucu oluşur.", "metadata": {"lang": "tr"}},
        {"content": "Futbol maçı 2-1 bitti.", "metadata": {"lang": "tr"}}
    ]
    
    results = reranker.rerank(query, docs)
    
    # The definition of inflation (TR) should be ranked higher than football (TR)
    assert "Enflasyon" in results[0]['content']
    print("✅ Multilingual test passed.")

def test_empty_documents():
    """Empty list test"""
    print("\n=== Test 5: Empty List Test ===")
    reranker = RerankerFactory.create("mgte", CONFIG)
    result = reranker.rerank("query", [])
    assert result == []
    print("✅ Empty list test passed.")

def test_missing_content():
    """Test for documents with missing/empty content"""
    print("\n=== Test 6: Missing Content Test ===")
    reranker = RerankerFactory.create("mgte", CONFIG)
    
    docs = [
        {"metadata": {"id": 1}},           # missing content key
        {"content": "", "metadata": {"id": 2}}, # empty content string
        {"content": "   ", "metadata": {"id": 3}}, # whitespace only content
        {"content": "Valid doc", "metadata": {"id": 4}} # Valid
    ]
    
    result = reranker.rerank("query", docs)
    
    # Should return only 1 valid document
    assert len(result) == 1
    assert result[0]["content"] == "Valid doc"
    print("✅ Missing content test passed.")

if __name__ == "__main__":
    try:
        test_mgte_integration()
        test_batch_processing()
        test_long_context()
        test_multilingual_capabilities()
        test_empty_documents()
        test_missing_content()
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")