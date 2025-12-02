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

def test_empty_documents():
    """Empty list test"""
    print("\n=== Test 2: Empty List Test ===")
    reranker = RerankerFactory.create("mgte", CONFIG)
    result = reranker.rerank("query", [])
    assert result == []
    print("✅ Empty list test passed.")

def test_missing_content():
    """Test for documents with missing/empty content"""
    print("\n=== Test 3: Missing Content Test ===")
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
        test_empty_documents()
        test_missing_content()
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        