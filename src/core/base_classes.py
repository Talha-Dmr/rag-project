"""
Base abstract classes for all swappable components in the RAG system.

This module defines the interfaces that all implementations must follow,
enabling modularity and easy component swapping.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


# Configuration Models
class LoaderConfig(BaseModel):
    """Configuration for data loaders"""
    batch_size: int = 10
    recursive: bool = False


class ChunkerConfig(BaseModel):
    """Configuration for chunking strategies"""
    chunk_size: int = 512
    chunk_overlap: int = 50


class EmbedderConfig(BaseModel):
    """Configuration for embedding models"""
    model_name: str
    device: str = "cpu"
    cache_folder: Optional[str] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector stores"""
    persist_directory: Optional[str] = None
    collection_name: str = "documents"


class RerankerConfig(BaseModel):
    """Configuration for reranking"""
    model_name: Optional[str] = None
    top_k: int = 5


# Abstract Base Classes

class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def load(self, source: str) -> List[Dict[str, Any]]:
        """
        Load data from source

        Args:
            source: Path or URL to data source

        Returns:
            List of documents with metadata
        """
        pass

    @abstractmethod
    def load_batch(self, sources: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple sources in batch

        Args:
            sources: List of paths or URLs

        Returns:
            List of documents with metadata
        """
        pass


class BaseChunker(ABC):
    """Abstract base class for chunking strategies"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        pass

    @abstractmethod
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents

        Args:
            documents: List of document dicts with 'content' field

        Returns:
            List of chunked documents with metadata
        """
        pass


class BaseEmbedder(ABC):
    """Abstract base class for embedding models"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get embedding dimension

        Returns:
            Dimension of embedding vectors
        """
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to vector store

        Args:
            texts: Document texts
            embeddings: Embedding vectors
            metadatas: Optional metadata for each document
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of documents with scores and metadata
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Get number of documents in store"""
        pass


class BaseReranker(ABC):
    """Abstract base class for reranking strategies"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance

        Args:
            query: Query text
            documents: List of documents with 'content' and 'score' fields
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents with updated scores
        """
        pass


class BaseLLM(ABC):
    """Abstract base class for LLM"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text with retrieved context

        Args:
            query: User query
            context: Retrieved context documents
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        pass
