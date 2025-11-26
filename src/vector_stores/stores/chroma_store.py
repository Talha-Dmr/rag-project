"""
ChromaDB vector store implementation.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.core.base_classes import BaseVectorStore
from src.vector_stores.base_store import register_store
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_store("chroma")
class ChromaStore(BaseVectorStore):
    """
    Vector store using ChromaDB

    Provides persistent storage with similarity search
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.persist_directory = self.config.get('persist_directory', './data/vector_db/chroma')
        self.collection_name = self.config.get('collection_name', 'documents')

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        try:
            # Initialize Chroma client with persistence
            self.client = chromadb.Client(Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            ))

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

            logger.info(
                f"ChromaDB initialized: collection='{self.collection_name}', "
                f"documents={self.collection.count()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to ChromaDB

        Args:
            texts: Document texts
            embeddings: Embedding vectors
            metadatas: Optional metadata for each document
        """
        if not texts or not embeddings:
            logger.warning("No documents to add")
            return

        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match")

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")

        try:
            # Generate IDs
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(texts))]

            # Prepare metadatas
            if metadatas is None:
                metadatas = [{}] * len(texts)

            # Convert metadata values to strings (Chroma requirement)
            clean_metadatas = []
            for metadata in metadatas:
                clean_meta = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        clean_meta[key] = str(value) if not isinstance(value, (int, float, bool)) else value
                clean_metadatas.append(clean_meta)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=clean_metadatas
            )

            logger.info(f"Added {len(texts)} documents to ChromaDB")

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

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
        try:
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'id': results['ids'][0][i] if 'ids' in results else None
                    }
                    documents.append(doc)

            logger.debug(f"Found {len(documents)} similar documents")

            return documents

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")

            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def get_count(self) -> int:
        """Get number of documents in store"""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

    def persist(self) -> None:
        """
        Persist the collection to disk

        Note: ChromaDB with Settings persist_directory automatically persists
        """
        logger.info("ChromaDB collection persisted")
