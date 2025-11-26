"""
Main RAG Pipeline orchestrator.
"""

from typing import List, Dict, Any, Optional
from src.dataset.data_manager import DataManager
from src.chunking.base_chunker import ChunkerFactory
from src.embeddings.base_embedder import EmbedderFactory
from src.vector_stores.base_store import VectorStoreFactory
from src.rag.llm_wrapper import HuggingFaceLLM
from src.rag.retriever import Retriever
from src.core.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline

    Combines document loading, chunking, embedding, storage, retrieval, and generation
    """

    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        chunker = None,
        embedder = None,
        vector_store = None,
        llm = None,
        retriever = None
    ):
        """
        Initialize RAG Pipeline

        Args:
            data_manager: DataManager for loading documents
            chunker: Chunker for splitting documents
            embedder: Embedder for generating embeddings
            vector_store: Vector store for similarity search
            llm: LLM for generation
            retriever: Retriever for fetching relevant docs
        """
        self.data_manager = data_manager
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = retriever

        logger.info("RAG Pipeline initialized")

    def index_documents(
        self,
        source: str,
        recursive: bool = False,
        file_extensions: Optional[List[str]] = None
    ) -> int:
        """
        Index documents from a source

        Args:
            source: Path to file or directory
            recursive: Search directories recursively
            file_extensions: Filter by file extensions

        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing documents from: {source}")

        # Load documents
        logger.info("Loading documents...")
        documents = self.data_manager.load_from_path(
            source,
            recursive=recursive,
            file_extensions=file_extensions
        )
        logger.info(f"Loaded {len(documents)} documents")

        # Chunk documents
        logger.info("Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Embed chunks
        logger.info("Embedding chunks...")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Store in vector database
        logger.info("Storing in vector database...")
        metadatas = [chunk['metadata'] for chunk in chunks]
        self.vector_store.add_documents(texts, embeddings, metadatas)

        logger.info(f"Successfully indexed {len(chunks)} chunks")

        return len(chunks)

    def query(
        self,
        query_text: str,
        k: int = 5,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            query_text: User query
            k: Number of documents to retrieve
            return_context: Include retrieved context in response

        Returns:
            Dictionary with answer and optional context
        """
        logger.info(f"Processing query: {query_text[:100]}...")

        # Retrieve relevant documents
        logger.info(f"Retrieving top {k} documents...")
        retrieved_docs = self.retriever.retrieve(query_text, k=k)

        if not retrieved_docs:
            logger.warning("No documents retrieved")
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'context': [] if return_context else None,
                'num_docs_retrieved': 0
            }

        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Generate answer
        logger.info("Generating answer...")
        context_texts = [doc['content'] for doc in retrieved_docs]
        answer = self.llm.generate_with_context(query_text, context_texts)

        logger.info("Query complete")

        result = {
            'answer': answer,
            'num_docs_retrieved': len(retrieved_docs)
        }

        if return_context:
            result['context'] = retrieved_docs

        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RAGPipeline':
        """
        Create RAG Pipeline from configuration

        Args:
            config: Configuration dictionary

        Returns:
            RAGPipeline instance
        """
        logger.info("Creating RAG Pipeline from config...")

        # Data Manager
        data_manager = DataManager(config.get('data_loader', {}))

        # Chunker
        chunking_config = config.get('chunking', {})
        chunker = ChunkerFactory.create(
            chunking_config.get('strategy', 'fixed_size'),
            chunking_config.get('config', {})
        )

        # Embedder
        embedder_config = config.get('embeddings', {})
        embedder = EmbedderFactory.create('huggingface', embedder_config)

        # Vector Store
        vector_store_config = config.get('vector_store', {})
        vector_store = VectorStoreFactory.create(
            vector_store_config.get('type', 'chroma'),
            vector_store_config.get('config', {})
        )

        # LLM
        llm_config = config.get('llm', {})
        llm = HuggingFaceLLM(llm_config)

        # Retriever
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            k=config.get('retrieval', {}).get('k', 10),
            score_threshold=config.get('retrieval', {}).get('score_threshold', 0.0)
        )

        return cls(
            data_manager=data_manager,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
            retriever=retriever
        )
