"""
Main RAG Pipeline orchestrator.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from src.dataset.data_manager import DataManager
from src.chunking.base_chunker import ChunkerFactory
from src.embeddings.base_embedder import EmbedderFactory
from src.reranking.base_reranker import RerankerFactory
from src.vector_stores.base_store import VectorStoreFactory
from src.rag.llm_wrapper import HuggingFaceLLM
from src.rag.retriever import Retriever
from src.core.base_classes import BaseReranker
from src.core.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline

    Combines document loading, chunking, embedding, storage, retrieval, and generation
    """

    def __init__(
        self,
        data_manager: DataManager,
        chunker,
        embedder,
        vector_store,
        llm,
        retriever,
        reranker=None,
        reranker_top_k: Optional[int] = None,
        hallucination_detector=None,
        gating_config: Optional[Dict[str, Any]] = None
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
            hallucination_detector: Optional hallucination detector
        """
        self.data_manager = data_manager
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.reranker_top_k = reranker_top_k
        self.hallucination_detector = hallucination_detector
        self.gating_config = gating_config or {}

        if hallucination_detector:
            logger.info("RAG Pipeline initialized with hallucination detection")
        else:
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

    def _compute_uncertainty_stats(
        self,
        detection_result: Dict[str, Any],
        uncertainty_source: Optional[str] = None
    ) -> Dict[str, float]:
        individual = detection_result.get("individual_results", [])
        if not individual:
            return {
                "contradiction_rate": 0.0,
                "contradiction_prob_mean": 0.0,
                "uncertainty_mean": 0.0
            }

        contradiction_probs = []
        uncertainties = []
        for result in individual:
            scores = result.get("scores")
            if scores:
                contradiction_probs.append(scores.get("contradiction", 0.0))
                if uncertainty_source == "entropy":
                    uncertainties.append(result.get("uncertainty_entropy", 0.0))
                elif uncertainty_source == "variance":
                    uncertainties.append(result.get("uncertainty_variance", 0.0))
                elif uncertainty_source == "contradiction_variance":
                    uncertainties.append(result.get("contradiction_variance", 0.0))
                else:
                    max_prob = max(scores.values())
                    uncertainties.append(1.0 - max_prob)
            else:
                contradiction_probs.append(1.0 if result.get("is_hallucination") else 0.0)
                uncertainties.append(1.0 - result.get("confidence", 0.0))

        contradiction_prob_mean = sum(contradiction_probs) / len(contradiction_probs)
        uncertainty_mean = sum(uncertainties) / len(uncertainties)

        return {
            "contradiction_rate": detection_result.get("hallucination_score", 0.0),
            "contradiction_prob_mean": contradiction_prob_mean,
            "uncertainty_mean": uncertainty_mean
        }

    def _compute_retrieval_stats(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, float]:
        if not retrieved_docs:
            return {
                "max_score": 0.0,
                "mean_score": 0.0
            }

        scores = [doc.get("score", 0.0) for doc in retrieved_docs]
        scores = [float(s) for s in scores if isinstance(s, (int, float))]
        if not scores:
            return {
                "max_score": 0.0,
                "mean_score": 0.0
            }

        return {
            "max_score": max(scores),
            "mean_score": sum(scores) / len(scores)
        }

    def _merge_gating_config(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not override:
            return dict(self.gating_config)

        merged = dict(self.gating_config)
        merged.update(override)
        return merged

    def _decide_gating_action(self, stats: Dict[str, float], gating_config: Dict[str, Any]) -> str:
        contradiction_rate = stats.get("contradiction_rate", 0.0)
        contradiction_prob = stats.get("contradiction_prob_mean", 0.0)
        uncertainty = stats.get("uncertainty_mean", 0.0)
        max_retrieval_score = stats.get("retrieval_max_score", 1.0)
        mean_retrieval_score = stats.get("retrieval_mean_score", 1.0)

        contradiction_rate_threshold = gating_config.get("contradiction_rate_threshold", 0.34)
        contradiction_prob_threshold = gating_config.get("contradiction_prob_threshold", 0.5)
        uncertainty_threshold = gating_config.get("uncertainty_threshold", 0.3)
        min_retrieval_score = gating_config.get("min_retrieval_score")
        min_mean_retrieval_score = gating_config.get("min_mean_retrieval_score")

        retrieval_low = False
        if isinstance(min_retrieval_score, (int, float)) and max_retrieval_score < float(min_retrieval_score):
            retrieval_low = True
        if isinstance(min_mean_retrieval_score, (int, float)) and mean_retrieval_score < float(min_mean_retrieval_score):
            retrieval_low = True

        should_gate = (
            contradiction_rate >= contradiction_rate_threshold or
            contradiction_prob >= contradiction_prob_threshold or
            uncertainty >= uncertainty_threshold or
            retrieval_low
        )

        if not should_gate:
            return "none"

        return gating_config.get("strategy", "abstain")

    def query(
        self,
        query_text: str,
        k: Optional[int] = None,
        return_context: bool = False,
        detect_hallucinations: bool = True,
        hallucination_aggregation: str = 'any',
        gating: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            query_text: User query
            k: Number of documents to retrieve
            return_context: Include retrieved context in response
            detect_hallucinations: Run hallucination detection on answer
            hallucination_aggregation: How to aggregate hallucination detection
                                      across contexts ('any', 'majority', 'all')

        Returns:
            Dictionary with answer, optional context, and hallucination detection results
        """
        logger.info(f"Processing query: {query_text[:100]}...")

        gating_config = self._merge_gating_config(gating)
        gating_enabled = bool(gating_config.get("enabled", False)) and bool(self.hallucination_detector)
        gating_action = "none"
        gating_stats = None
        gating_attempts = 0

        current_k = k if k is not None else self.retriever.k
        max_retries = int(gating_config.get("max_retries", 1))
        k_multiplier = float(gating_config.get("k_multiplier", 2.0))
        max_k = gating_config.get("max_k")
        abstain_message = gating_config.get(
            "abstain_message",
            "I don't have enough reliable information to answer that."
        )

        while True:
            # Retrieve relevant documents
            logger.info(f"Retrieving top {current_k} documents...")
            retrieved_docs = self.retriever.retrieve(query_text, k=current_k)

            if self.reranker:
                logger.info("Applying reranker...")
                rerank_top_k = self.reranker_top_k or current_k
                retrieved_docs = self.reranker.rerank(query_text, retrieved_docs, top_k=rerank_top_k)
                logger.info(f"Reranked documents → top {len(retrieved_docs)} kept")

            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return {
                    'answer': "I couldn't find relevant information to answer your question.",
                    'context': [] if return_context else None,
                    'num_docs_retrieved': 0,
                    'hallucination_detected': False
                }

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            retrieval_stats = self._compute_retrieval_stats(retrieved_docs)

            # Generate answer
            logger.info("Generating answer...")
            context_texts = [doc['content'] for doc in retrieved_docs]
            answer = self.llm.generate_with_context(query_text, context_texts)

            result = {
                'answer': answer,
                'num_docs_retrieved': len(retrieved_docs)
            }

            # Hallucination detection
            if detect_hallucinations and self.hallucination_detector:
                logger.info("Checking for hallucinations...")

                try:
                    detection_result = self.hallucination_detector.verify_answer_with_contexts(
                        answer=answer,
                        contexts=context_texts,
                        aggregation=hallucination_aggregation
                    )

                    result['hallucination_detected'] = detection_result['is_hallucination']
                    result['hallucination_score'] = detection_result['hallucination_score']
                    result['hallucination_details'] = {
                        'num_contexts': detection_result['num_contexts'],
                        'num_contradictions': detection_result['num_contradictions'],
                        'aggregation': detection_result['aggregation']
                    }

                    if detection_result['is_hallucination']:
                        logger.warning(
                            f"Hallucination detected! Score: {detection_result['hallucination_score']:.2f} "
                            f"({detection_result['num_contradictions']}/{detection_result['num_contexts']} contexts)"
                        )
                    else:
                        logger.info("No hallucinations detected")

                except Exception as e:
                    logger.error(f"Hallucination detection failed: {e}")
                    result['hallucination_detected'] = None
                    result['hallucination_error'] = str(e)
                    detection_result = None

            elif detect_hallucinations and not self.hallucination_detector:
                logger.warning("Hallucination detection requested but detector not available")
                result['hallucination_detected'] = None
                detection_result = None
            else:
                result['hallucination_detected'] = False
                detection_result = None

            # Adaptive gating
            if gating_enabled and detection_result:
                gating_stats = self._compute_uncertainty_stats(
                    detection_result,
                    gating_config.get("uncertainty_source")
                )
                gating_stats.update({
                    "retrieval_max_score": retrieval_stats.get("max_score", 0.0),
                    "retrieval_mean_score": retrieval_stats.get("mean_score", 0.0)
                })
                decision = self._decide_gating_action(gating_stats, gating_config)

                if decision == "retrieve_more" and gating_attempts < max_retries:
                    gating_attempts += 1
                    gating_action = "retrieve_more"
                    next_k = int(current_k * k_multiplier)
                    if max_k is not None:
                        next_k = min(next_k, int(max_k))
                    if next_k == current_k:
                        break
                    current_k = next_k
                    logger.info(f"Gating triggered: retrieving more (k={current_k})")
                    continue

                if decision in {"abstain", "retrieve_more"}:
                    if decision == "abstain":
                        gating_action = "abstain"
                    result['answer'] = abstain_message
                else:
                    gating_action = "none"

            break

        if gating_enabled:
            result['gating'] = {
                'enabled': True,
                'strategy': gating_config.get("strategy", "abstain"),
                'action': gating_action,
                'attempts': gating_attempts,
                'k_used': current_k,
                'thresholds': {
                    'contradiction_rate': gating_config.get("contradiction_rate_threshold"),
                    'contradiction_prob': gating_config.get("contradiction_prob_threshold"),
                    'uncertainty': gating_config.get("uncertainty_threshold")
                },
                'stats': gating_stats
            }

        if return_context:
            result['context'] = retrieved_docs

        logger.info("Query complete")
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
        embedder_type = embedder_config.get('type', 'huggingface')  # mgte, huggingface, etc.
        embedder = EmbedderFactory.create(embedder_type, embedder_config)

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

        # --- Reranker (EBCAR, MGTE, etc.) ---
        reranker = None
        reranker_cfg = config.get("reranker", {})

        if reranker_cfg:
            reranker_type = reranker_cfg["type"]
            reranker = RerankerFactory.create(
                reranker_type,
                {
                    "embedder": embedder,
                    **reranker_cfg
                }
            )
            reranker_top_k = reranker_cfg.get("top_k")
            logger.info(f"Loaded reranker: {reranker_type}")
        else:
            logger.info("No reranker specified in config.")
            reranker_top_k = None

        # Hallucination Detector (optional)
        hallucination_detector = None
        training_config = config.get('training', {})
        model_path = training_config.get('output', {}).get('final_model_dir')
        detector_config = config.get('hallucination_detector', {})

        if detector_config:
            model_path = detector_config.get('model_path', model_path)

        if model_path and Path(model_path).exists():
            try:
                from src.rag.hallucination_detector import HallucinationDetector

                logger.info(f"Loading hallucination detector from: {model_path}")
                hallucination_detector = HallucinationDetector(
                    model_path=model_path,
                    max_length=detector_config.get(
                        'max_length',
                        training_config.get('data', {}).get('max_seq_length', 256)
                    ),
                    device=detector_config.get('device'),
                    base_model=detector_config.get('base_model'),
                    lora_config=detector_config.get('lora'),
                    mc_dropout_samples=detector_config.get('mc_dropout_samples', 1),
                    swag_config=detector_config.get('swag')
                )
                logger.info("Hallucination detector loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load hallucination detector: {e}")
                logger.warning("Continuing without hallucination detection")
        else:
            logger.info("No hallucination detector model found - continuing without hallucination detection")

        return cls(
            data_manager=data_manager,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
            retriever=retriever,
            reranker=reranker,
            reranker_top_k=reranker_top_k,
            hallucination_detector=hallucination_detector,
            gating_config=config.get('gating', {})
        )
