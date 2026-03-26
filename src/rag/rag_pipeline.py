"""
Main RAG Pipeline orchestrator.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import OrderedDict
import numpy as np
from src.dataset.data_manager import DataManager
from src.chunking.base_chunker import ChunkerFactory
from src.embeddings.base_embedder import EmbedderFactory
from src.reranking.base_reranker import RerankerFactory
from src.vector_stores.base_store import VectorStoreFactory
from src.rag.llm_wrapper import HuggingFaceLLM
from src.rag.openrouter_llm import OpenRouterLLM
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
        gating_config: Optional[Dict[str, Any]] = None,
        source_config: Optional[Dict[str, Any]] = None,
        hallucination_aggregation: str = "any",
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
        self.source_config = source_config or {}
        self.hallucination_aggregation = hallucination_aggregation or "any"
        self._source_embedding_cache_size = int(
            self.source_config.get("embedding_cache_size", 20000)
        )
        self._source_embedding_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

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
        entailment_probs = []
        neutral_probs = []
        conflict_masses = []
        uncertainties = []
        confidences = []
        top2_margins = []
        label_counts = {
            "entailment": 0,
            "neutral": 0,
            "contradiction": 0,
        }
        for result in individual:
            scores = result.get("scores")
            if scores:
                entailment_prob = float(scores.get("entailment", 0.0))
                neutral_prob = float(scores.get("neutral", 0.0))
                contradiction_prob = float(scores.get("contradiction", 0.0))
                contradiction_probs.append(contradiction_prob)
                entailment_probs.append(entailment_prob)
                neutral_probs.append(neutral_prob)
                conflict_masses.append(min(entailment_prob, contradiction_prob))
                if uncertainty_source == "entropy":
                    uncertainties.append(result.get("uncertainty_entropy", 0.0))
                elif uncertainty_source == "variance":
                    uncertainties.append(result.get("uncertainty_variance", 0.0))
                elif uncertainty_source == "contradiction_variance":
                    uncertainties.append(result.get("contradiction_variance", 0.0))
                elif uncertainty_source == "logit_mi":
                    uncertainties.append(result.get("uncertainty_logit_mi", 0.0))
                elif uncertainty_source == "logit_variance":
                    uncertainties.append(result.get("uncertainty_logit_variance", 0.0))
                elif uncertainty_source == "logit_entropy":
                    uncertainties.append(result.get("uncertainty_logit_entropy", 0.0))
                elif uncertainty_source == "rep_mi":
                    uncertainties.append(result.get("uncertainty_rep_mi", 0.0))
                elif uncertainty_source == "rep_variance":
                    uncertainties.append(result.get("uncertainty_rep_variance", 0.0))
                elif uncertainty_source == "rep_entropy":
                    uncertainties.append(result.get("uncertainty_rep_entropy", 0.0))
                else:
                    max_prob = max(scores.values())
                    uncertainties.append(1.0 - max_prob)
                confidences.append(max(scores.values()))
                sorted_probs = sorted(scores.values(), reverse=True)
                if len(sorted_probs) >= 2:
                    top2_margins.append(float(sorted_probs[0] - sorted_probs[1]))
            else:
                contradiction_probs.append(1.0 if result.get("is_hallucination") else 0.0)
                uncertainties.append(1.0 - result.get("confidence", 0.0))
                confidences.append(float(result.get("confidence", 0.0)))

            label = result.get("label")
            if label in label_counts:
                label_counts[label] += 1

        contradiction_prob_mean = sum(contradiction_probs) / len(contradiction_probs)
        uncertainty_mean = sum(uncertainties) / len(uncertainties)
        contradiction_prob_std = float(np.std(np.array(contradiction_probs, dtype=float)))
        contradiction_support_values = [
            max(0.0, c - n) for c, n in zip(contradiction_probs, neutral_probs)
        ] if contradiction_probs and neutral_probs else []
        contradiction_soft_values = [
            c * (1.0 - n) for c, n in zip(contradiction_probs, neutral_probs)
        ] if contradiction_probs and neutral_probs else []
        entailment_prob_mean = (
            float(sum(entailment_probs) / len(entailment_probs))
            if entailment_probs else 0.0
        )
        neutral_prob_mean = (
            float(sum(neutral_probs) / len(neutral_probs))
            if neutral_probs else 0.0
        )
        conflict_mass_mean = (
            float(sum(conflict_masses) / len(conflict_masses))
            if conflict_masses else 0.0
        )
        confidence_mean = (
            float(sum(confidences) / len(confidences))
            if confidences else 0.0
        )
        top2_margin_mean = (
            float(sum(top2_margins) / len(top2_margins))
            if top2_margins else 0.0
        )
        contradiction_neutral_gap_mean = (
            float(
                sum((c - n) for c, n in zip(contradiction_probs, neutral_probs))
                / len(contradiction_probs)
            )
            if contradiction_probs and neutral_probs else 0.0
        )
        contradiction_support_mean = (
            float(sum(contradiction_support_values) / len(contradiction_support_values))
            if contradiction_support_values else 0.0
        )
        contradiction_soft_mean = (
            float(sum(contradiction_soft_values) / len(contradiction_soft_values))
            if contradiction_soft_values else 0.0
        )
        if contradiction_support_values:
            top_k = max(1, int(np.ceil(len(contradiction_support_values) / 2.0)))
            contradiction_support_topk = float(
                sum(sorted(contradiction_support_values, reverse=True)[:top_k]) / top_k
            )
            contradiction_soft_topk = float(
                sum(sorted(contradiction_soft_values, reverse=True)[:top_k]) / top_k
            )
        else:
            contradiction_support_topk = 0.0
            contradiction_soft_topk = 0.0
        contradiction_weighted_rate = (
            float(sum(
                conf if result.get("label") == "contradiction" else 0.0
                for conf, result in zip(confidences, individual)
            ) / len(individual))
            if individual and confidences else 0.0
        )
        entailment_neutral_gap_mean = (
            float(
                sum((e - n) for e, n in zip(entailment_probs, neutral_probs))
                / len(entailment_probs)
            )
            if entailment_probs and neutral_probs else 0.0
        )

        total_labels = sum(label_counts.values())
        if total_labels > 0:
            dist = np.array(
                [
                    label_counts["entailment"],
                    label_counts["neutral"],
                    label_counts["contradiction"],
                ],
                dtype=float
            ) / float(total_labels)
            nonzero = dist[dist > 0.0]
            label_entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(3.0))
            label_disagreement = float(1.0 - dist.max())
        else:
            label_entropy = 0.0
            label_disagreement = 0.0

        detector_conflict = self._clamp01(
            0.50 * conflict_mass_mean
            + 0.30 * label_disagreement
            + 0.20 * contradiction_prob_mean
        )
        detector_conflict_consensus = self._clamp01(
            0.45 * float(detection_result.get("hard_contradiction_rate", 0.0) or 0.0)
            + 0.25 * contradiction_weighted_rate
            + 0.20 * conflict_mass_mean
            + 0.10 * label_disagreement
        )

        return {
            "contradiction_rate": detection_result.get("hallucination_score", 0.0),
            "hard_contradiction_rate": detection_result.get("hard_contradiction_rate", 0.0),
            "contradiction_prob_mean": contradiction_prob_mean,
            "hallucination_prob_mean": detection_result.get("hallucination_prob_mean", contradiction_prob_mean),
            "hallucination_prob_topk": detection_result.get("hallucination_prob_topk", contradiction_prob_mean),
            "contradiction_margin_mean": detection_result.get("contradiction_margin_mean", 0.0),
            "uncertainty_mean": uncertainty_mean,
            # Additional conflict/ambiguity signals for aleatoric modeling.
            "entailment_prob_mean": entailment_prob_mean,
            "neutral_prob_mean": neutral_prob_mean,
            "contradiction_prob_std": contradiction_prob_std,
            "conflict_mass_mean": conflict_mass_mean,
            "label_entropy": label_entropy,
            "label_disagreement": label_disagreement,
            "confidence_mean": confidence_mean,
            "top2_margin_mean": top2_margin_mean,
            "entailment_rate": float(label_counts["entailment"]) / float(total_labels or 1),
            "neutral_rate": float(label_counts["neutral"]) / float(total_labels or 1),
            "contradiction_label_rate": float(label_counts["contradiction"]) / float(total_labels or 1),
            "entailment_count": float(label_counts["entailment"]),
            "neutral_count": float(label_counts["neutral"]),
            "contradiction_count": float(label_counts["contradiction"]),
            "contradiction_neutral_gap_mean": contradiction_neutral_gap_mean,
            "contradiction_support_mean": contradiction_support_mean,
            "contradiction_support_topk": contradiction_support_topk,
            "contradiction_soft_mean": contradiction_soft_mean,
            "contradiction_soft_topk": contradiction_soft_topk,
            "contradiction_weighted_rate": contradiction_weighted_rate,
            "detector_conflict": detector_conflict,
            "detector_conflict_consensus": detector_conflict_consensus,
            "entailment_neutral_gap_mean": entailment_neutral_gap_mean,
        }

    def _get_selected_metric(
        self,
        stats: Dict[str, float],
        gating_config: Dict[str, Any],
        config_key: str,
        default_key: str,
        default: float = 0.0,
    ) -> float:
        metric_name = str(gating_config.get(config_key, default_key) or default_key)
        value = stats.get(metric_name)
        if not isinstance(value, (int, float)):
            value = stats.get(default_key, default)
            metric_name = default_key
        stats[f"selected_{config_key}"] = float(value)
        return float(value)

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

    def _compute_source_consistency(
        self,
        retrieved_docs: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Optional[float]:
        if not retrieved_docs:
            return None

        max_docs = int(top_k or self.source_config.get("consistency_top_k", 5))
        docs = retrieved_docs[:max_docs]
        texts = [doc.get("content", "") for doc in docs if doc.get("content")]
        if len(texts) < 2:
            return 1.0

        try:
            vectors = self._get_cached_source_embeddings(texts)
            if vectors.ndim != 2 or vectors.shape[0] < 2:
                return None

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
            sim = vectors @ vectors.T
            tri = sim[np.triu_indices(sim.shape[0], k=1)]
            if tri.size == 0:
                return 1.0
            return float(tri.mean())
        except Exception as e:
            logger.warning(f"Failed to compute source consistency: {e}")
            return None

    def _get_cached_source_embeddings(self, texts: List[str]) -> np.ndarray:
        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        misses: List[str] = []
        miss_indices: List[int] = []

        for idx, text in enumerate(texts):
            cached = self._source_embedding_cache.get(text)
            if cached is not None:
                self._source_embedding_cache.move_to_end(text)
                vectors[idx] = cached
            else:
                misses.append(text)
                miss_indices.append(idx)

        if misses:
            embedded = self.embedder.embed_batch(misses)
            for idx, text, vec in zip(miss_indices, misses, embedded):
                arr = np.asarray(vec, dtype=float)
                vectors[idx] = arr
                self._source_embedding_cache[text] = arr
                self._source_embedding_cache.move_to_end(text)
                while len(self._source_embedding_cache) > self._source_embedding_cache_size:
                    self._source_embedding_cache.popitem(last=False)

        dense_vectors = [
            vec if vec is not None else np.asarray(self.embedder.embed_text(texts[i]), dtype=float)
            for i, vec in enumerate(vectors)
        ]
        return np.array(dense_vectors, dtype=float)

    def _merge_gating_config(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not override:
            return dict(self.gating_config)

        merged = dict(self.gating_config)
        merged.update(override)
        return merged

    def _clamp01(self, x: Optional[float], default: float = 0.0) -> float:
        if not isinstance(x, (int, float)):
            return float(default)
        return float(min(1.0, max(0.0, x)))

    def _risk_budgeted_action(self, stats: Dict[str, float], gating_config: Dict[str, Any]) -> str:
        """
        Cost-aware 3-action controller: answer vs retrieve_more vs abstain.

        This is intentionally simple and config-driven so we can iterate without changing
        the baseline stack. It fuses multiple risk signals into a single risk score and
        then chooses the lowest-cost action.
        """
        # Risk signals (all normalized/clamped to [0,1] where possible)
        contradiction_rate = self._clamp01(self._get_selected_metric(
            stats, gating_config, "contradiction_rate_metric", "contradiction_rate", 0.0
        ))
        contradiction_prob = self._clamp01(self._get_selected_metric(
            stats, gating_config, "contradiction_prob_metric", "contradiction_prob_mean", 0.0
        ))
        uncertainty = self._clamp01(self._get_selected_metric(
            stats, gating_config, "uncertainty_metric", "uncertainty_mean", 0.0
        ))
        source_consistency = stats.get("source_consistency")
        sc = self._clamp01(source_consistency, default=1.0)

        retrieval_max = self._clamp01(stats.get("retrieval_max_score", 0.0))
        retrieval_mean = self._clamp01(stats.get("retrieval_mean_score", 0.0))
        retrieval_gap = self._clamp01(max(0.0, retrieval_max - retrieval_mean))
        entailment_prob = self._clamp01(stats.get("entailment_prob_mean", 0.0))
        neutral_prob = self._clamp01(stats.get("neutral_prob_mean", 0.0))
        label_entropy = self._clamp01(stats.get("label_entropy", 0.0))
        label_disagreement = self._clamp01(stats.get("label_disagreement", 0.0))
        low_confidence = self._clamp01(
            1.0 - max(entailment_prob, neutral_prob, contradiction_prob)
        )
        uncertainty_scale = float(gating_config.get("uncertainty_scale", 0.05))
        if uncertainty_scale <= 0.0:
            uncertainty_scale = 0.05
        uncertainty_scaled = self._clamp01(uncertainty / uncertainty_scale)
        ambiguity = self._clamp01(
            0.50 * neutral_prob
            + 0.25 * label_disagreement
            + 0.25 * label_entropy
        )

        weights = gating_config.get("risk_weights") or {}
        w_cr = float(weights.get("contradiction_rate", 1.0))
        w_cp = float(weights.get("contradiction_prob", 0.0))
        w_u = float(weights.get("uncertainty", 0.0))
        w_inc = float(weights.get("inconsistency", 0.0))  # uses (1 - source_consistency)
        w_ret = float(weights.get("retrieval_weakness", 0.0))  # uses (1 - retrieval_max)
        w_lc = float(weights.get("low_confidence", 0.0))
        w_rg = float(weights.get("retrieval_gap", 0.0))
        w_amb = float(weights.get("ambiguity", 0.0))
        w_floor = float(gating_config.get("base_risk_floor", 0.0))
        risk_formula = str(gating_config.get("risk_formula", "linear")).strip().lower()

        if risk_formula == "saturating_union":
            component_values = {
                "contradiction_rate": self._clamp01(w_cr * contradiction_rate),
                "contradiction_prob": self._clamp01(w_cp * contradiction_prob),
                "uncertainty": self._clamp01(w_u * uncertainty_scaled),
                "source_inconsistency": self._clamp01(w_inc * (1.0 - sc)),
                "retrieval_weakness": self._clamp01(w_ret * (1.0 - retrieval_max)),
                "low_confidence": self._clamp01(w_lc * low_confidence),
                "retrieval_gap": self._clamp01(w_rg * retrieval_gap),
                "ambiguity": self._clamp01(w_amb * ambiguity),
            }
            components = list(component_values.values())
            remaining_safe = 1.0
            for comp in components:
                remaining_safe *= (1.0 - comp)
            risk = self._clamp01(max(w_floor, 1.0 - remaining_safe))
        else:
            component_values = {
                "contradiction_rate": w_cr * contradiction_rate,
                "contradiction_prob": w_cp * contradiction_prob,
                "uncertainty": w_u * uncertainty,
                "source_inconsistency": w_inc * (1.0 - sc),
                "retrieval_weakness": w_ret * (1.0 - retrieval_max),
                "low_confidence": w_lc * low_confidence,
                "retrieval_gap": w_rg * retrieval_gap,
                "ambiguity": w_amb * ambiguity,
            }
            risk = sum(component_values.values())
            risk = self._clamp01(max(w_floor, risk))

        # Action costs
        # - answer_cost: proportional to risk
        # - retrieve_more: pay a fixed cost + assume risk shrinks by a factor
        # - abstain: fixed cost (coverage penalty)
        retrieve_more_cost = float(gating_config.get("retrieve_more_cost", 0.10))
        retrieve_more_risk_factor = float(gating_config.get("retrieve_more_risk_factor", 0.75))
        abstain_cost = float(gating_config.get("abstain_cost", 0.35))

        answer_cost = risk
        retrieve_cost = retrieve_more_cost + retrieve_more_risk_factor * risk

        # Optional guard: if retrieval quality is already very weak, retrieving more is unlikely to help.
        min_rm = gating_config.get("retrieve_more_min_retrieval_max")
        min_rmean = gating_config.get("retrieve_more_min_retrieval_mean")
        if isinstance(min_rm, (int, float)) and retrieval_max < float(min_rm):
            retrieve_cost = float("inf")
        if isinstance(min_rmean, (int, float)) and retrieval_mean < float(min_rmean):
            retrieve_cost = float("inf")

        # Store for downstream reporting (eval scripts ignore unknown keys safely).
        stats["risk_score"] = float(risk)
        stats["policy_cost_answer"] = float(answer_cost)
        stats["policy_cost_retrieve_more"] = float(retrieve_cost)
        stats["policy_cost_abstain"] = float(abstain_cost)
        stats["risk_low_confidence"] = float(low_confidence)
        stats["risk_retrieval_gap"] = float(retrieval_gap)
        stats["risk_ambiguity"] = float(ambiguity)
        stats["risk_uncertainty_scaled"] = float(uncertainty_scaled)
        stats["risk_component_contradiction_rate"] = float(component_values["contradiction_rate"])
        stats["risk_component_contradiction_prob"] = float(component_values["contradiction_prob"])
        stats["risk_component_uncertainty"] = float(component_values["uncertainty"])
        stats["risk_component_source_inconsistency"] = float(component_values["source_inconsistency"])
        stats["risk_component_retrieval_weakness"] = float(component_values["retrieval_weakness"])
        stats["risk_component_low_confidence"] = float(component_values["low_confidence"])
        stats["risk_component_retrieval_gap"] = float(component_values["retrieval_gap"])
        stats["risk_component_ambiguity"] = float(component_values["ambiguity"])
        stats["risk_detector_component"] = float(
            component_values["contradiction_rate"]
            + component_values["contradiction_prob"]
            + component_values["uncertainty"]
            + component_values["low_confidence"]
            + component_values["ambiguity"]
        )
        stats["risk_retrieval_component"] = float(
            component_values["retrieval_weakness"] + component_values["retrieval_gap"]
        )
        stats["risk_source_component"] = float(component_values["source_inconsistency"])

        # Choose minimal cost action.
        # Tie-breaker is safety-biased: abstain > retrieve_more > answer.
        costs = {
            "none": answer_cost,
            "retrieve_more": retrieve_cost,
            "abstain": abstain_cost,
        }
        best = min(costs.items(), key=lambda kv: (kv[1], 0 if kv[0] == "abstain" else 1 if kv[0] == "retrieve_more" else 2))
        return best[0]

    def _decide_gating_action(self, stats: Dict[str, float], gating_config: Dict[str, Any]) -> str:
        contradiction_rate = self._get_selected_metric(
            stats, gating_config, "contradiction_rate_metric", "contradiction_rate", 0.0
        )
        contradiction_prob = self._get_selected_metric(
            stats, gating_config, "contradiction_prob_metric", "contradiction_prob_mean", 0.0
        )
        uncertainty = self._get_selected_metric(
            stats, gating_config, "uncertainty_metric", "uncertainty_mean", 0.0
        )
        source_consistency = stats.get("source_consistency")
        max_retrieval_score = stats.get("retrieval_max_score", 1.0)
        mean_retrieval_score = stats.get("retrieval_mean_score", 1.0)

        contradiction_rate_threshold = gating_config.get("contradiction_rate_threshold", 0.34)
        contradiction_prob_threshold = gating_config.get("contradiction_prob_threshold", 0.5)
        uncertainty_threshold = gating_config.get("uncertainty_threshold", 0.3)
        min_retrieval_score = gating_config.get("min_retrieval_score")
        min_mean_retrieval_score = gating_config.get("min_mean_retrieval_score")
        source_consistency_threshold = gating_config.get("source_consistency_threshold")

        retrieval_low = False
        if isinstance(min_retrieval_score, (int, float)) and max_retrieval_score < float(min_retrieval_score):
            retrieval_low = True
        if isinstance(min_mean_retrieval_score, (int, float)) and mean_retrieval_score < float(min_mean_retrieval_score):
            retrieval_low = True
        if isinstance(source_consistency_threshold, (int, float)) and source_consistency is not None:
            if source_consistency < float(source_consistency_threshold):
                retrieval_low = True

        contradiction_rate_trigger = contradiction_rate >= contradiction_rate_threshold
        contradiction_prob_trigger = contradiction_prob >= contradiction_prob_threshold
        uncertainty_trigger = uncertainty >= uncertainty_threshold
        source_consistency_trigger = (
            isinstance(source_consistency_threshold, (int, float))
            and source_consistency is not None
            and source_consistency < float(source_consistency_threshold)
        )
        stats["gate_trigger_contradiction_rate"] = float(1.0 if contradiction_rate_trigger else 0.0)
        stats["gate_trigger_contradiction_prob"] = float(1.0 if contradiction_prob_trigger else 0.0)
        stats["gate_trigger_uncertainty"] = float(1.0 if uncertainty_trigger else 0.0)
        stats["gate_trigger_retrieval_low"] = float(1.0 if retrieval_low else 0.0)
        stats["gate_trigger_source_consistency"] = float(1.0 if source_consistency_trigger else 0.0)

        strategy = gating_config.get("strategy", "abstain")
        if strategy == "risk_budgeted":
            # Policy is 3-action and does not require a separate should_gate precheck.
            return self._risk_budgeted_action(stats, gating_config)

        should_gate = (
            contradiction_rate >= contradiction_rate_threshold or
            contradiction_prob >= contradiction_prob_threshold or
            uncertainty >= uncertainty_threshold or
            retrieval_low
        )

        if not should_gate:
            return "none"

        return strategy

    def query(
        self,
        query_text: str,
        k: Optional[int] = None,
        return_context: bool = False,
        return_sources: bool = False,
        include_sources_in_answer: bool = False,
        detect_hallucinations: bool = True,
        hallucination_aggregation: Optional[str] = None,
        gating: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            query_text: User query
            k: Number of documents to retrieve
            return_context: Include retrieved context in response
            return_sources: Include formatted sources list in response
            include_sources_in_answer: Append sources to answer text
            detect_hallucinations: Run hallucination detection on answer
            hallucination_aggregation: How to aggregate hallucination detection
                                      across contexts ('any', 'majority', 'all')

        Returns:
            Dictionary with answer, optional context, and hallucination detection results
        """
        logger.info(f"Processing query: {query_text[:100]}...")
        aggregation_mode = hallucination_aggregation or self.hallucination_aggregation or "any"

        gating_config = self._merge_gating_config(gating)
        source_consistency_gate = gating_config.get("source_consistency_threshold") is not None
        gating_enabled = bool(gating_config.get("enabled", False)) and (
            bool(self.hallucination_detector) or source_consistency_gate
        )
        gating_action = "none"
        gating_stats = None
        gating_attempts = 0
        final_retrieved_docs: List[Dict[str, Any]] = []

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
                    'sources': [] if return_sources else None,
                    'num_docs_retrieved': 0,
                    'hallucination_detected': False
                }

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            retrieval_stats = self._compute_retrieval_stats(retrieved_docs)
            source_consistency = None
            source_metric_needed = False
            if gating_enabled:
                strategy_name = str(gating_config.get("strategy", "") or "").strip().lower()
                source_metric_needed = (
                    gating_config.get("source_consistency_threshold") is not None
                    or strategy_name == "risk_budgeted"
                    or str(gating_config.get("contradiction_rate_metric", "")).strip()
                    in {"source_consistency", "source_inconsistency", "combined_conflict"}
                    or str(gating_config.get("contradiction_prob_metric", "")).strip()
                    in {"source_consistency", "source_inconsistency", "combined_conflict"}
                    or str(gating_config.get("uncertainty_metric", "")).strip()
                    in {"source_consistency", "source_inconsistency", "combined_conflict"}
                )
            if gating_enabled and source_metric_needed:
                source_consistency = self._compute_source_consistency(
                    retrieved_docs,
                    gating_config.get("source_consistency_top_k")
                )

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
                        aggregation=aggregation_mode
                    )

                    result['hallucination_detected'] = detection_result['is_hallucination']
                    result['hallucination_score'] = detection_result['hallucination_score']
                    result['hallucination_details'] = {
                        'num_contexts': detection_result['num_contexts'],
                        'num_contradictions': detection_result['num_contradictions'],
                        'aggregation': detection_result['aggregation']
                    }
                    for key in (
                        "hard_contradiction_rate",
                        "hallucination_prob_mean",
                        "hallucination_prob_topk",
                        "contradiction_margin_mean",
                        "contradiction_neutral_gap_mean",
                    ):
                        if key in detection_result:
                            result['hallucination_details'][key] = detection_result[key]

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
            if gating_enabled:
                gating_stats = {
                    "retrieval_max_score": retrieval_stats.get("max_score", 0.0),
                    "retrieval_mean_score": retrieval_stats.get("mean_score", 0.0),
                    "source_consistency": source_consistency,
                    "source_inconsistency": (
                        self._clamp01(1.0 - float(source_consistency))
                        if isinstance(source_consistency, (int, float))
                        else 0.0
                    ),
                }
                if detection_result:
                    gating_stats.update(
                        self._compute_uncertainty_stats(
                            detection_result,
                            gating_config.get("uncertainty_source")
                        )
                    )
                else:
                    gating_stats.update({
                        "contradiction_rate": 0.0,
                        "contradiction_prob_mean": 0.0,
                        "uncertainty_mean": 0.0
                    })

                gating_stats["combined_conflict"] = self._clamp01(
                    0.45 * float(
                        gating_stats.get(
                            "detector_conflict_consensus",
                            gating_stats.get("detector_conflict", 0.0),
                        )
                        or 0.0
                    )
                    + 0.35 * float(gating_stats.get("source_inconsistency", 0.0) or 0.0)
                    + 0.20 * max(
                        0.0,
                        float(gating_stats.get("retrieval_max_score", 0.0) or 0.0)
                        - float(gating_stats.get("retrieval_mean_score", 0.0) or 0.0)
                    )
                )

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
                    gating_action = decision
                    result['answer'] = abstain_message
                else:
                    gating_action = "none"

            final_retrieved_docs = retrieved_docs
            break

        if gating_enabled:
            result['gating'] = {
                'enabled': True,
                'strategy': gating_config.get("strategy", "abstain"),
                'action': gating_action,
                'attempts': gating_attempts,
                'k_used': current_k,
                'thresholds': {
                    'contradiction_rate_metric': gating_config.get("contradiction_rate_metric", "contradiction_rate"),
                    'contradiction_prob_metric': gating_config.get("contradiction_prob_metric", "contradiction_prob_mean"),
                    'uncertainty_metric': gating_config.get("uncertainty_metric", "uncertainty_mean"),
                    'contradiction_rate': gating_config.get("contradiction_rate_threshold"),
                    'contradiction_prob': gating_config.get("contradiction_prob_threshold"),
                    'uncertainty': gating_config.get("uncertainty_threshold"),
                    'source_consistency': gating_config.get("source_consistency_threshold"),
                    'min_retrieval_score': gating_config.get("min_retrieval_score"),
                    'min_mean_retrieval_score': gating_config.get("min_mean_retrieval_score"),
                },
                'stats': gating_stats
            }

        is_abstain = bool(abstain_message) and result.get("answer") == abstain_message
        if gating_action == "abstain":
            is_abstain = True

        if return_context:
            result['context'] = final_retrieved_docs

        if return_sources or include_sources_in_answer:
            if is_abstain:
                if return_sources:
                    result['sources'] = []
            else:
                sources = self._build_source_entries(final_retrieved_docs)
                if return_sources:
                    result['sources'] = sources
                if include_sources_in_answer:
                    sources_text = self._format_sources_text(sources)
                    if sources_text:
                        result['answer'] = f"{result['answer']}\n\n{sources_text}"

        logger.info("Query complete")
        return result

    def _build_source_entries(
        self,
        retrieved_docs: List[Dict[str, Any]],
        max_sources: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        max_sources = int(max_sources or self.source_config.get("max_sources", 5))
        include_scores = bool(self.source_config.get("include_scores", False))
        entries: List[Dict[str, Any]] = []
        seen = set()

        for doc in retrieved_docs:
            metadata = doc.get("metadata") or {}
            raw_source = (
                metadata.get("title")
                or metadata.get("source")
                or metadata.get("url")
                or metadata.get("path")
            )
            if not raw_source:
                continue

            page = metadata.get("page") or metadata.get("page_number")
            section = metadata.get("section") or metadata.get("heading")
            key = (str(raw_source), str(page), str(section))
            if key in seen:
                continue
            seen.add(key)

            label = str(raw_source)
            if isinstance(raw_source, str):
                label = Path(raw_source).name or raw_source

            entry: Dict[str, Any] = {
                "source": str(raw_source),
                "label": label
            }

            if page is not None:
                entry["page"] = page
            if section:
                entry["section"] = section
            if include_scores and doc.get("score") is not None:
                try:
                    entry["score"] = float(doc["score"])
                except (TypeError, ValueError):
                    pass

            entries.append(entry)
            if len(entries) >= max_sources:
                break

        return entries

    def _format_sources_text(self, sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return ""

        lines = ["Sources:"]
        for idx, source in enumerate(sources, start=1):
            label = source.get("label") or source.get("source") or "source"
            extras = []
            if source.get("page") is not None:
                extras.append(f"p.{source['page']}")
            if source.get("section"):
                extras.append(f"sec:{source['section']}")
            if "score" in source:
                extras.append(f"score={source['score']:.2f}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"[{idx}] {label}{suffix}")

        return "\n".join(lines)

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
        llm_type = str(llm_config.get("type", "huggingface")).strip().lower()
        if llm_type in {"openrouter", "api", "openai_compatible"}:
            llm = OpenRouterLLM(llm_config)
        else:
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
        detector_enabled = bool(detector_config.get('enabled', True))

        if detector_config:
            model_path = detector_config.get('model_path', model_path)

        if not detector_enabled:
            logger.info("Hallucination detector disabled in config")
        elif model_path and Path(model_path).exists():
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
                        lora_config=detector_config.get('lora') or detector_config.get('lora_config'),
                        mc_dropout_samples=detector_config.get('mc_dropout_samples', 1),
                        swag_config=detector_config.get('swag'),
                        logit_sampling_config=detector_config.get('logit_sampling'),
                        representation_sampling_config=detector_config.get('representation_sampling')
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
            gating_config=config.get('gating', {}),
            source_config=config.get('sources', {}),
            hallucination_aggregation=(
                config.get("hallucination_aggregation")
                or detector_config.get("aggregation")
                or "any"
            ),
        )
