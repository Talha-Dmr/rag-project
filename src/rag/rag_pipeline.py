"""
Main RAG Pipeline orchestrator.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import OrderedDict
import re
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for some tooling paths.
    torch = None

from src.dataset.data_manager import DataManager
from src.chunking.base_chunker import ChunkerFactory
from src.embeddings.base_embedder import EmbedderFactory
from src.reranking.base_reranker import RerankerFactory
from src.vector_stores.base_store import VectorStoreFactory
from src.rag.llm_wrapper import HuggingFaceLLM
from src.rag.openrouter_llm import OpenRouterLLM
from src.rag.answer_quality import audit_answer_quality, build_quality_feedback, split_claims
from src.rag.retriever import Retriever
from src.core.base_classes import BaseLLM, BaseReranker
from src.core.logger import get_logger

logger = get_logger(__name__)

MODEL_ABSTAIN_PHRASES = (
    "i don't know based on the provided context",
    "i do not know based on the provided context",
    "the provided context does not contain enough information",
    "the context does not contain enough information",
)

SOURCE_FAMILY_PATTERNS = {
    "bcbs": re.compile(r"\b(?:bcbs|basel committee|bis)\b", re.IGNORECASE),
    "eba": re.compile(r"\b(?:eba|european banking authority)\b", re.IGNORECASE),
    "ecb": re.compile(
        r"\b(?:ecb|european central bank|single supervisory mechanism|ssm)\b",
        re.IGNORECASE,
    ),
    "pra": re.compile(
        r"\b(?:pra|prudential regulation authority|bank of england|boe)\b",
        re.IGNORECASE,
    ),
}

SOURCE_BALANCE_STOPWORDS = {
    "and", "are", "body", "data", "do", "for", "frame", "how", "its",
    "materials", "quality", "responsibility", "senior", "the", "their", "when",
}

FINREG_QUERY_EXPANSIONS = [
    (
        re.compile(r"\bstress(?:\s|-)?test|srep|icaap|ilaap\b", re.IGNORECASE),
        "adverse scenarios risk concentrations capital adequacy resilience supervisory review SREP ICAAP ILAAP",
    ),
    (
        re.compile(r"\bict|cyber|security risk|incident|resilience\b", re.IGNORECASE),
        "identify protect detect respond recover testing cybersecurity incident response business continuity",
    ),
    (
        re.compile(r"\boutsourc|cloud|third(?:\s|-)?party|service provider\b", re.IGNORECASE),
        "retained responsibility governance risk assessment monitoring audit access exit planning sub-outsourcing",
    ),
    (
        re.compile(r"\bclimate|esg|environmental|transition risk|physical risk\b", re.IGNORECASE),
        "identify measure manage monitor governance transition risk physical risk risk appetite strategy",
    ),
    (
        re.compile(r"\bmanual workaround|lineage|rdarr|bcbs 239|risk data\b", re.IGNORECASE),
        "temporary transition controls ownership auditability remediation plan data lineage accuracy completeness timeliness",
    ),
    (
        re.compile(r"\bintraday liquidity|payment|settlement\b", re.IGNORECASE),
        "payment settlement obligations liquidity risk monitoring tools stress scenarios reporting",
    ),
    (
        re.compile(r"\bremuneration|risk culture|board|senior management\b", re.IGNORECASE),
        "risk appetite board oversight staff training remuneration alignment governance risk culture incentives",
    ),
    (
        re.compile(r"\bmodel risk|model validation|ai[- ]based|complexity\b", re.IGNORECASE),
        "model development validation implementation use constraints governance controls limitations model changes",
    ),
]


class _NoopLLM(BaseLLM):
    """LLM placeholder for retrieval/detector-only workflows."""

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        raise RuntimeError("LLM is disabled for this pipeline config.")

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        quality_feedback: Optional[str] = None
    ) -> str:
        raise RuntimeError("LLM is disabled for this pipeline config.")


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
        retrieval_config: Optional[Dict[str, Any]] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        source_config: Optional[Dict[str, Any]] = None,
        answer_quality_config: Optional[Dict[str, Any]] = None,
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
        self.retrieval_config = retrieval_config or {}
        self.runtime_config = runtime_config or {}
        self.source_config = source_config or {}
        self.answer_quality_config = answer_quality_config or {}
        self.hallucination_aggregation = hallucination_aggregation or "any"
        self._source_embedding_cache_size = int(
            self.source_config.get("embedding_cache_size", 20000)
        )
        self._source_embedding_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        if hallucination_detector:
            logger.info("RAG Pipeline initialized with hallucination detection")
        else:
            logger.info("RAG Pipeline initialized")

    def _sequential_cuda_config(self) -> Dict[str, Any]:
        cfg = self.runtime_config.get("sequential_cuda", {})
        return cfg if isinstance(cfg, dict) else {}

    def _sequential_cuda_enabled(self) -> bool:
        cfg = self._sequential_cuda_config()
        return bool(cfg.get("enabled", False)) and torch is not None and torch.cuda.is_available()

    def _cuda_barrier(self, stage: str, when: str) -> None:
        if not self._sequential_cuda_enabled():
            return

        cfg = self._sequential_cuda_config()
        try:
            if bool(cfg.get("synchronize", True)):
                torch.cuda.synchronize()
            empty_before = bool(cfg.get("empty_cache_before", False)) and when == "before"
            empty_after = bool(cfg.get("empty_cache_after", False)) and when == "after"
            if empty_before or empty_after:
                torch.cuda.empty_cache()
            if bool(cfg.get("log_memory", False)) and when == "after":
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(
                    "CUDA stage %s complete: allocated=%.2fGiB reserved=%.2fGiB",
                    stage,
                    allocated,
                    reserved,
                )
        except Exception as exc:
            logger.warning("CUDA barrier failed around %s/%s: %s", stage, when, exc)

    def _run_cuda_stage(self, stage: str, fn, *args, **kwargs):
        self._cuda_barrier(stage, "before")
        try:
            return fn(*args, **kwargs)
        finally:
            self._cuda_barrier(stage, "after")

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

    def _detect_generated_answer(
        self,
        answer: str,
        context_texts: List[str],
        aggregation_mode: str,
        query_text: str,
        detect_hallucinations: bool,
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Run the detector for one generated answer and return result fields."""
        model_requested_abstain = self._model_requested_abstain(answer)
        result: Dict[str, Any] = {
            "model_requested_abstain": model_requested_abstain,
            "hallucination_detector_ran": False,
        }

        if model_requested_abstain:
            result["hallucination_detected"] = False
            return result, None

        if detect_hallucinations and self.hallucination_detector:
            logger.info("Checking for hallucinations...")
            try:
                result["hallucination_detector_ran"] = True
                detection_result = self._run_cuda_stage(
                    "detector",
                    self.hallucination_detector.verify_answer_with_contexts,
                    answer=answer,
                    contexts=context_texts,
                    aggregation=aggregation_mode,
                    question=query_text,
                )

                result["hallucination_detected"] = detection_result["is_hallucination"]
                result["answer_include_detected"] = detection_result.get("answer_include_detected")
                result["answer_include_risk"] = detection_result.get(
                    "answer_include_risk",
                    detection_result.get("unsupported_risk"),
                )
                result["answer_include_score"] = detection_result.get(
                    "answer_include_score",
                    detection_result.get("support_score"),
                )
                # Backward-compatible aliases.
                result["unsupported_answer_detected"] = detection_result.get("is_unsupported")
                result["unsupported_risk"] = detection_result.get("unsupported_risk")
                result["support_score"] = detection_result.get("support_score")
                result["hallucination_score"] = detection_result["hallucination_score"]
                result["hallucination_details"] = {
                    "num_contexts": detection_result["num_contexts"],
                    "num_contradictions": detection_result["num_contradictions"],
                    "num_answer_not_included": detection_result.get("num_answer_not_included"),
                    "num_unsupported": detection_result.get("num_unsupported"),
                    "aggregation": detection_result["aggregation"],
                }
                for key in (
                    "answer_include_risk",
                    "answer_include_prob_mean",
                    "answer_include_prob_topk",
                    "answer_include_threshold",
                    "answer_include_score",
                    "hard_answer_not_included_rate",
                    "unsupported_risk",
                    "unsupported_prob_mean",
                    "unsupported_prob_topk",
                    "unsupported_threshold",
                    "support_score",
                    "best_context_index",
                    "best_context_label",
                    "best_context_scores",
                    "hard_unsupported_rate",
                    "hard_contradiction_rate",
                    "hallucination_prob_mean",
                    "hallucination_prob_topk",
                    "contradiction_margin_mean",
                    "contradiction_neutral_gap_mean",
                ):
                    if key in detection_result:
                        result["hallucination_details"][key] = detection_result[key]

                if detection_result["is_hallucination"]:
                    logger.warning(
                        "Hallucination detected! Score: %.2f (%s/%s contexts)",
                        detection_result["hallucination_score"],
                        detection_result["num_contradictions"],
                        detection_result["num_contexts"],
                    )
                else:
                    logger.info("No hallucinations detected")
                return result, detection_result

            except Exception as e:
                logger.error(f"Hallucination detection failed: {e}")
                result["hallucination_detected"] = None
                result["hallucination_error"] = str(e)
                return result, None

        if detect_hallucinations and not self.hallucination_detector:
            logger.warning("Hallucination detection requested but detector not available")
            result["hallucination_detected"] = None
        else:
            result["hallucination_detected"] = False
        return result, None

    def _claim_support_stats(
        self,
        answer: str,
        context_texts: List[str],
        aggregation_mode: str,
        query_text: str,
        detect_hallucinations: bool,
    ) -> Dict[str, Any]:
        cfg = self.answer_quality_config or {}
        if not bool(cfg.get("claim_level", False)):
            return {}
        if not detect_hallucinations or not self.hallucination_detector:
            return {}

        claims = split_claims(answer, max_claims=int(cfg.get("max_claims", 6)))
        if not claims:
            return {
                "claim_count": 0,
                "claim_include_rate": 1.0,
                "claim_unsupported_rate": 0.0,
                "claim_answer_include_risk_max": 0.0,
                "unsupported_claims": [],
            }

        claim_results: list[dict[str, Any]] = []
        unsupported_claims: list[str] = []
        risks: list[float] = []
        included = 0
        claim_aggregation = str(cfg.get("claim_aggregation") or aggregation_mode)

        for claim in claims:
            try:
                claim_result = self._run_cuda_stage(
                    "claim_detector",
                    self.hallucination_detector.verify_answer_with_contexts,
                    answer=claim,
                    contexts=context_texts,
                    aggregation=claim_aggregation,
                    question=query_text,
                )
            except Exception as exc:
                logger.warning("Claim-level support check failed: %s", exc)
                continue

            risk = float(claim_result.get("answer_include_risk", 1.0) or 1.0)
            is_included = bool(claim_result.get("answer_include_detected"))
            risks.append(risk)
            included += int(is_included)
            if not is_included:
                unsupported_claims.append(claim)
            claim_results.append(
                {
                    "claim": claim,
                    "answer_include_detected": is_included,
                    "answer_include_risk": risk,
                    "answer_include_score": claim_result.get("answer_include_score"),
                    "best_context_label": claim_result.get("best_context_label"),
                }
            )

        checked = len(claim_results)
        if checked == 0:
            return {}
        include_rate = included / checked
        mean_risk = float(sum(risks) / len(risks)) if risks else 0.0
        return {
            "claim_count": checked,
            "claim_include_rate": float(include_rate),
            "claim_unsupported_rate": float(1.0 - include_rate),
            "claim_answer_include_risk_max": float(max(risks) if risks else 0.0),
            "claim_answer_include_risk_mean": mean_risk,
            "unsupported_claims": unsupported_claims,
            "claim_results": claim_results,
        }

    def _audit_answer_quality(
        self,
        query_text: str,
        answer: str,
        context_texts: List[str],
        aggregation_mode: str,
        detect_hallucinations: bool,
    ) -> Dict[str, Any]:
        if not bool((self.answer_quality_config or {}).get("enabled", False)):
            return {}

        audit = audit_answer_quality(
            query_text,
            answer,
            context_texts,
            self.answer_quality_config,
        )
        audit.update(
            self._claim_support_stats(
                answer,
                context_texts,
                aggregation_mode,
                query_text,
                detect_hallucinations,
            )
        )
        return audit

    def _quality_stats(self, audit: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not audit:
            return {
                "answer_completeness_score": 1.0,
                "answer_completeness_risk": 0.0,
                "context_coverage": 1.0,
                "claim_include_rate": 1.0,
                "claim_unsupported_rate": 0.0,
                "claim_answer_include_risk_max": 0.0,
            }
        return {
            "answer_completeness_score": float(audit.get("answer_completeness_score", 1.0) or 0.0),
            "answer_completeness_risk": float(audit.get("answer_completeness_risk", 0.0) or 0.0),
            "context_coverage": float(audit.get("context_coverage", 1.0) or 0.0),
            "claim_include_rate": float(audit.get("claim_include_rate", 1.0) or 0.0),
            "claim_unsupported_rate": float(audit.get("claim_unsupported_rate", 0.0) or 0.0),
            "claim_answer_include_risk_max": float(
                audit.get("claim_answer_include_risk_max", 0.0) or 0.0
            ),
        }

    def _should_rewrite_for_quality(self, audit: Dict[str, Any]) -> bool:
        if not audit or not bool((self.answer_quality_config or {}).get("rewrite_on_low_coverage", False)):
            return False
        min_coverage = float(self.answer_quality_config.get("min_concept_coverage", 0.55))
        min_context_coverage = float(self.answer_quality_config.get("min_context_coverage", min_coverage))
        min_claim_rate = float(self.answer_quality_config.get("min_claim_include_rate", 0.75))
        max_claim_risk = float(self.answer_quality_config.get("max_claim_answer_include_risk", 0.97))
        has_required = bool(audit.get("required_concepts"))
        if bool(audit.get("asks_specific_unsupported")) and audit.get("missing_concepts"):
            return True
        if has_required and float(audit.get("answer_completeness_score", 1.0)) < min_coverage:
            return True
        if has_required and float(audit.get("context_coverage", 1.0)) < min_context_coverage:
            return True
        if float(audit.get("claim_include_rate", 1.0)) < min_claim_rate:
            return True
        if float(audit.get("claim_answer_include_risk_max", 0.0)) >= max_claim_risk:
            return True
        return False

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
                "uncertainty_mean": 0.0,
                "answer_include_risk": detection_result.get("answer_include_risk", 0.0),
                "answer_include_score": detection_result.get("answer_include_score", 0.0),
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
            "hard_answer_not_included_rate": detection_result.get(
                "hard_answer_not_included_rate",
                detection_result.get("hard_unsupported_rate", 0.0),
            ),
            "answer_include_risk": detection_result.get(
                "answer_include_risk",
                detection_result.get("unsupported_risk", 0.0),
            ),
            "answer_include_prob_mean": detection_result.get(
                "answer_include_prob_mean",
                detection_result.get("unsupported_prob_mean", 0.0),
            ),
            "answer_include_prob_topk": detection_result.get(
                "answer_include_prob_topk",
                detection_result.get("unsupported_prob_topk", 0.0),
            ),
            "answer_include_score": detection_result.get(
                "answer_include_score",
                detection_result.get("support_score", 0.0),
            ),
            # Backward-compatible aliases.
            "hard_unsupported_rate": detection_result.get("hard_unsupported_rate", 0.0),
            "unsupported_risk": detection_result.get("unsupported_risk", 0.0),
            "unsupported_prob_mean": detection_result.get("unsupported_prob_mean", 0.0),
            "unsupported_prob_topk": detection_result.get("unsupported_prob_topk", 0.0),
            "support_score": detection_result.get("support_score", 0.0),
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
        answer_completeness_score = stats.get("answer_completeness_score", 1.0)
        context_coverage = stats.get("context_coverage", 1.0)
        claim_include_rate = stats.get("claim_include_rate", 1.0)
        claim_answer_include_risk_max = stats.get("claim_answer_include_risk_max", 0.0)
        source_consistency = stats.get("source_consistency")
        max_retrieval_score = stats.get("retrieval_max_score", 1.0)
        mean_retrieval_score = stats.get("retrieval_mean_score", 1.0)

        contradiction_rate_threshold = gating_config.get("contradiction_rate_threshold", 0.34)
        contradiction_prob_threshold = gating_config.get("contradiction_prob_threshold", 0.5)
        uncertainty_threshold = gating_config.get("uncertainty_threshold", 0.3)
        min_answer_completeness = gating_config.get("min_answer_completeness_score")
        min_context_coverage = gating_config.get("min_context_coverage")
        min_claim_include_rate = gating_config.get("min_claim_include_rate")
        max_claim_answer_include_risk = gating_config.get("max_claim_answer_include_risk")
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
        completeness_trigger = (
            isinstance(min_answer_completeness, (int, float))
            and isinstance(answer_completeness_score, (int, float))
            and answer_completeness_score < float(min_answer_completeness)
        )
        context_coverage_trigger = (
            isinstance(min_context_coverage, (int, float))
            and isinstance(context_coverage, (int, float))
            and context_coverage < float(min_context_coverage)
        )
        claim_include_trigger = (
            isinstance(min_claim_include_rate, (int, float))
            and isinstance(claim_include_rate, (int, float))
            and claim_include_rate < float(min_claim_include_rate)
        )
        claim_risk_trigger = (
            isinstance(max_claim_answer_include_risk, (int, float))
            and isinstance(claim_answer_include_risk_max, (int, float))
            and claim_answer_include_risk_max >= float(max_claim_answer_include_risk)
        )
        stats["gate_trigger_contradiction_rate"] = float(1.0 if contradiction_rate_trigger else 0.0)
        stats["gate_trigger_contradiction_prob"] = float(1.0 if contradiction_prob_trigger else 0.0)
        stats["gate_trigger_uncertainty"] = float(1.0 if uncertainty_trigger else 0.0)
        stats["gate_trigger_retrieval_low"] = float(1.0 if retrieval_low else 0.0)
        stats["gate_trigger_source_consistency"] = float(1.0 if source_consistency_trigger else 0.0)
        stats["gate_trigger_answer_completeness"] = float(1.0 if completeness_trigger else 0.0)
        stats["gate_trigger_context_coverage"] = float(1.0 if context_coverage_trigger else 0.0)
        stats["gate_trigger_claim_include"] = float(1.0 if claim_include_trigger else 0.0)
        stats["gate_trigger_claim_risk"] = float(1.0 if claim_risk_trigger else 0.0)

        strategy = gating_config.get("strategy", "abstain")
        if strategy == "risk_budgeted":
            # Policy is 3-action and does not require a separate should_gate precheck.
            return self._risk_budgeted_action(stats, gating_config)

        should_gate = (
            contradiction_rate >= contradiction_rate_threshold or
            contradiction_prob >= contradiction_prob_threshold or
            uncertainty >= uncertainty_threshold or
            retrieval_low or
            completeness_trigger or
            context_coverage_trigger or
            claim_include_trigger or
            claim_risk_trigger
        )

        if not should_gate:
            return "none"

        return strategy

    def _normalize_generated_answer(self, answer: str) -> str:
        if not answer:
            return ""

        cleaned = answer.strip()
        lower = cleaned.lower()

        for phrase in MODEL_ABSTAIN_PHRASES:
            idx = lower.find(phrase)
            if idx != -1:
                end = idx + len(phrase)
                return cleaned[:end].strip().rstrip(".") + "."

        return cleaned

    def _model_requested_abstain(self, answer: str) -> bool:
        lower = (answer or "").strip().lower()
        return any(phrase in lower for phrase in MODEL_ABSTAIN_PHRASES)

    def _expanded_retrieval_queries(self, query_text: str) -> List[str]:
        retrieval_cfg = getattr(self.retriever, "config", {}) or {}
        # Most call sites construct Retriever directly, so read the original
        # config from the pipeline when available.
        retrieval_cfg = getattr(self, "retrieval_config", retrieval_cfg)
        expansion_cfg = retrieval_cfg.get("query_expansion", {})
        if not bool(expansion_cfg.get("enabled", False)):
            return [query_text]

        queries = [query_text]
        max_queries = int(expansion_cfg.get("max_queries", 3) or 3)
        query_norm = query_text.strip()
        for pattern, expansion in FINREG_QUERY_EXPANSIONS:
            if pattern.search(query_norm):
                queries.append(f"{query_text} {expansion}")
                if len(queries) >= max_queries:
                    break

        fallback_terms = str(expansion_cfg.get("fallback_terms", "") or "").strip()
        if fallback_terms and len(queries) < max_queries:
            queries.append(f"{query_text} {fallback_terms}")

        deduped: List[str] = []
        seen = set()
        for query in queries:
            key = " ".join(query.lower().split())
            if key not in seen:
                seen.add(key)
                deduped.append(query)
        return deduped[:max_queries]

    def _retrieve_documents(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        retrieval_cfg = getattr(self, "retrieval_config", {})
        expansion_cfg = retrieval_cfg.get("query_expansion", {})
        queries = self._expanded_retrieval_queries(query_text)
        if len(queries) == 1:
            return self._run_cuda_stage("retrieval", self.retriever.retrieve, query_text, k=k)

        per_query_k = int(expansion_cfg.get("per_query_k", k) or k)
        max_candidates = int(expansion_cfg.get("max_candidates", max(k, per_query_k)) or max(k, per_query_k))
        merge_strategy = str(expansion_cfg.get("merge_strategy", "score") or "score").strip().lower()
        merged: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        buckets: list[list[Dict[str, Any]]] = []
        round_robin_seen_keys: set[str] = set()
        for expanded_query in queries:
            bucket: list[Dict[str, Any]] = []
            retrieved = self._run_cuda_stage(
                "retrieval",
                self.retriever.retrieve,
                expanded_query,
                k=per_query_k,
            )
            for doc in retrieved:
                content = str(doc.get("content", ""))
                metadata = doc.get("metadata") or {}
                source_key = metadata.get("source") or metadata.get("file_path") or metadata.get("title") or ""
                key = f"{source_key}|{content[:500]}"
                existing = merged.get(key)
                if existing is None or float(doc.get("score", 0.0) or 0.0) > float(existing.get("score", 0.0) or 0.0):
                    merged[key] = doc
                if key not in round_robin_seen_keys:
                    round_robin_seen_keys.add(key)
                    bucket.append(doc)
            buckets.append(bucket)

        if merge_strategy == "round_robin":
            docs = []
            for rank in range(per_query_k):
                for bucket in buckets:
                    if rank < len(bucket):
                        docs.append(bucket[rank])
                    if len(docs) >= max_candidates:
                        break
                if len(docs) >= max_candidates:
                    break
        else:
            docs = sorted(
                merged.values(),
                key=lambda item: float(item.get("score", 0.0) or 0.0),
                reverse=True,
            )
        logger.info(
            "Query expansion retrieved %d unique candidates from %d queries",
            len(docs),
            len(queries),
        )
        return docs[:max_candidates]

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
            retrieved_docs = self._retrieve_documents(query_text, current_k)

            if self.reranker:
                logger.info("Applying reranker...")
                rerank_top_k = self.reranker_top_k or current_k
                reranked_docs = self._run_cuda_stage(
                    "reranker",
                    self.reranker.rerank,
                    query_text,
                    retrieved_docs,
                    top_k=rerank_top_k,
                )
                retrieved_docs = self._balance_source_families(
                    query_text,
                    reranked_docs,
                    rerank_top_k,
                )
                logger.info(f"Reranked documents -> top {len(retrieved_docs)} kept")

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
            answer = self._run_cuda_stage(
                "llm_generate",
                self.llm.generate_with_context,
                query_text,
                context_texts,
            )
            answer = self._normalize_generated_answer(answer)

            result = {
                'answer': answer,
                'num_docs_retrieved': len(retrieved_docs),
            }

            answer_quality = self._audit_answer_quality(
                query_text,
                answer,
                context_texts,
                aggregation_mode,
                False,
            )
            rewrite_count = 0
            max_quality_rewrites = int(self.answer_quality_config.get("max_rewrites", 0) or 0)
            if (
                answer_quality
                and not self._model_requested_abstain(answer)
                and self._should_rewrite_for_quality(answer_quality)
                and max_quality_rewrites > 0
            ):
                feedback = build_quality_feedback(
                    answer_quality,
                    max_missing=int(self.answer_quality_config.get("max_feedback_concepts", 5)),
                )
                if feedback:
                    rewrite_count = 1
                    logger.info("Rewriting answer for low completeness/claim support")
                    answer = self._run_cuda_stage(
                        "llm_rewrite",
                        self.llm.generate_with_context,
                        query_text,
                        context_texts,
                        max_tokens=self.answer_quality_config.get("rewrite_max_tokens"),
                        quality_feedback=feedback,
                    )
                    answer = self._normalize_generated_answer(answer)
                    result = {
                        "answer": answer,
                        "num_docs_retrieved": len(retrieved_docs),
                        "answer_quality_rewrite_attempted": True,
                    }
                    answer_quality = self._audit_answer_quality(
                        query_text,
                        answer,
                        context_texts,
                        aggregation_mode,
                        False,
                    )

            detection_updates, detection_result = self._detect_generated_answer(
                answer,
                context_texts,
                aggregation_mode,
                query_text,
                detect_hallucinations,
            )
            result.update(detection_updates)
            if bool((self.answer_quality_config or {}).get("claim_level", False)):
                answer_quality = self._audit_answer_quality(
                    query_text,
                    answer,
                    context_texts,
                    aggregation_mode,
                    detect_hallucinations,
                )

            if answer_quality:
                result["answer_quality"] = answer_quality
                result["answer_quality_rewrite_count"] = rewrite_count
                result["answer_completeness_score"] = answer_quality.get("answer_completeness_score")
                result["answer_completeness_risk"] = answer_quality.get("answer_completeness_risk")
                result["context_coverage"] = answer_quality.get("context_coverage")
                result["answer_quality_missing_context_concepts"] = answer_quality.get("missing_context_concepts")
                result["answer_quality_missing_concepts"] = answer_quality.get("missing_concepts")
                result["claim_include_rate"] = answer_quality.get("claim_include_rate")
                result["claim_unsupported_rate"] = answer_quality.get("claim_unsupported_rate")
                result["claim_answer_include_risk_max"] = answer_quality.get(
                    "claim_answer_include_risk_max"
                )

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
                gating_stats.update(self._quality_stats(answer_quality))

                model_requested_abstain = bool(result.get("model_requested_abstain"))
                if model_requested_abstain:
                    gating_action = "abstain"
                    result['answer'] = abstain_message

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

                decision = "abstain" if model_requested_abstain else self._decide_gating_action(
                    gating_stats, gating_config
                )

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
                    'min_answer_completeness_score': gating_config.get("min_answer_completeness_score"),
                    'min_context_coverage': gating_config.get("min_context_coverage"),
                    'min_claim_include_rate': gating_config.get("min_claim_include_rate"),
                    'max_claim_answer_include_risk': gating_config.get("max_claim_answer_include_risk"),
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

    def verify_candidate_answer(
        self,
        query_text: str,
        candidate_answer: str,
        k: Optional[int] = None,
        return_context: bool = False,
        return_sources: bool = False,
        hallucination_aggregation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify a candidate answer against retrieved evidence without generating.

        This is the detector-first path used for RAG gating and real-life smoke
        tests where an answer already exists and the system must decide whether
        the retrieved corpus supports it.
        """
        if not self.hallucination_detector:
            raise RuntimeError("Hallucination detector is not available.")

        current_k = k if k is not None else self.retriever.k
        aggregation_mode = hallucination_aggregation or self.hallucination_aggregation or "any"

        logger.info(f"Retrieving top {current_k} documents for candidate verification...")
        retrieved_docs = self._retrieve_documents(query_text, current_k)

        if self.reranker and retrieved_docs:
            rerank_top_k = self.reranker_top_k or current_k
            retrieved_docs = self._run_cuda_stage(
                "reranker",
                self.reranker.rerank,
                query_text,
                retrieved_docs,
                top_k=rerank_top_k,
            )

        context_texts = [doc["content"] for doc in retrieved_docs]
        detection_result = self.hallucination_detector.verify_answer_with_contexts(
            answer=candidate_answer,
            contexts=context_texts,
            aggregation=aggregation_mode,
            question=query_text,
        )

        result = {
            "query": query_text,
            "candidate_answer": candidate_answer,
            "num_docs_retrieved": len(retrieved_docs),
            "hallucination_detected": detection_result.get("is_hallucination"),
            "answer_include_detected": detection_result.get("answer_include_detected"),
            "answer_include_risk": detection_result.get(
                "answer_include_risk",
                detection_result.get("unsupported_risk"),
            ),
            "answer_include_score": detection_result.get(
                "answer_include_score",
                detection_result.get("support_score"),
            ),
            # Backward-compatible aliases.
            "unsupported_answer_detected": detection_result.get("is_unsupported"),
            "hallucination_score": detection_result.get("hallucination_score"),
            "unsupported_risk": detection_result.get("unsupported_risk"),
            "support_score": detection_result.get("support_score"),
            "hallucination_details": {
                "aggregation": detection_result.get("aggregation"),
                "num_contexts": detection_result.get("num_contexts"),
                "num_contradictions": detection_result.get("num_contradictions"),
                "num_answer_not_included": detection_result.get("num_answer_not_included"),
                "num_unsupported": detection_result.get("num_unsupported"),
                "answer_include_threshold": detection_result.get("answer_include_threshold"),
                "hard_answer_not_included_rate": detection_result.get("hard_answer_not_included_rate"),
                "answer_include_risk": detection_result.get("answer_include_risk"),
                "answer_include_score": detection_result.get("answer_include_score"),
                "answer_include_prob_mean": detection_result.get("answer_include_prob_mean"),
                "answer_include_prob_topk": detection_result.get("answer_include_prob_topk"),
                # Backward-compatible aliases.
                "unsupported_threshold": detection_result.get("unsupported_threshold"),
                "hard_unsupported_rate": detection_result.get("hard_unsupported_rate"),
                "hard_contradiction_rate": detection_result.get("hard_contradiction_rate"),
                "best_context_index": detection_result.get("best_context_index"),
                "best_context_label": detection_result.get("best_context_label"),
                "best_context_scores": detection_result.get("best_context_scores"),
                "unsupported_prob_mean": detection_result.get("unsupported_prob_mean"),
                "unsupported_prob_topk": detection_result.get("unsupported_prob_topk"),
                "hallucination_prob_mean": detection_result.get("hallucination_prob_mean"),
                "hallucination_prob_topk": detection_result.get("hallucination_prob_topk"),
            },
            "individual_results": detection_result.get("individual_results", []),
        }

        if return_context:
            result["context"] = retrieved_docs
        if return_sources:
            result["sources"] = self._build_source_entries(retrieved_docs)

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

    def _extract_named_source_families(self, query_text: str) -> List[str]:
        named_families: List[str] = []
        for family, pattern in SOURCE_FAMILY_PATTERNS.items():
            if pattern.search(query_text):
                named_families.append(family)
        return named_families

    def _canonicalize_source_family(self, raw_value: Any) -> Optional[str]:
        if raw_value is None:
            return None

        normalized = str(raw_value).strip().lower()
        if not normalized:
            return None

        for family, pattern in SOURCE_FAMILY_PATTERNS.items():
            if pattern.search(normalized):
                return family

        return None

    def _doc_source_family(self, doc: Dict[str, Any]) -> Optional[str]:
        metadata = doc.get("metadata") or {}
        for key in ("family", "source_org", "title", "source", "url", "path"):
            family = self._canonicalize_source_family(metadata.get(key))
            if family:
                return family
        return None

    def _source_balance_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(token) > 2 and token not in SOURCE_BALANCE_STOPWORDS
        }

    def _family_doc_match_score(self, query_text: str, doc: Dict[str, Any]) -> tuple[float, float]:
        metadata = doc.get("metadata") or {}
        query_tokens = self._source_balance_tokens(query_text)
        title_parts = [
            metadata.get("title"),
            metadata.get("section_heading"),
            metadata.get("doc_id"),
        ]
        title_tokens = self._source_balance_tokens(
            " ".join(part for part in title_parts if isinstance(part, str) and part.strip())
        )
        overlap = 0.0
        if query_tokens and title_tokens:
            overlap = float(len(query_tokens & title_tokens) / len(query_tokens))
        return overlap, float(doc.get("score", 0.0) or 0.0)

    def _balance_source_families(
        self,
        query_text: str,
        retrieved_docs: List[Dict[str, Any]],
        target_k: int,
    ) -> List[Dict[str, Any]]:
        min_named_families = int(self.source_config.get("balance_min_named_families", 2))
        if not bool(self.source_config.get("balance_named_families", False)):
            return retrieved_docs[:target_k]

        named_families = self._extract_named_source_families(query_text)
        if len(named_families) < min_named_families:
            return retrieved_docs[:target_k]

        family_candidates: Dict[str, List[Dict[str, Any]]] = {family: [] for family in named_families}
        for doc in retrieved_docs:
            family = self._doc_source_family(doc)
            if family in family_candidates:
                family_candidates[family].append(doc)

        if sum(1 for docs in family_candidates.values() if docs) < min_named_families:
            return retrieved_docs[:target_k]

        balanced_docs: List[Dict[str, Any]] = []
        used_doc_ids = set()

        for family in named_families:
            docs = family_candidates.get(family) or []
            if not docs:
                continue
            doc = max(docs, key=lambda candidate: self._family_doc_match_score(query_text, candidate))
            balanced_docs.append(doc)
            used_doc_ids.add(id(doc))

        for doc in retrieved_docs:
            if len(balanced_docs) >= target_k:
                break
            if id(doc) in used_doc_ids:
                continue
            balanced_docs.append(doc)
            used_doc_ids.add(id(doc))

        if len(balanced_docs) != len(retrieved_docs[:target_k]):
            logger.info(
                "Applied source-family balancing for %s → families=%s",
                query_text[:80],
                ",".join(named_families),
            )

        return balanced_docs[:target_k]

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
        if llm_type in {"none", "disabled", "noop", "detector_only"}:
            llm = _NoopLLM(llm_config)
        elif llm_type in {"openrouter", "api", "openai_compatible"}:
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
                        batch_size=detector_config.get('batch_size', 8),
                        base_model=detector_config.get('base_model'),
                        lora_config=detector_config.get('lora') or detector_config.get('lora_config'),
                        mc_dropout_samples=detector_config.get('mc_dropout_samples', 1),
                        swag_config=detector_config.get('swag'),
                        logit_sampling_config=detector_config.get('logit_sampling'),
                        representation_sampling_config=detector_config.get('representation_sampling'),
                        artifact_verifier_config=detector_config.get('artifact_verifier'),
                        risk_mode=detector_config.get('risk_mode', 'contradiction'),
                        unsupported_threshold=detector_config.get(
                            'unsupported_threshold',
                            detector_config.get('answer_include_threshold', 0.5),
                        ),
                        answer_include_threshold=detector_config.get('answer_include_threshold'),
                        hypothesis_format=detector_config.get('hypothesis_format', 'answer_only'),
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
            retrieval_config=config.get('retrieval', {}),
            runtime_config=config.get('runtime', {}),
            source_config=config.get('sources', {}),
            answer_quality_config=config.get('answer_quality', {}),
            hallucination_aggregation=(
                config.get("hallucination_aggregation")
                or detector_config.get("aggregation")
                or "any"
            ),
        )
