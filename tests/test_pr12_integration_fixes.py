from src.rag.answer_quality import audit_answer_quality, infer_required_concepts
from src.rag.hallucination_detector import HallucinationDetector
from src.rag.rag_pipeline import RAGPipeline
from scripts.run_finreg_real_life_benchmark import full_rag_expected_behavior_rubric


class FakeDetector:
    def verify_answer_with_contexts(self, answer, contexts, aggregation="any", question=None):
        return {
            "is_hallucination": False,
            "answer_include_detected": True,
            "answer_include_risk": 0.1,
            "answer_include_score": 0.9,
            "is_unsupported": False,
            "unsupported_risk": 0.1,
            "support_score": 0.9,
            "hallucination_score": 0.0,
            "individual_results": [],
            "aggregation": aggregation,
            "num_contexts": len(contexts),
            "num_contradictions": 0,
            "num_answer_not_included": 0,
            "num_unsupported": 0,
        }


class FakeRetriever:
    k = 3

    def retrieve(self, query, k):
        raise AssertionError("verify_candidate_answer must use _retrieve_documents")


def test_verify_candidate_answer_uses_pipeline_retrieval_wrapper():
    pipeline = RAGPipeline(
        data_manager=None,
        chunker=None,
        embedder=None,
        vector_store=None,
        llm=None,
        retriever=FakeRetriever(),
        hallucination_detector=FakeDetector(),
    )

    calls = []

    def fake_retrieve(query_text, k):
        calls.append((query_text, k))
        return [{"content": "supporting context", "score": 0.8, "metadata": {}}]

    pipeline._retrieve_documents = fake_retrieve

    result = pipeline.verify_candidate_answer("What is required?", "A supported answer.", k=5)

    assert calls == [("What is required?", 5)]
    assert result["answer_include_detected"] is True
    assert result["num_docs_retrieved"] == 1


def test_no_context_detector_result_has_pipeline_expected_fields():
    detector = object.__new__(HallucinationDetector)
    detector.answer_include_threshold = 0.97
    detector.unsupported_threshold = 0.97

    result = detector.verify_answer_with_contexts(
        answer="Candidate answer",
        contexts=[],
        aggregation="answer_include_best",
    )

    assert result["is_unsupported"] is True
    assert result["answer_include_risk"] == 1.0
    assert result["num_contexts"] == 0
    assert result["num_contradictions"] == 0
    assert result["aggregation"] == "answer_include_best"


def test_threshold_question_does_not_force_missing_evidence_concepts():
    concepts = infer_required_concepts(
        "What threshold applies to material outsourcing?",
        ["The context states that material outsourcing requires board approval."],
    )

    assert "no exact requirement" not in concepts
    assert "avoid overclaiming" not in concepts
    assert "evidence limits" not in concepts


def test_unestablished_threshold_question_keeps_missing_evidence_concepts():
    concepts = infer_required_concepts(
        "The evidence does not establish an exact threshold; what should we say?",
        ["The context discusses outsourcing governance but does not specify a threshold."],
    )

    assert "no exact requirement" in concepts
    assert "avoid overclaiming" in concepts
    assert "evidence limits" in concepts


def test_ict_quality_accepts_common_lifecycle_word_forms():
    answer = (
        "ICT controls should include identifying risks, protecting systems, "
        "monitoring to detect issues, incident and problem management to respond, "
        "business continuity and backups to recover, and assessed or tested controls."
    )

    audit = audit_answer_quality(
        "What controls are expected for ICT and security risk management?",
        answer,
        ["ICT security risk management includes protection, detection, response, recovery, and testing."],
        {"max_required_concepts": 8},
    )

    assert "identify risks" in audit["hit_concepts"]
    assert "protect" in audit["hit_concepts"]
    assert "detect" in audit["hit_concepts"]
    assert "respond" in audit["hit_concepts"]
    assert "recover" in audit["hit_concepts"]
    assert "vulnerability and testing" in audit["hit_concepts"]


def test_operational_resilience_quality_uses_impact_tolerance_concepts():
    concepts = infer_required_concepts(
        "What are impact tolerances in operational resilience supervision?",
        ["Firms set impact tolerances for important business services and maximum tolerable disruption."],
        max_concepts=8,
    )

    assert "important business services" in concepts
    assert "maximum tolerable disruption" in concepts
    assert "impact tolerance" in concepts
    assert "operational resilience" in concepts


def test_governance_remuneration_quality_uses_expected_concepts():
    concepts = infer_required_concepts(
        "How do risk culture, staff training, board oversight, and remuneration alignment work together in governance?",
        ["Boards oversee risk culture, training, remuneration alignment, and the institution's risk profile."],
        max_concepts=8,
    )

    assert "risk culture" in concepts
    assert "staff training" in concepts
    assert "board oversight" in concepts
    assert "risk profile" in concepts
    assert "remuneration" in concepts


def test_answer_quality_reports_missing_context_concepts():
    audit = audit_answer_quality(
        "When can manual workarounds be tolerated, and what controls should surround them?",
        "Manual workarounds may be temporary and should have controls and ownership.",
        [
            "Manual workarounds should be temporary during transition, with controls, "
            "ownership, auditability, remediation, and data lineage."
        ],
        {"max_required_concepts": 8},
    )

    assert "temporary manual controls" in audit["hit_concepts"]
    assert "auditability" in audit["missing_context_concepts"]
    assert "remediation" in audit["missing_context_concepts"]
    assert "data lineage" in audit["missing_context_concepts"]
    assert audit["context_coverage"] < 1.0


def test_answer_quality_accepts_config_forced_concept_rules():
    audit = audit_answer_quality(
        "What should an implementation roadmap contain?",
        "The roadmap should define owners and milestones.",
        ["The implementation roadmap should include owners, milestones, dependencies, and residual risks."],
        {
            "max_required_concepts": 4,
            "concept_patterns": {
                "implementation dependencies": [r"\bdependencies\b"],
                "residual risks": [r"\bresidual risks?\b"],
            },
            "forced_concept_rules": [
                {
                    "when_any": ["implementation roadmap"],
                    "concepts": ["implementation dependencies", "residual risks"],
                }
            ],
        },
    )

    assert "implementation dependencies" in audit["required_concepts"]
    assert "residual risks" in audit["required_concepts"]
    assert "implementation dependencies" in audit["missing_context_concepts"]
    assert "residual risks" in audit["missing_context_concepts"]


def test_gating_can_abstain_on_low_context_coverage_after_detector_passes():
    pipeline = RAGPipeline(
        data_manager=None,
        chunker=None,
        embedder=None,
        vector_store=None,
        llm=None,
        retriever=FakeRetriever(),
        hallucination_detector=FakeDetector(),
    )

    action = pipeline._decide_gating_action(
        {
            "hard_contradiction_rate": 0.0,
            "answer_include_risk": 0.1,
            "uncertainty_mean": 0.0,
            "retrieval_max_score": 1.0,
            "retrieval_mean_score": 1.0,
            "answer_completeness_score": 0.9,
            "context_coverage": 0.25,
        },
        {
            "strategy": "abstain",
            "contradiction_rate_metric": "hard_contradiction_rate",
            "contradiction_rate_threshold": 0.4,
            "contradiction_prob_metric": "answer_include_risk",
            "contradiction_prob_threshold": 0.963,
            "uncertainty_threshold": 1.0,
            "min_context_coverage": 0.4,
        },
    )

    assert action == "abstain"


def test_quality_does_not_expand_context_when_query_concepts_are_sufficient():
    concepts = infer_required_concepts(
        "How do BCBS and ECB-style RDARR expectations connect governance, architecture, controls, and report scope?",
        [
            "RDARR was reviewed in the SREP cycle. Supervisors also discuss ownership, "
            "capital adequacy, monitoring, governance, architecture, and controls."
        ],
        max_concepts=8,
    )

    assert "risk data aggregation" in concepts
    assert "governance" in concepts
    assert "architecture" in concepts
    assert "controls" in concepts
    assert "capital adequacy" not in concepts
    assert "SREP" not in concepts
    assert "ownership" not in concepts


def test_false_premise_question_does_not_require_unasked_context_concepts():
    audit = audit_answer_quality(
        "Is intraday liquidity monitoring mainly used to rank retail branch customer satisfaction?",
        (
            "Intraday liquidity monitoring is used to assess liquidity risk and is not "
            "related to ranking retail branch customer satisfaction."
        ),
        [
            "Supervisors use intraday liquidity tools for monitoring cash flows, "
            "payment and settlement behaviour, controls, architecture, and stress testing."
        ],
        {"max_required_concepts": 8},
    )

    assert "intraday liquidity" in audit["required_concepts"]
    assert "payment and settlement behaviour" not in audit["required_concepts"]
    assert "architecture" not in audit["required_concepts"]
    assert "controls" not in audit["required_concepts"]


def test_operational_resilience_allows_equivalent_disruption_language():
    audit = audit_answer_quality(
        "What are impact tolerances in operational resilience supervision?",
        (
            "Impact tolerances define the acceptable level of disruption to important "
            "business services during operational disruptions."
        ),
        ["Firms set impact tolerances for important business services and maximum tolerable disruption."],
        {"max_required_concepts": 8},
    )

    assert "maximum tolerable disruption" in audit["hit_concepts"]


def test_full_rag_rubric_accepts_not_used_false_premise_refutation():
    rubric = full_rag_expected_behavior_rubric(
        {
            "expected_behavior": "refute_or_abstain",
            "answer": (
                "Intraday liquidity monitoring is used to assess liquidity risk. "
                "It is not used to rank retail branch customer satisfaction."
            ),
            "expected_answer_points": ["no", "liquidity"],
            "forbidden_claims": ["rank retail branch customer satisfaction"],
            "abstained": False,
        }
    )

    assert rubric["expected_behavior_match"] is True
    assert rubric["forbidden_claim_hits"] == []
