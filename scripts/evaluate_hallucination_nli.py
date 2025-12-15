import json
import argparse
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ----------------------------
# Utility
# ----------------------------

def split_claims(text: str) -> List[str]:
    """
    Basit claim splitter.
    (İleride daha sofistike yapılabilir.)
    """
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in text.split(".") if len(s.strip()) > 5]


def batch(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


# ----------------------------
# NLI Evaluator
# ----------------------------

class HallucinationEvaluator:
    def __init__(
        self,
        model_path: str,
        device: str = None,
        batch_size: int = 16,
        alpha_nei: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.alpha_nei = alpha_nei

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def nli_scores(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> List[Dict[str, float]]:

        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        results = []
        for row in probs:
            scores = {
                self.id2label[i]: row[i].item()
                for i in range(row.size(0))
            }
            results.append(scores)

        return results

    def hallucination_risk(self, scores: Dict[str, float]) -> float:
    
        return (
            scores.get("contradiction", 0.0)
            + self.alpha_nei * scores.get("neutral", 0.0)
        )


# ----------------------------
# Main evaluation logic
# ----------------------------

def evaluate_example(example: Dict, evaluator: HallucinationEvaluator):

    claims = split_claims(example["answer"])
    evidence_chunks = example["evidence_chunks"]

    all_results = []

    for claim in claims:
        pairs = [
            (chunk, claim)
            for chunk in evidence_chunks
        ]

        claim_results = []

        for chunk_batch in batch(pairs, evaluator.batch_size):
            premises = [p for p, _ in chunk_batch]
            hypotheses = [h for _, h in chunk_batch]

            scores = evaluator.nli_scores(premises, hypotheses)

            for premise, score in zip(premises, scores):
                risk = evaluator.hallucination_risk(score)
                claim_results.append({
                    "premise": premise,
                    "scores": score,
                    "risk": risk,
                })

        # Aggregation: en kötü evidence = claim riski
        claim_risk = max(r["risk"] for r in claim_results)

        all_results.append({
            "claim": claim,
            "claim_risk": claim_risk,
            "evidence_results": claim_results,
        })

    # Final answer risk = en riskli claim
    final_risk = max(c["claim_risk"] for c in all_results) if all_results else 0.0

    return {
        "id": example.get("id"),
        "question": example.get("question"),
        "final_hallucination_risk": final_risk,
        "claims": all_results,
    }


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Fine-tuned DeBERTa checkpoint or output dir")
    parser.add_argument("--input", required=True, help="JSONL input file")
    parser.add_argument("--output", required=True, help="JSONL output file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha-nei", type=float, default=0.5)

    args = parser.parse_args()

    evaluator = HallucinationEvaluator(
        model_path=args.model_path,
        batch_size=args.batch_size,
        alpha_nei=args.alpha_nei,
    )

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Evaluating"):
            example = json.loads(line)
            result = evaluate_example(example, evaluator)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()