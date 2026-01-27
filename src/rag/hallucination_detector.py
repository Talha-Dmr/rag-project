"""
Hallucination detector for RAG pipeline.

Uses fine-tuned DeBERTa model to detect hallucinations in generated answers
by performing NLI (Natural Language Inference) between context and answer.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Inference wrapper for hallucination detection model.

    Detects potential hallucinations in RAG-generated answers by checking
    if the answer is supported by the retrieved context.
    """

    # Label mapping
    LABEL_ENTAILMENT = 0
    LABEL_NEUTRAL = 1
    LABEL_CONTRADICTION = 2

    LABEL_NAMES = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    }

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 8,
        base_model: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hallucination detector.

        Args:
            model_path: Path to exported model directory
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for batch predictions
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.base_model = base_model
        self.lora_config = lora_config

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model and tokenizer
        logger.info(f"Loading hallucination detector from: {model_path}")
        self._load_model()

        logger.info(f"Hallucination detector initialized on {self.device}")

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        model_pt = self.model_path / "model.pt"
        model_dir = self.model_path / "model"
        tokenizer_dir = self.model_path / "tokenizer"

        # Check if paths exist
        if not model_dir.exists():
            # Try loading directly from model_path (HF export)
            model_dir = self.model_path

        if not tokenizer_dir.exists():
            tokenizer_dir = self.model_path

        # Case 1: training checkpoint (model.pt)
        if model_pt.exists():
            base_model = self.base_model
            lora_config = self.lora_config

            # Try to infer base model from training_state if available
            state_path = self.model_path / "training_state.pt"
            if state_path.exists():
                try:
                    state = torch.load(state_path, map_location="cpu")
                    cfg = state.get("config", {})
                    base_model = base_model or cfg.get("model", {}).get("base_model")
                    lora_config = lora_config or cfg.get("model", {}).get("lora")
                except Exception:
                    pass

            if not base_model:
                raise ValueError(
                    "base_model is required to load a checkpoint with model.pt"
                )

            try:
                from src.training.utils.model_utils import load_model_and_tokenizer
            except Exception as exc:
                raise ImportError(
                    "Failed to import training utilities needed for checkpoint loading."
                ) from exc

            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=base_model,
                num_labels=3,
                cache_dir=None,
                device=self.device,
                lora_config=lora_config
            )
            self.model.load_state_dict(torch.load(model_pt, map_location=self.device))
            self.model.eval()

            logger.info("Loaded model from training checkpoint")
            return

        # Case 2: HF export
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

            self.model.eval()
            self.model = self.model.to(self.device)

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(
        self,
        premise: str,
        hypothesis: str,
        return_scores: bool = True
    ) -> Dict[str, Any]:
        """
        Detect hallucination for a single premise-hypothesis pair.

        Args:
            premise: Context or retrieved document text
            hypothesis: Generated answer or claim to verify
            return_scores: Whether to return probability scores

        Returns:
            Dictionary containing:
                - is_hallucination: bool (True if contradiction detected)
                - label: str (entailment/neutral/contradiction)
                - confidence: float (confidence in prediction)
                - scores: Dict[str, float] (optional, probabilities per label)
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Get prediction
        predicted_class = torch.argmax(probs).item()
        label = self.LABEL_NAMES[predicted_class]
        confidence = probs[predicted_class].item()

        # Check if hallucination (contradiction)
        is_hallucination = (predicted_class == self.LABEL_CONTRADICTION)

        result = {
            'is_hallucination': is_hallucination,
            'label': label,
            'confidence': confidence
        }

        if return_scores:
            result['scores'] = {
                'entailment': probs[self.LABEL_ENTAILMENT].item(),
                'neutral': probs[self.LABEL_NEUTRAL].item(),
                'contradiction': probs[self.LABEL_CONTRADICTION].item()
            }

        return result

    def detect_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect hallucinations for multiple premise-hypothesis pairs.

        Args:
            premises: List of context texts
            hypotheses: List of generated answers/claims
            return_scores: Whether to return probability scores

        Returns:
            List of detection results (same format as detect())
        """
        if len(premises) != len(hypotheses):
            raise ValueError("Number of premises must match number of hypotheses")

        # Process in batches
        all_results = []

        for i in range(0, len(premises), self.batch_size):
            batch_premises = premises[i:i + self.batch_size]
            batch_hypotheses = hypotheses[i:i + self.batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Process results
            for j in range(len(batch_premises)):
                predicted_class = torch.argmax(probs[j]).item()
                label = self.LABEL_NAMES[predicted_class]
                confidence = probs[j][predicted_class].item()

                is_hallucination = (predicted_class == self.LABEL_CONTRADICTION)

                result = {
                    'is_hallucination': is_hallucination,
                    'label': label,
                    'confidence': confidence
                }

                if return_scores:
                    result['scores'] = {
                        'entailment': probs[j][self.LABEL_ENTAILMENT].item(),
                        'neutral': probs[j][self.LABEL_NEUTRAL].item(),
                        'contradiction': probs[j][self.LABEL_CONTRADICTION].item()
                    }

                all_results.append(result)

        return all_results

    def verify_answer_with_contexts(
        self,
        answer: str,
        contexts: List[str],
        aggregation: str = 'any'
    ) -> Dict[str, Any]:
        """
        Verify if answer is supported by retrieved contexts.

        Args:
            answer: Generated answer to verify
            contexts: List of retrieved context documents
            aggregation: How to aggregate results across contexts
                        'any': Hallucination if ANY context contradicts
                        'majority': Hallucination if MAJORITY contradict
                        'all': Hallucination if ALL contexts contradict

        Returns:
            Dictionary containing:
                - is_hallucination: bool (aggregated decision)
                - individual_results: List[Dict] (per-context results)
                - hallucination_score: float (fraction of contradictions)
        """
        if not contexts:
            logger.warning("No contexts provided for verification")
            return {
                'is_hallucination': False,
                'individual_results': [],
                'hallucination_score': 0.0,
                'note': 'No contexts available'
            }

        # Check each context
        premises = contexts
        hypotheses = [answer] * len(contexts)

        individual_results = self.detect_batch(premises, hypotheses)

        # Count contradictions
        num_contradictions = sum(
            1 for r in individual_results if r['is_hallucination']
        )
        hallucination_score = num_contradictions / len(contexts)

        # Aggregate decision
        if aggregation == 'any':
            is_hallucination = (num_contradictions > 0)
        elif aggregation == 'majority':
            is_hallucination = (num_contradictions > len(contexts) / 2)
        elif aggregation == 'all':
            is_hallucination = (num_contradictions == len(contexts))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return {
            'is_hallucination': is_hallucination,
            'individual_results': individual_results,
            'hallucination_score': hallucination_score,
            'num_contexts': len(contexts),
            'num_contradictions': num_contradictions,
            'aggregation': aggregation
        }
