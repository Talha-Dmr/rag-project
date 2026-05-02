"""
Hallucination detector for RAG pipeline.

Uses fine-tuned DeBERTa model to detect hallucinations in generated answers
by performing NLI (Natural Language Inference) between context and answer.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import os
import math
import re

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

    CANONICAL_LABEL_ALIASES = {
        "entailment": "entailment",
        "supports": "entailment",
        "supported": "entailment",
        "support": "entailment",
        "neutral": "neutral",
        "not_enough_info": "neutral",
        "not enough info": "neutral",
        "nei": "neutral",
        "unknown": "neutral",
        "contradiction": "contradiction",
        "refutes": "contradiction",
        "refute": "contradiction",
        "conflict": "contradiction",
    }

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 8,
        base_model: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        mc_dropout_samples: int = 1,
        swag_config: Optional[Dict[str, Any]] = None,
        logit_sampling_config: Optional[Dict[str, Any]] = None,
        representation_sampling_config: Optional[Dict[str, Any]] = None,
        artifact_verifier_config: Optional[Dict[str, Any]] = None
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
        self.mc_dropout_samples = max(1, int(mc_dropout_samples))
        self.swag_config = swag_config or {}
        self.swag_enabled = bool(self.swag_config.get("enabled", False))
        self.swag_state = None
        self.swag_param_names = []
        self.swag_param_cache = {}
        self.swag_base_params = {}
        self.logit_sampling_config = logit_sampling_config or {}
        self.logit_sampling_enabled = bool(self.logit_sampling_config.get("enabled", False))
        self.representation_sampling_config = representation_sampling_config or {}
        self.representation_sampling_enabled = bool(
            self.representation_sampling_config.get("enabled", False)
        )
        self.artifact_verifier_config = artifact_verifier_config or {}
        self.artifact_verifier_enabled = bool(
            self.artifact_verifier_config.get("enabled", False)
        )
        self.artifact_verifier_confidence_threshold = float(
            self.artifact_verifier_config.get("confidence_threshold", 0.42)
        )
        self._classifier_module = None
        self.local_files_only = (
            os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
        )

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model and tokenizer
        logger.info(f"Loading hallucination detector from: {model_path}")
        self._load_model()
        self._configure_label_mapping()
        if self.representation_sampling_enabled:
            try:
                self._classifier_module = self._resolve_classifier_module()
            except Exception as e:
                logger.warning(
                    "Representation sampling requested but classifier head was not resolved: %s",
                    e
                )
                self.representation_sampling_enabled = False
        if self.swag_enabled:
            self._setup_swag()

        logger.info(f"Hallucination detector initialized on {self.device}")
        if self.logit_sampling_enabled:
            logger.info("Logit sampling enabled: %s", self.logit_sampling_config)
        if self.representation_sampling_enabled:
            logger.info("Representation sampling enabled: %s", self.representation_sampling_config)
        if self.artifact_verifier_enabled:
            logger.info("Artifact verifier enabled: %s", self.artifact_verifier_config)

    def _artifact_verifier_result(
        self,
        premise: str,
        hypothesis: str,
        *,
        query: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Return a high-confidence lexical verifier result, or None."""
        if not self.artifact_verifier_enabled:
            return None

        try:
            from src.rag.artifact_verifier import predict_support_status, to_detector_result
        except Exception as exc:
            logger.warning("Artifact verifier import failed; falling back to neural detector: %s", exc)
            return None

        prediction = predict_support_status(
            query=query,
            candidate_answer=hypothesis,
            evidence_span=premise,
        )
        result = to_detector_result(prediction)
        result["artifact_verifier_confidence_threshold"] = (
            self.artifact_verifier_confidence_threshold
        )
        if result["confidence"] < self.artifact_verifier_confidence_threshold:
            return None
        return result

    def _resolve_classifier_module(self) -> torch.nn.Module:
        """Find classifier head across plain HF and PEFT wrapped models."""
        candidates = []
        if hasattr(self.model, "classifier"):
            candidates.append(getattr(self.model, "classifier"))
        if hasattr(self.model, "base_model"):
            base_model = getattr(self.model, "base_model")
            if hasattr(base_model, "classifier"):
                candidates.append(getattr(base_model, "classifier"))
            if hasattr(base_model, "model") and hasattr(base_model.model, "classifier"):
                candidates.append(getattr(base_model.model, "classifier"))

        for mod in candidates:
            if isinstance(mod, torch.nn.Module):
                return mod

        for name, mod in self.model.named_modules():
            if name.endswith("classifier"):
                return mod

        raise RuntimeError("Could not find classifier head for representation sampling.")

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        model_pt = self.model_path / "model.pt"
        model_dir = self.model_path / "model"
        tokenizer_dir = self.model_path / "tokenizer"

        # Case 0: exported PEFT adapter bundle. Prefer the adapter directory over
        # root pytorch_model.bin when both exist; some exports keep a PEFT state
        # dict at the root that is not directly loadable as a plain HF model.
        if (model_dir / "adapter_config.json").exists():
            base_model = self.base_model
            if not base_model:
                adapter_config_path = model_dir / "adapter_config.json"
                try:
                    import json

                    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                    base_model = adapter_config.get("base_model_name_or_path")
                except Exception:
                    base_model = None
            if not base_model:
                raise ValueError(
                    "base_model is required to load an exported PEFT adapter bundle"
                )

            try:
                from peft import PeftModel
                from src.training.utils.model_utils import load_model_and_tokenizer
            except Exception as exc:
                raise ImportError(
                    "peft and training utilities are required to load PEFT adapter bundles."
                ) from exc

            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=base_model,
                num_labels=3,
                cache_dir=None,
                device=self.device,
                lora_config=None,
                local_files_only=self.local_files_only
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                str(model_dir),
                local_files_only=self.local_files_only,
            )
            self.model.eval()
            self.model = self.model.to(self.device)
            logger.info("Loaded model from exported PEFT adapter bundle")
            return

        # Check if paths exist
        if (self.model_path / "pytorch_model.bin").exists() or (
            self.model_path / "model.safetensors"
        ).exists():
            # Prefer a full HuggingFace export at the root over a nested LoRA
            # adapter directory. Some exported bundles keep both.
            model_dir = self.model_path
        elif not model_dir.exists():
            # Try loading directly from model_path (HF export)
            model_dir = self.model_path

        if not tokenizer_dir.exists():
            tokenizer_dir = self.model_path

        # Case 1: training checkpoint (model.pt)
        if model_pt.exists():
            base_model = self.base_model
            lora_config = self.lora_config
            checkpoint_state = None
            checkpoint_weights = torch.load(model_pt, map_location=self.device, weights_only=False)

            # Try to infer base model from training_state if available
            state_path = self.model_path / "training_state.pt"
            if state_path.exists():
                try:
                    checkpoint_state = torch.load(state_path, map_location="cpu", weights_only=False)
                    cfg = checkpoint_state.get("config", {})
                    base_model = base_model or cfg.get("model", {}).get("base_model")
                    lora_config = lora_config or cfg.get("model", {}).get("lora")
                except Exception:
                    pass

            if not lora_config:
                lora_config = self._infer_lora_config_from_checkpoint(checkpoint_weights)

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
                lora_config=lora_config,
                local_files_only=self.local_files_only
            )
            missing, unexpected = self.model.load_state_dict(checkpoint_weights, strict=False)
            if missing or unexpected:
                logger.warning(
                    "Checkpoint load (strict=False). Missing keys: %s | Unexpected keys: %s",
                    missing,
                    unexpected
                )
            self.model.eval()

            logger.info("Loaded model from training checkpoint")
            return

        # Case 2: HF export
        try:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_dir),
                    local_files_only=self.local_files_only
                )
            except Exception as tokenizer_error:
                if not self.base_model:
                    raise
                logger.warning(
                    "Local tokenizer load failed from %s (%s). Falling back to base_model tokenizer: %s",
                    tokenizer_dir,
                    tokenizer_error,
                    self.base_model,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model,
                    local_files_only=self.local_files_only
                )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                local_files_only=self.local_files_only
            )

            self.model.eval()
            self.model = self.model.to(self.device)

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _infer_lora_config_from_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, Any]]:
        """Infer a minimal LoRA config from a training checkpoint state_dict."""
        lora_a_keys = [key for key in state_dict.keys() if ".lora_A." in key]
        if not lora_a_keys:
            return None

        target_modules = set()
        modules_to_save = set()
        inferred_r: Optional[int] = None

        for key in lora_a_keys:
            match = re.search(r"\.([^.]+)\.lora_A\.[^.]+\.weight$", key)
            if match:
                target_modules.add(match.group(1))
            tensor = state_dict.get(key)
            if tensor is not None and getattr(tensor, "shape", None):
                try:
                    inferred_r = int(tensor.shape[0])
                except Exception:
                    pass

        for key in state_dict.keys():
            match = re.search(r"\.([^.]+)\.modules_to_save\.[^.]+\.", key)
            if match:
                modules_to_save.add(match.group(1))

        inferred = {
            "enabled": True,
            "r": inferred_r or 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "SEQ_CLS",
            "target_modules": sorted(target_modules) or ["query_proj", "key_proj", "value_proj"],
        }
        if modules_to_save:
            inferred["modules_to_save"] = sorted(modules_to_save)

        logger.info("Inferred LoRA config from checkpoint: %s", inferred)
        return inferred

    @classmethod
    def _canonicalize_label_name(cls, value: object) -> str:
        text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
        text = " ".join(text.split())
        return cls.CANONICAL_LABEL_ALIASES.get(text, "")

    def _configure_label_mapping(self) -> None:
        """
        Infer model label ids from config.id2label when possible.

        This keeps the detector compatible with both:
        - entailment / neutral / contradiction exports
        - FEVER-style SUPPORTS / REFUTES / NOT ENOUGH INFO exports
        """
        default_names = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }
        self.LABEL_NAMES = dict(default_names)
        self.LABEL_ENTAILMENT = 0
        self.LABEL_NEUTRAL = 1
        self.LABEL_CONTRADICTION = 2

        config = getattr(self.model, "config", None)
        id2label = getattr(config, "id2label", None)
        if not isinstance(id2label, dict):
            return

        canonical_to_id: Dict[str, int] = {}
        pretty_names: Dict[int, str] = {}
        for raw_id, raw_name in id2label.items():
            try:
                label_id = int(raw_id)
            except Exception:
                continue
            canonical = self._canonicalize_label_name(raw_name)
            if not canonical:
                continue
            canonical_to_id[canonical] = label_id
            pretty_names[label_id] = canonical

        required = {"entailment", "neutral", "contradiction"}
        if not required.issubset(set(canonical_to_id.keys())):
            logger.warning(
                "Could not fully infer detector labels from id2label=%s; using default label ids.",
                id2label,
            )
            return

        self.LABEL_ENTAILMENT = int(canonical_to_id["entailment"])
        self.LABEL_NEUTRAL = int(canonical_to_id["neutral"])
        self.LABEL_CONTRADICTION = int(canonical_to_id["contradiction"])
        self.LABEL_NAMES = {
            idx: pretty_names.get(idx, default_names.get(idx, str(idx)))
            for idx in range(max(pretty_names.keys()) + 1)
        }
        logger.info(
            "Configured detector label mapping: entailment=%s neutral=%s contradiction=%s from id2label=%s",
            self.LABEL_ENTAILMENT,
            self.LABEL_NEUTRAL,
            self.LABEL_CONTRADICTION,
            id2label,
        )

    def _setup_swag(self) -> None:
        swag_path = self.swag_config.get("path")
        if not swag_path:
            raise ValueError("swag.path is required when swag is enabled")

        self.swag_state = torch.load(swag_path, map_location="cpu")
        param_names = self.swag_state.get("param_names")
        if not param_names:
            param_names = list(self.swag_state.get("mean", {}).keys())
        self.swag_param_names = list(param_names or [])

        # Cache LoRA parameters by name
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.swag_param_names:
                continue
            self.swag_param_cache[name] = param
            self.swag_base_params[name] = param.detach().clone()

        if not self.swag_param_cache:
            logger.warning("SWAG enabled but no matching LoRA parameters found.")

    def _apply_swag_sample(self) -> None:
        if not self.swag_state or not self.swag_param_cache:
            return

        mean = self.swag_state.get("mean", {})
        sq_mean = self.swag_state.get("sq_mean", {})

        for name, param in self.swag_param_cache.items():
            mu = mean.get(name)
            second = sq_mean.get(name)
            if mu is None or second is None:
                continue
            mu = mu.to(param.device)
            second = second.to(param.device)
            var = torch.clamp(second - mu.pow(2), min=1e-12)
            std = torch.sqrt(var)
            sample = mu + torch.randn_like(std) * std
            param.data.copy_(sample)

    def _restore_swag_base(self) -> None:
        if not self.swag_base_params:
            return
        for name, param in self.swag_param_cache.items():
            base = self.swag_base_params.get(name)
            if base is None:
                continue
            param.data.copy_(base)

    def _predict_probs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.swag_enabled:
            if not self.swag_state or not self.swag_param_cache:
                logger.warning("SWAG enabled but not initialized; falling back to single pass.")
            else:
                num_samples = max(1, int(self.swag_config.get("num_samples", 5)))
                probs_sum = None
                probs_sq_sum = None
                with torch.no_grad():
                    for _ in range(num_samples):
                        self._apply_swag_sample()
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        probs_sum = probs if probs_sum is None else probs_sum + probs
                        probs_sq = probs.pow(2)
                        probs_sq_sum = probs_sq if probs_sq_sum is None else probs_sq_sum + probs_sq
                self._restore_swag_base()
                mean = probs_sum / float(num_samples)
                var = probs_sq_sum / float(num_samples) - mean.pow(2)
                return mean, var

        if self.mc_dropout_samples <= 1:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                return torch.softmax(logits, dim=-1)

        was_training = self.model.training
        self.model.train()  # enable dropout
        probs_sum = None
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

        if not was_training:
            self.model.eval()

        return probs_sum / float(self.mc_dropout_samples)

    def _logit_sampling_metrics(self, base_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.logit_sampling_config
        steps = int(cfg.get("steps", 30))
        step_size = float(cfg.get("step_size", 0.03))
        noise_std = float(cfg.get("noise_std", 0.1))
        prior_strength = float(cfg.get("prior_strength", 1.0))
        entropy_weight = float(cfg.get("entropy_weight", 0.5))
        energy = cfg.get("energy", "hybrid")

        base_logits = base_logits.detach()
        base_probs = torch.softmax(base_logits, dim=-1)
        logits = base_logits.clone()
        samples = []

        for _ in range(steps):
            logits = logits.detach().requires_grad_(True)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

            if energy == "entropy":
                energy_val = -entropy
            elif energy == "prior_l2":
                energy_val = 0.5 * (logits - base_logits).pow(2).mean()
            elif energy == "prior_kl":
                energy_val = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    base_probs,
                    reduction="batchmean"
                )
            else:
                prior = 0.5 * (logits - base_logits).pow(2).mean()
                energy_val = prior_strength * prior - entropy_weight * entropy

            grad = torch.autograd.grad(energy_val, logits)[0]
            logits = logits - step_size * grad + torch.randn_like(logits) * noise_std
            samples.append(torch.softmax(logits, dim=-1).detach())

        stacked = torch.stack(samples, dim=0)
        mean_probs = stacked.mean(dim=0)
        ent = -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum(dim=-1)
        mean_ent = -(stacked * stacked.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=0)
        mi = ent - mean_ent
        var = stacked.var(dim=0).mean(dim=-1)
        return {"mi": mi, "variance": var, "entropy": ent}

    def _representation_sampling_metrics(
        self,
        base_logits: torch.Tensor,
        base_representation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        cfg = self.representation_sampling_config
        steps = int(cfg.get("steps", 30))
        step_size = float(cfg.get("step_size", 0.02))
        noise_std = float(cfg.get("noise_std", 0.05))
        prior_strength = float(cfg.get("prior_strength", 1.0))
        entropy_weight = float(cfg.get("entropy_weight", 0.5))
        energy = cfg.get("energy", "hybrid")

        if self._classifier_module is None:
            raise RuntimeError("Classifier module unavailable for representation sampling.")

        base_logits = base_logits.detach()
        base_probs = torch.softmax(base_logits, dim=-1)
        base_z = base_representation.detach()
        z = base_z.clone()
        samples = []

        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            logits = self._classifier_module(z)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

            if energy == "entropy":
                energy_val = -entropy
            elif energy == "prior_l2":
                energy_val = 0.5 * (z - base_z).pow(2).mean()
            elif energy == "prior_kl":
                energy_val = F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    base_probs,
                    reduction="batchmean"
                )
            else:
                prior = 0.5 * (z - base_z).pow(2).mean()
                energy_val = prior_strength * prior - entropy_weight * entropy

            grad = torch.autograd.grad(energy_val, z)[0]
            z = z - step_size * grad + torch.randn_like(z) * noise_std
            samples.append(torch.softmax(self._classifier_module(z), dim=-1).detach())

        stacked = torch.stack(samples, dim=0)
        mean_probs = stacked.mean(dim=0)
        ent = -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum(dim=-1)
        mean_ent = -(stacked * stacked.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=0)
        mi = ent - mean_ent
        var = stacked.var(dim=0).mean(dim=-1)
        return {"mi": mi, "variance": var, "entropy": ent}

    def _forward_with_classifier_input(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> (torch.Tensor, torch.Tensor):
        if self._classifier_module is None:
            raise RuntimeError("Classifier module unavailable for representation sampling.")

        rep_cache: Dict[str, torch.Tensor] = {}

        def capture(
            _module: torch.nn.Module,
            module_input: tuple,
            _module_output: torch.Tensor
        ) -> None:
            if module_input:
                rep_cache["z"] = module_input[0].detach()

        hook_handle = self._classifier_module.register_forward_hook(capture)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                base_logits = outputs.logits
        finally:
            hook_handle.remove()

        base_z = rep_cache.get("z")
        if base_z is None:
            raise RuntimeError("Could not capture classifier input representation.")

        return base_logits, base_z

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
        artifact_result = self._artifact_verifier_result(premise, hypothesis)
        if artifact_result is not None:
            if not return_scores:
                artifact_result.pop("scores", None)
            return artifact_result

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

        if self.logit_sampling_enabled or self.representation_sampling_enabled:
            rep_metrics = None
            if self.representation_sampling_enabled:
                base_logits, base_z = self._forward_with_classifier_input(inputs)
            else:
                with torch.no_grad():
                    base_logits = self.model(**inputs).logits
                base_z = None
            with torch.no_grad():
                probs = torch.softmax(base_logits, dim=-1)
            with torch.enable_grad():
                logit_metrics = (
                    self._logit_sampling_metrics(base_logits)
                    if self.logit_sampling_enabled else None
                )
                if self.representation_sampling_enabled and base_z is not None:
                    rep_metrics = self._representation_sampling_metrics(base_logits, base_z)
            var = None
        else:
            logit_metrics = None
            rep_metrics = None
            pred = self._predict_probs(inputs)
            if isinstance(pred, tuple):
                probs, var = pred
            else:
                probs, var = pred, None

        probs = probs[0]
        var_row = var[0] if var is not None else None

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
            # Normalized entropy as uncertainty in [0,1]
            probs_safe = torch.clamp(probs, min=1e-12)
            entropy = -(probs_safe * torch.log(probs_safe)).sum().item()
            norm_entropy = entropy / float(torch.log(torch.tensor(len(self.LABEL_NAMES))).item())
            result['uncertainty_entropy'] = float(norm_entropy)
            if var_row is not None:
                result['uncertainty_variance'] = float(var_row.mean().item())
                result['contradiction_variance'] = float(
                    var_row[self.LABEL_CONTRADICTION].item()
                )
            if logit_metrics is not None:
                mi_val = float(logit_metrics["mi"][0].item())
                var_val = float(logit_metrics["variance"][0].item())
                ent_val = float(logit_metrics["entropy"][0].item())
                norm_ent = ent_val / float(torch.log(torch.tensor(len(self.LABEL_NAMES))).item())
                result['uncertainty_logit_mi'] = mi_val
                result['uncertainty_logit_variance'] = var_val
                result['uncertainty_logit_entropy'] = norm_ent
            if rep_metrics is not None:
                mi_val = float(rep_metrics["mi"][0].item())
                var_val = float(rep_metrics["variance"][0].item())
                ent_val = float(rep_metrics["entropy"][0].item())
                norm_ent = ent_val / float(torch.log(torch.tensor(len(self.LABEL_NAMES))).item())
                result['uncertainty_rep_mi'] = mi_val
                result['uncertainty_rep_variance'] = var_val
                result['uncertainty_rep_entropy'] = norm_ent

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

        artifact_results: List[Optional[Dict[str, Any]]] = []
        if self.artifact_verifier_enabled:
            for premise, hypothesis in zip(premises, hypotheses):
                result = self._artifact_verifier_result(premise, hypothesis)
                if result is not None and not return_scores:
                    result.pop("scores", None)
                artifact_results.append(result)
            if all(result is not None for result in artifact_results):
                return [result for result in artifact_results if result is not None]

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

            if self.logit_sampling_enabled or self.representation_sampling_enabled:
                rep_metrics = None
                if self.representation_sampling_enabled:
                    base_logits, base_z = self._forward_with_classifier_input(inputs)
                else:
                    with torch.no_grad():
                        base_logits = self.model(**inputs).logits
                    base_z = None
                with torch.no_grad():
                    probs = torch.softmax(base_logits, dim=-1)
                with torch.enable_grad():
                    logit_metrics = (
                        self._logit_sampling_metrics(base_logits)
                        if self.logit_sampling_enabled else None
                    )
                    if self.representation_sampling_enabled and base_z is not None:
                        rep_metrics = self._representation_sampling_metrics(base_logits, base_z)
                var = None
            else:
                logit_metrics = None
                rep_metrics = None
                pred = self._predict_probs(inputs)
                if isinstance(pred, tuple):
                    probs, var = pred
                else:
                    probs, var = pred, None

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
                    probs_safe = torch.clamp(probs[j], min=1e-12)
                    entropy = -(probs_safe * torch.log(probs_safe)).sum().item()
                    norm_entropy = entropy / float(torch.log(torch.tensor(len(self.LABEL_NAMES))).item())
                    result['uncertainty_entropy'] = float(norm_entropy)
                    if var is not None:
                        var_row = var[j]
                        result['uncertainty_variance'] = float(var_row.mean().item())
                        result['contradiction_variance'] = float(
                            var_row[self.LABEL_CONTRADICTION].item()
                        )
                    if logit_metrics is not None:
                        mi_val = float(logit_metrics["mi"][j].item())
                        var_val = float(logit_metrics["variance"][j].item())
                        ent_val = float(logit_metrics["entropy"][j].item())
                        norm_ent = ent_val / float(torch.log(torch.tensor(len(self.LABEL_NAMES))).item())
                        result['uncertainty_logit_mi'] = mi_val
                        result['uncertainty_logit_variance'] = var_val
                        result['uncertainty_logit_entropy'] = norm_ent
                    if rep_metrics is not None:
                        mi_val = float(rep_metrics["mi"][j].item())
                        var_val = float(rep_metrics["variance"][j].item())
                        ent_val = float(rep_metrics["entropy"][j].item())
                        norm_ent = ent_val / float(torch.log(torch.tensor(len(self.LABEL_NAMES))).item())
                        result['uncertainty_rep_mi'] = mi_val
                        result['uncertainty_rep_variance'] = var_val
                        result['uncertainty_rep_entropy'] = norm_ent

                all_results.append(result)

        if artifact_results:
            for index, artifact_result in enumerate(artifact_results):
                if artifact_result is not None:
                    all_results[index] = artifact_result

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
        hard_contradiction_rate = num_contradictions / len(contexts)
        contradiction_probs = [
            float((r.get("scores") or {}).get("contradiction", 0.0))
            for r in individual_results
        ]
        entailment_probs = [
            float((r.get("scores") or {}).get("entailment", 0.0))
            for r in individual_results
        ]
        neutral_probs = [
            float((r.get("scores") or {}).get("neutral", 0.0))
            for r in individual_results
        ]

        if contradiction_probs:
            hallucination_prob_mean = float(sum(contradiction_probs) / len(contradiction_probs))
            top_k = max(1, math.ceil(len(contradiction_probs) / 2))
            top_probs = sorted(contradiction_probs, reverse=True)[:top_k]
            hallucination_prob_topk = float(sum(top_probs) / len(top_probs))
            contradiction_margin_mean = float(sum(
                max(0.0, cp - max(ep, np_))
                for cp, ep, np_ in zip(contradiction_probs, entailment_probs, neutral_probs)
            ) / len(contradiction_probs))
            contradiction_neutral_gap_mean = float(sum(
                cp - np_ for cp, np_ in zip(contradiction_probs, neutral_probs)
            ) / len(contradiction_probs))
        else:
            hallucination_prob_mean = 0.0
            hallucination_prob_topk = 0.0
            contradiction_margin_mean = 0.0
            contradiction_neutral_gap_mean = 0.0

        # Aggregate decision / score
        agg = (aggregation or "any").strip().lower()
        if agg == 'any':
            hallucination_score = hard_contradiction_rate
            is_hallucination = (num_contradictions > 0)
        elif agg == 'majority':
            hallucination_score = hard_contradiction_rate
            is_hallucination = (num_contradictions > len(contexts) / 2)
        elif agg == 'all':
            hallucination_score = hard_contradiction_rate
            is_hallucination = (num_contradictions == len(contexts))
        elif agg == 'prob_mean':
            hallucination_score = hallucination_prob_mean
            is_hallucination = (hallucination_score >= 0.40)
        elif agg == 'prob_topk':
            hallucination_score = hallucination_prob_topk
            is_hallucination = (hallucination_score >= 0.40)
        elif agg == 'margin_mean':
            hallucination_score = contradiction_margin_mean
            is_hallucination = (hallucination_score >= 0.08)
        elif agg == 'prob_mean_plus_margin':
            hallucination_score = float(min(
                1.0,
                0.75 * hallucination_prob_mean + 0.25 * min(1.0, contradiction_margin_mean * 4.0)
            ))
            is_hallucination = (hallucination_score >= 0.40)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return {
            'is_hallucination': is_hallucination,
            'individual_results': individual_results,
            'hallucination_score': hallucination_score,
            'hard_contradiction_rate': hard_contradiction_rate,
            'hallucination_prob_mean': hallucination_prob_mean,
            'hallucination_prob_topk': hallucination_prob_topk,
            'contradiction_margin_mean': contradiction_margin_mean,
            'contradiction_neutral_gap_mean': contradiction_neutral_gap_mean,
            'num_contexts': len(contexts),
            'num_contradictions': num_contradictions,
            'aggregation': aggregation
        }
