"""
LLM wrapper for HuggingFace models.
"""

from typing import List, Dict, Any, Optional
import re
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as TRANSFORMERS_VERSION
from src.core.base_classes import BaseLLM
from src.core.logger import get_logger

logger = get_logger(__name__)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class HuggingFaceLLM(BaseLLM):
    """
    LLM wrapper for HuggingFace causal language models
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.device = self.config.get('device', 'cpu')
        self.cache_folder = self.config.get('cache_folder', None)
        self.model_name = self._resolve_local_model(
            self.config.get('model_name', 'meta-llama/Llama-2-7b-chat-hf')
        )
        # Backward-compatible alias:
        # many configs use `max_tokens`; prefer explicit `max_new_tokens` when present.
        self.max_new_tokens = self.config.get(
            'max_new_tokens',
            self.config.get('max_tokens', 512)
        )
        self.max_prompt_tokens = int(self.config.get('max_prompt_tokens', 2048))
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.9)
        self.load_in_8bit = self.config.get('load_in_8bit', False)
        self.use_chat_template = bool(self.config.get('use_chat_template', True))
        self.truncation_side = str(self.config.get("truncation_side", "left")).strip().lower()
        self.enable_thinking = self.config.get('enable_thinking')
        self.device_map = self.config.get('device_map')
        self.low_cpu_mem_usage = bool(self.config.get('low_cpu_mem_usage', False))
        self.attn_implementation = self.config.get('attn_implementation')
        self.use_cache = bool(self.config.get('use_cache', True))
        self.repetition_penalty = float(self.config.get('repetition_penalty', 1.0) or 1.0)
        self.no_repeat_ngram_size = int(self.config.get('no_repeat_ngram_size', 0) or 0)
        self.trust_remote_code = bool(self.config.get("trust_remote_code", False))
        self.clear_cuda_cache_after_generate = bool(
            self.config.get('clear_cuda_cache_after_generate', self.device == 'cuda')
        )
        self.dtype = self._resolve_dtype(self.config.get('dtype'))
        self.local_files_only = bool(
            self.config.get("local_files_only", _env_truthy("HF_HUB_OFFLINE"))
        )

        logger.info(f"Loading LLM: {self.model_name} on {self.device}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_folder,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.truncation_side in {"left", "right"}:
                self.tokenizer.truncation_side = self.truncation_side

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **self._model_kwargs()
            )

            if not self.load_in_8bit and not self.device_map:
                self.model = self.model.to(self.device)

            self.model.eval()  # Set to evaluation mode

            logger.info(f"Successfully loaded LLM: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load LLM {self.model_name}: {e}")
            raise

    def _model_kwargs(self) -> Dict[str, Any]:
        dtype_key = "dtype" if str(TRANSFORMERS_VERSION).split(".", 1)[0] == "5" else "torch_dtype"
        model_kwargs = {
            'cache_dir': self.cache_folder,
            'local_files_only': self.local_files_only,
            'trust_remote_code': self.trust_remote_code,
            dtype_key: self.dtype,
        }
        if self.device_map:
            model_kwargs['device_map'] = self.device_map
        if self.low_cpu_mem_usage:
            model_kwargs['low_cpu_mem_usage'] = True
        if self.attn_implementation:
            model_kwargs['attn_implementation'] = self.attn_implementation

        if self.load_in_8bit and self.device == 'cuda':
            model_kwargs['load_in_8bit'] = True
            model_kwargs.setdefault('device_map', 'auto')
        return model_kwargs

    def _resolve_local_model(self, model_name: str) -> str:
        """Resolve a Hugging Face repo id to a bundled local snapshot when present."""
        raw_name = str(model_name)
        model_path = Path(raw_name)
        if model_path.exists():
            return str(model_path)

        if "/" not in raw_name:
            return raw_name

        cache_root = Path(self.cache_folder or "./models/llm")
        snapshots_dir = cache_root / f"models--{raw_name.replace('/', '--')}" / "snapshots"
        if not snapshots_dir.is_dir():
            return raw_name

        snapshots = sorted(
            (
                path for path in snapshots_dir.iterdir()
                if path.is_dir()
                and (path / "config.json").exists()
                and self._snapshot_has_weights(path)
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not snapshots:
            return raw_name

        resolved = str(snapshots[0])
        logger.info("Resolved LLM %s to local snapshot: %s", raw_name, resolved)
        return resolved

    def _snapshot_has_weights(self, snapshot_dir: Path) -> bool:
        """Return true only when a local HF snapshot has complete weight files."""
        if any(path.exists() for path in snapshot_dir.glob("*.bin")):
            return True
        if any(path.exists() for path in snapshot_dir.glob("*.safetensors")):
            return True

        index_path = snapshot_dir / "model.safetensors.index.json"
        if not index_path.exists():
            return False
        try:
            import json

            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = {
                str(filename)
                for filename in (index.get("weight_map") or {}).values()
                if filename
            }
        except Exception:
            return False
        return bool(weight_files) and all((snapshot_dir / filename).exists() for filename in weight_files)

    def _resolve_dtype(self, raw_dtype: Any) -> Any:
        if raw_dtype is None:
            return torch.float16 if self.device == 'cuda' else torch.float32

        if raw_dtype is torch.float16 or raw_dtype is torch.float32 or raw_dtype is torch.bfloat16:
            return raw_dtype

        name = str(raw_dtype).strip().lower()
        if name == "auto":
            return "auto"
        if name in {"float16", "fp16", "half"}:
            return torch.float16
        if name in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if name in {"float32", "fp32"}:
            return torch.float32

        raise ValueError(f"Unsupported dtype: {raw_dtype}")

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
        max_tokens = max_tokens or self.max_new_tokens
        temperature = self.temperature if temperature is None else temperature

        try:
            # Tokenize input (respect model context length)
            max_positions = getattr(self.model.config, "n_positions", None)
            if max_positions is None:
                max_positions = getattr(self.model.config, "max_position_embeddings", 2048)

            # Reserve space for generation
            max_prompt_length = int(max_positions) - int(max_tokens)
            if max_prompt_length <= 0:
                max_prompt_length = int(max_positions) // 2

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(self.max_prompt_tokens, max_prompt_length)
            ).to(self.device)

            # Generate
            with torch.inference_mode():
                generate_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": self.use_cache,
                    # Mitigates occasional invalid probs on some GPU/dtype/model combos.
                    "remove_invalid_values": True,
                    "renormalize_logits": True,
                }
                if self.repetition_penalty and self.repetition_penalty != 1.0:
                    generate_kwargs["repetition_penalty"] = self.repetition_penalty
                if self.no_repeat_ngram_size > 0:
                    generate_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
                if temperature > 0:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = self.top_p

                try:
                    outputs = self.model.generate(**inputs, **generate_kwargs)
                except RuntimeError as rte:
                    message = str(rte).lower()
                    if "probability tensor contains" not in message:
                        raise
                    logger.warning(
                        "Sampling failed with invalid probabilities; retrying with greedy decoding."
                    )
                    fallback_kwargs = dict(generate_kwargs)
                    fallback_kwargs["do_sample"] = False
                    fallback_kwargs.pop("temperature", None)
                    fallback_kwargs.pop("top_p", None)
                    outputs = self.model.generate(**inputs, **fallback_kwargs)

            try:
                input_length = inputs["input_ids"].shape[-1]
                generated_tokens = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                if not generated_text:
                    generated_text = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
            finally:
                del inputs
                if "outputs" in locals():
                    del outputs
                if "generated_tokens" in locals():
                    del generated_tokens
                if self.clear_cuda_cache_after_generate and self.device == 'cuda':
                    torch.cuda.empty_cache()

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        quality_feedback: Optional[str] = None
    ) -> str:
        """
        Generate text with retrieved context

        Args:
            query: User query
            context: Retrieved context documents
            max_tokens: Maximum tokens to generate
            temperature: Optional sampling temperature override
            quality_feedback: Optional rewrite or completeness instruction

        Returns:
            Generated response
        """
        context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])
        system_message = (
            "You are a careful financial-regulation RAG assistant. "
            "Use only the provided context. If the answer is not in the context, "
            "say: \"I don't know based on the provided context.\" "
            "If the question asks for a specific rule, deadline, threshold, portal, "
            "template, or approval requirement, state it only when the context says it explicitly. "
            "If the question says the evidence does not establish a specific number, deadline, "
            "approval, portal, or threshold, explicitly state that the specific item is not "
            "established and do not replace it with a broad regulatory discussion. "
            "If the evidence is incomplete or mixed, say what is supported and what is not established. "
            "For broad what/how/main-elements questions, cover all distinct supported elements "
            "that are relevant to the question instead of giving only one narrow example. "
            "When the question asks for areas, controls, capabilities, responsibilities, or how several "
            "items fit together, explicitly address each named item from the question when the context supports it. "
            "For risk-management lifecycle questions, cover supported identify/assess or measure, "
            "manage or mitigate, monitor or report, and governance elements. "
            "For ICT or security risk questions, cover supported identify, protect, detect, respond, "
            "recover, testing, governance, and incident-management elements. "
            "For management-body suitability questions, cover supported knowledge, skills, experience, "
            "reputation, time commitment, independence, and conflicts-of-interest dimensions. "
            "For climate or ESG questions, cover supported governance, risk identification, measurement, "
            "management, monitoring, physical risk, and transition risk elements. "
            "For false-premise yes/no questions, start by clearly rejecting the premise when "
            "the context supports rejection. Do not output role labels. Answer in 2-5 concise sentences."
        )
        if quality_feedback:
            system_message = f"{system_message} {quality_feedback.strip()}"
        user_message = (
            f"QUESTION: {query}\n\n"
            "CONTEXT:\n<<<\n"
            f"{context_text}\n"
            ">>>\n\n"
            f"QUESTION AGAIN: {query}\n\n"
            "Answer the question using only the context above."
        )

        prompt = None
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if self.use_chat_template and chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = bool(self.enable_thinking)
            try:
                prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        if prompt is None:
            prompt = (
                f"{system_message}\n\n"
                f"{user_message}\n\n"
                "ANSWER:"
            )

        answer = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        answer = self._strip_role_leaks(answer)
        answer = self._strip_unrelated_tail(answer)

        # Post-process to drop prompt-injection or extra roles
        for marker in [
            "\n\nDocument ",
            "\nQUESTION:",
            "\nCONTEXT:",
            "\nYou are an AI assistant",
            "\nHuman:",
            "\nAssistant:",
            "\nUser:",
            "\nSystem:",
            "\nDeveloper:",
            "\nYou're an AI language model",
            "\nYou are Claude",
            "Human:",
            "Assistant:",
            "QUESTION:",
            "CONTEXT:",
            "You're an AI language model",
            "You are Claude",
        ]:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()

        return answer

    def _strip_role_leaks(self, answer: str) -> str:
        if not answer:
            return answer

        role_patterns = [
            r"\bSystem:\s",
            r"\bUser:\s",
            r"\bAssistant:\s",
            r"\bHuman:\s",
            r"\bDeveloper:\s",
            r"\bRole:\s",
            r"\bInstruction:\s",
            r"\bPrompt:\s",
            r"\bHuman Resources Manager:\s",
            r"You're an AI language model",
            r"You are Claude",
            r"\bClaude\.",
        ]

        earliest = None
        for pattern in role_patterns:
            match = re.search(pattern, answer)
            if match:
                pos = match.start()
                if earliest is None or pos < earliest:
                    earliest = pos

        if earliest is not None and earliest > 0:
            return answer[:earliest].strip()

        return answer.strip()

    def _strip_unrelated_tail(self, answer: str) -> str:
        if not answer:
            return answer

        cleaned = re.sub(
            r"^\s*Based on the (?:given|provided) context,\s*I will (?:generate|provide) a response:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*following using the above context:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        tail_patterns = [
            r"\bQuestion:\s",
            r"\bANSWER:\s",
            r"\bTo address this prompt,?\b",
            r"\bTo address the prompt,?\b",
            r"\bHuman-computer interaction research\b",
            r"\bPlease provide your suggestions\b",
            r"\bWrite a Python code snippet\b",
            r"\bDiscuss potential ethical considerations\b",
            r"\n\s*1\.\s+Write a Python code snippet",
            r"\n\s*2\.\s+Explain what natural language processing algorithms",
            r"\n\s*3\.\s+Discuss potential ethical considerations",
            r"\bHuman Resources:\s",
            r"\bHuman resources department\b",
            r"\bHuman resources policies\b",
            r"\bHuman resources department:\s",
            r"\bHuman Resources Department:\s",
            r"\bPremier [A-Z][a-z]+ [A-Z][a-z]+\b",
            r"\bWhat actions might\b",
            r"\bTo which company would you apply\b",
            r"\bPlease send your resume\b",
            r"\bJob Title:\s",
            r"\bdefy gravity\b",
        ]

        earliest = None
        for pattern in tail_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                pos = match.start()
                if pos > 0 and (earliest is None or pos < earliest):
                    earliest = pos

        if earliest is not None:
            return cleaned[:earliest].strip()

        return cleaned.strip()
