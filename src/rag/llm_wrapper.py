"""
LLM wrapper for HuggingFace models.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.core.base_classes import BaseLLM
from src.core.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceLLM(BaseLLM):
    """
    LLM wrapper for HuggingFace causal language models
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = self.config.get('model_name', 'meta-llama/Llama-2-7b-chat-hf')
        self.device = self.config.get('device', 'cpu')
        self.cache_folder = self.config.get('cache_folder', None)
        self.max_new_tokens = self.config.get('max_new_tokens', 512)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.9)
        self.load_in_8bit = self.config.get('load_in_8bit', False)

        logger.info(f"Loading LLM: {self.model_name} on {self.device}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_folder
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            model_kwargs = {
                'cache_dir': self.cache_folder,
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
            }

            if self.load_in_8bit and self.device == 'cuda':
                model_kwargs['load_in_8bit'] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            if not self.load_in_8bit:
                self.model = self.model.to(self.device)

            self.model.eval()  # Set to evaluation mode

            logger.info(f"Successfully loaded LLM: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load LLM {self.model_name}: {e}")
            raise

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
        temperature = temperature or self.temperature

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Remove prompt from output (model echoes the prompt)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text with retrieved context

        Args:
            query: User query
            context: Retrieved context documents
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        # Format context
        context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])

        # Create prompt
        prompt = f"""Use the following context documents to answer the question. If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context_text}

Question: {query}

Answer:"""

        return self.generate(prompt, max_tokens=max_tokens)
