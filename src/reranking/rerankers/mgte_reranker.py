import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.core.base_classes import BaseReranker
from src.core.logger import get_logger
from src.reranking.base_reranker import register_reranker

logger = get_logger(__name__)

@register_reranker("mgte")
class MGTEReranker(BaseReranker):
    """
    Implementation of mGTE Reranker compatible with BaseReranker interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        
        # Default model from the paper
        self.model_name = self.config.get("model_name_or_path", "Alibaba-NLP/gte-multilingual-reranker-base")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = self.config.get("max_length", 8192)
        self.batch_size = self.config.get("batch_size", 4)
        
        logger.info(f"Loading mGTE Reranker: {self.model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of document dictionaries.
        Expected format for documents: [{'content': 'text...', 'metadata': ...}, ...]
        """
        if not documents:
            return []

        # Extract text content specifically, handling cases where 'content' might be missing
        doc_texts = [doc.get("content", "") for doc in documents]
        
        # Filter out empty documents to avoid errors, but keep track of indices if needed
        # For simplicity, we assume valid documents come from the retriever.
        
        logger.info(f"Reranking {len(documents)} documents for query: {query[:50]}...")
        
        # Prepare pairs: [CLS] query [SEP] document
        pairs = [[query, doc_text] for doc_text in doc_texts]
        
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs, return_dict=True)
                
                # Model produces a single logit for the score
                scores = outputs.logits.view(-1).float()
                all_scores.extend(scores.cpu().numpy().tolist())

        # Update scores in the original document dictionaries
        for doc, score in zip(documents, all_scores):
            doc["score"] = score
            
        # Sort documents by score in descending order
        documents.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply top_k slicing if requested
        if top_k is not None:
            documents = documents[:top_k]
            
        return documents