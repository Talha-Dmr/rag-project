import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.core.base_classes import BaseReranker
from src.reranking.base_reranker import register_reranker
from src.core.logger import get_logger

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
        
        try:
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
        except Exception as e:
            logger.error(f"Failed to load mGTE model: {e}")
            raise

    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance using mGTE.

        Args:
            query: Query text
            documents: List of documents with 'content' field
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        # Validate documents: filter out empty content
        valid_docs = [doc for doc in documents if doc.get('content', '').strip()]
        if not valid_docs:
            logger.warning("No valid documents found with content to rerank.")
            return []

        logger.info(f"Reranking {len(valid_docs)} documents for query: {query[:50]}...")
        
        try:
            # Prepare pairs: [CLS] query [SEP] document
            doc_texts = [doc['content'] for doc in valid_docs]
            pairs = [[query, doc_text] for doc_text in doc_texts]
            
            all_scores = []
            
            # Dynamic batching optimization
            effective_batch_size = min(self.batch_size, len(pairs))
            
            with torch.no_grad():
                for i in range(0, len(pairs), effective_batch_size):
                    batch_pairs = pairs[i : i + effective_batch_size]
                    
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

            # Update scores in the valid documents
            for doc, score in zip(valid_docs, all_scores):
                doc["score"] = score
                
            # Sort documents by score in descending order
            valid_docs.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply top_k slicing if requested
            result = valid_docs[:top_k] if top_k else valid_docs
            
            return result

        except Exception as e:
            logger.error(f"Error during mGTE reranking: {e}")
            # Fallback: return original documents sliced by top_k
            return documents[:top_k] if top_k else documents