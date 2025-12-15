import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from src.core.base_classes import BaseReranker
from src.reranking.base_reranker import register_reranker
from src.core.logger import get_logger
from src.embeddings.base_embedder import EmbedderFactory

logger = get_logger(__name__)




@register_reranker("ebcar")
class EBCARReranker(BaseReranker):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})

        # EBCAR model yüklemesi...
        model_path = self.config.get("checkpoint")
        hidden = self.config.get("hidden_size", 768)
        self.device = self.config.get("device", "cpu")

        # embedder (mGTE embedding için)
        embedder_name = self.config.get("embedder_name", "mgte")
        self.embedder = EmbedderFactory.create(embedder_name)

        if model_path is None:
            raise ValueError("EBCAR checkpoint path is required")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"EBCAR checkpoint not found: {model_path}")

        # Model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        # Embed query
        q_emb = torch.tensor(self.embedder.embed_text(query), dtype=torch.float32).to(self.device)

        # Prepare passage embeddings
        p_embs = []
        doc_ids = []
        positions = []

        for doc in documents:
            emb = self.embedder.embed_text(doc["content"])
            p_embs.append(torch.tensor(emb, dtype=torch.float32))
            doc_ids.append(doc.get("doc_id", 0))
            positions.append(doc.get("position", 0))

        p_embs = torch.stack(p_embs).unsqueeze(0).to(self.device)
        doc_ids = torch.tensor([doc_ids], dtype=torch.long).to(self.device)
        positions = torch.tensor([positions], dtype=torch.long).to(self.device)

        # Forward
        out = self.model(q_emb.unsqueeze(0), p_embs, doc_ids, positions)

        # compute scores
        if out.dim() != 2 or q_emb.dim() != 1:
            raise ValueError(
                f"Unexpected tensor shapes: out={out.shape}, q_emb={q_emb.shape}"
            )

        if out.size(1) != q_emb.size(0):
            raise ValueError(
                f"Embedding dimension mismatch: {out.size(1)} != {q_emb.size(0)}"
            )
        scores = torch.matmul(out, q_emb.unsqueeze(1)).squeeze(1)

        # attach
        for doc, score in zip(documents, scores.tolist()):
            doc["score"] = float(score)

        # sort
        documents = sorted(documents, key=lambda d: d["score"], reverse=True)

        # trim
        if top_k:
            documents = documents[:top_k]

        return documents