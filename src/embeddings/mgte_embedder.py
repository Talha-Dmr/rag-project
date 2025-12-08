import torch
from transformers import AutoTokenizer, AutoModel
from src.embeddings.base_embedder import BaseEmbedder


class MGTEEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "llmrails/mgte-large", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.dim = self.model.config.hidden_size

    def embed_text(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens)
        emb = out.last_hidden_state[:, 0, :]  # CLS embedding
        return emb.squeeze().cpu().tolist()

    def embed_batch(self, texts):
        out_list = []
        for t in texts:
            out_list.append(self.embed_text(t))
        return out_list

    def get_dimension(self):
        return self.dim
