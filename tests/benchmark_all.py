import sys
import os
import time
import pandas as pd
from typing import List, Dict

# Path settings
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from src.reranking.base_reranker import RerankerFactory
import src.reranking.rerankers
from src.core.logger import get_logger

logger = get_logger(__name__)

def run_benchmark() -> None:
    print("=== Reranker Benchmark Starting ===\n")

    # 1. Test Verisi
    # Zorlu bir senaryo: Hem İngilizce hem Türkçe, hem de çeldirici (distractor) içeren dökümanlar.
    query = "What causes inflation?"
    
    documents = [
        # Tam Alakalı (Relevant)
        {"content": "Inflation is caused by an increase in the money supply or a decrease in the demand for money.", "metadata": {"id": "doc_1", "type": "relevant"}},
        {"content": "Enflasyon, dolaşımdaki para arzının artması veya mal ve hizmetlere olan talebin arzı aşması sonucu oluşur.", "metadata": {"id": "doc_2", "type": "relevant_tr"}},
        
        # Yarı Alakalı / Konuyla İlgili ama Cevap Değil (Related)
        {"content": "The central bank decided to raise interest rates to combat the rising cost of living.", "metadata": {"id": "doc_3", "type": "related"}},
        {"content": "Economic growth often slows down during periods of high deflation.", "metadata": {"id": "doc_4", "type": "related_antonym"}},
        
        # Alakasız / Çeldirici (Irrelevant)
        {"content": "To inflate a balloon, you need to blow air into it forcefully.", "metadata": {"id": "doc_5", "type": "irrelevant_lexical_overlap"}}, # 'inflate' kelimesi geçiyor ama konu farklı
        {"content": "The football match ended with a score of 2-1.", "metadata": {"id": "doc_6", "type": "irrelevant"}}
    ]

    # 2. Modeller ve Konfigürasyonları
    models_to_test = {
        "BM25": {
            "type": "bm25",
            "config": {} # BM25 genellikle parametre istemez
        },
        "Cross-Encoder (Standard)": {
            "type": "cross_encoder",
            "config": {
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", # Endüstri standardı küçük model
                "device": "cpu"
            }
        },
        "mGTE (Alibaba)": {
            "type": "mgte",
            "config": {
                "model_name_or_path": "Alibaba-NLP/gte-multilingual-reranker-base",
                "device": "cpu",
                "batch_size": 4
            }
        }
    }

    results = []

    # 3. Benchmark Döngüsü
    for model_display_name, model_info in models_to_test.items():
        print(f"\n>> Test Ediliyor: {model_display_name}...")
        
        try:
            # Yükleme Süresi Ölçümü
            load_start = time.time()
            reranker = RerankerFactory.create(model_info["type"], model_info["config"])
            load_time = time.time() - load_start
            
            # Kopyasını alalım ki orijinal liste bozulmasın
            docs_copy = [d.copy() for d in documents]
            
            # Rerank Süresi Ölçümü
            rerank_start = time.time()
            ranked_docs = reranker.rerank(query, docs_copy)
            rerank_time = time.time() - rerank_start
            
            # En iyi dökümanı al
            top_doc = ranked_docs[0]
            
            results.append({
                "Model": model_display_name,
                "Top 1 Doc ID": top_doc["metadata"]["id"],
                "Top 1 Score": f"{top_doc['score']:.4f}",
                "Load Time (s)": f"{load_time:.2f}",
                "Inference Time (s)": f"{rerank_time:.4f}"
            })
            
            # Detaylı sıralamayı konsola yaz
            print(f"   Süre: {rerank_time:.4f}s")
            print("   Sıralama:")
            for i, doc in enumerate(ranked_docs, 1):
                print(f"     {i}. [{doc['score']:.4f}] {doc['content'][:60]}... ({doc['metadata']['type']})")

        except Exception as e:
            print(f"   HATA: {str(e)}")

    # 4. Özet Tablo
    print("\n\n=== ÖZET KARŞILAŞTIRMA TABLOSU ===")
    df = pd.DataFrame(results)
    # Tabloyu güzelleştirme
    print(df.to_markdown(index=False) if hasattr(df, 'to_markdown') else df)

if __name__ == "__main__":
    run_benchmark()