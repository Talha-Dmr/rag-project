import sys
import os

# Dosyanın bulunduğu yer: rag-project/tests/test_mgte.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Bir üst dizine (rag-project klasörüne) çıkıyoruz
project_root = os.path.abspath(os.path.join(current_dir, "../")) 
sys.path.append(project_root)

from src.reranking.base_reranker import RerankerFactory
import src.reranking.rerankers  # Register işlemini tetiklemek için import şart
from src.core.logger import get_logger

# Basit bir loglama görelim
logger = get_logger(__name__)

def test_mgte_integration():
    print("=== mGTE Reranker Test Başlıyor ===")
    
    # 1. Konfigürasyon
    config = {
        "model_name_or_path": "Alibaba-NLP/gte-multilingual-reranker-base",
        "device": "cpu",  # Hızlı test için CPU yeterli, varsa 'cuda' yapabilirsin
        "batch_size": 2
    }
    
    try:
        # 2. Factory üzerinden modeli oluşturma
        print("Model yükleniyor (biraz zaman alabilir)...")
        reranker = RerankerFactory.create("mgte", config)
        print("Model başarıyla yüklendi!")
        
        # 3. Test Verisi (BaseReranker formatına uygun: Dict listesi)
        query = "What is the capital of Turkey?"
        documents = [
            {"content": "Paris is the capital of France.", "metadata": {"id": 1}},
            {"content": "Ankara is the capital of Turkey.", "metadata": {"id": 2}},
            {"content": "Istanbul is the largest city in Turkey.", "metadata": {"id": 3}},
            {"content": "Berlin is the capital of Germany.", "metadata": {"id": 4}}
        ]
        
        # 4. Rerank işlemi
        print(f"\nSorgu: {query}")
        print("Dökümanlar skorlanıyor...")
        
        results = reranker.rerank(query, documents, top_k=2)
        
        # 5. Sonuçları Yazdır
        print("\n=== Sonuçlar (Top 2) ===")
        for rank, doc in enumerate(results, 1):
            print(f"{rank}. Skor: {doc['score']:.4f} | İçerik: {doc['content']}")
            
            # Basit bir assertion (Ankara'nın en üstte olmasını bekliyoruz)
            if rank == 1 and "Ankara" not in doc['content']:
                print("UYARI: Beklenen döküman ilk sırada değil!")
                
    except Exception as e:
        print(f"\nHATA OLUŞTU: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mgte_integration()