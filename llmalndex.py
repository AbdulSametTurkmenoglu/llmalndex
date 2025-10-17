import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

print("🚀 1. Adım: Temel Kurulum ve Ayarlar Başlatılıyor...")

storage_path = "./storage"

print("🧠 LLM (Claude) ve Embedding Modeli (e5-large) Yükleniyor...")
Settings.llm = Anthropic(
    api_key="x",
    model="claude-sonnet-4-5-20250929",
    temperature=0.3,
    max_tokens=1024
)
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")

print("2. Adım: Veri Yükleme ve Index Oluşturma...")

try:
    if not os.path.exists(storage_path) or not os.listdir(storage_path):
        raise FileNotFoundError("Storage directory is empty or doesn't exist")

    print("✅ Mevcut index bulundu. Diskten yükleniyor...")
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    print("✅ Index başarıyla yüklendi.")

except (FileNotFoundError, Exception) as e:
    print(f"Mevcut index bulunamadı ({str(e)}). Belgelerden yeni bir index oluşturuluyor...")

    if not os.path.exists("data/"):
        print("Hata: 'data/' klasörü bulunamadı. Lütfen belgeleri 'data/' klasörüne yerleştirin.")
        sys.exit(1)

    documents = SimpleDirectoryReader("data/").load_data()
    print(f"✅ Yüklenen belge sayısı: {len(documents)}")

    if len(documents) == 0:
        print("Hata: Hiç belge yüklenemedi. 'data/' klasöründe belge olduğundan emin olun.")
        sys.exit(1)

    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    os.makedirs(storage_path, exist_ok=True)
    storage_context.persist(persist_dir=storage_path)
    print("✅ Yeni index oluşturuldu ve diske kaydedildi.")

print("3. Adım: Hızlandırılmış Sorgulama Motoru Oluşturuluyor...")
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)
print(" Sorgulama motoru hazır!")

print("4. Adım: Sorgu Gönderiliyor ve Yanıt Bekleniyor...")
query = "Verim, Suç ve Ceza kitabına göre suç türleri nelerdir?"
print(f"❓ Sorgu: {query}")

response = query_engine.query(query)

print("\n" + "=" * 50)
print("🔹 YANIT")
print("=" * 50)
print(response)
print("\nİşlem tamamlandı.")