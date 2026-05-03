import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import build_vector_database, generate_answer

# =========================
# ⚙️ Config
# =========================
PDF_PATH = "data/documents/Policy.pdf"   # change if needed
CHUNK_SIZE = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_TYPE = "faiss"
LLM_MODEL = "llama3"   # or gemma

# =========================
# 🔄 Build System
# =========================
print("🔄 Building RAG system...")

vector_db, bm25 = build_vector_database(
    pdf_path=PDF_PATH,
    chunk_size=CHUNK_SIZE,
    embedding_model=EMBEDDING_MODEL,
    db_type=DB_TYPE
)

print("✅ System Ready!")

# =========================
# 💬 Query Loop
# =========================
while True:
    query = input("\nAsk a question (or 'exit'): ")

    if query.lower() == "exit":
        break

    answer, docs = generate_answer(
        query,
        vector_db,
        bm25,
        llm_model=LLM_MODEL
    )

    print("\n💡 Answer:\n", answer)

    print("\n📄 Retrieved Context:")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}:\n{doc.page_content[:200]}...")