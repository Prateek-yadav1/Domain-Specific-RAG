# =========================
# 📦 Imports
# =========================
from loader import load_pdf
from chunking import split_documents
from bm25_retriever import BM25Retriever
from embeddings import get_embeddings
from vectorstore import create_vector_store
from llm import get_llm


# =========================
# 🔧 Build Pipeline
# =========================
def build_vector_database(pdf_path, chunk_size, embedding_model, db_type):
    """
    Builds both:
    - Vector DB (semantic search)
    - BM25 (keyword search)
    """

    # 1. Load documents
    docs = load_pdf(pdf_path)

    # 2. Chunking
    chunks = split_documents(docs, chunk_size)

    # 3. Embeddings
    embeddings = get_embeddings(embedding_model)

    # 4. Vector Store
    vector_db = create_vector_store(chunks, embeddings, db_type)

    # 5. BM25 Retriever
    bm25 = BM25Retriever(chunks)

    return vector_db, bm25


# =========================
# 🔍 Hybrid Retrieval
# =========================
def retrieve_docs(query, vector_db, bm25, k=3):
    """
    Hybrid retrieval:
    - Vector similarity search
    - BM25 keyword search
    - Merge results
    """

    # Vector search
    vector_docs = vector_db.similarity_search(query, k=k)

    # BM25 search
    bm25_docs = bm25.get_top_k(query, k=k)

    # Merge results (remove duplicates)
    combined = []
    seen = set()

    for doc in vector_docs + bm25_docs:
        text = doc.page_content

        if text not in seen:
            combined.append(doc)
            seen.add(text)

    return combined[:k]


# =========================
# 🤖 Answer Generation
# =========================
def generate_answer(query, vector_db, bm25, llm_model="llama3"):
    """
    Generate answer using:
    - Hybrid retrieved context
    - LLM
    """

    # 1. Retrieve documents
    docs = retrieve_docs(query, vector_db, bm25)

    # 2. Build context
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Prompt
    prompt = f"""
Answer the question using ONLY the context.
Give a SHORT and PRECISE answer (1-2 lines).
Do NOT add extra explanation.

Context:
{context}

Question:
{query}

Answer:
"""

    # 4. Load LLM
    llm = get_llm(llm_model)

    # 5. Generate response
    response = llm.invoke(prompt)

    return response, docs