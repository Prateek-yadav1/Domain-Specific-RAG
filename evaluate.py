import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import re

from rag_pipeline import build_vector_database, generate_answer
from evaluation import compute_rouge, compute_bleu, compute_bertscore


# -----------------------------
# ⚙️ CONFIG
# -----------------------------
PDF_PATH = "data/documents/N5.pdf"
CHUNK_SIZE = 500
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_TYPE = "faiss"
LLM_MODEL = "llama3"   # or "gemma"


# -----------------------------
# 🔧 Accuracy Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_accuracy(reference, prediction):
    reference = clean_text(reference)
    prediction = clean_text(prediction)

    if reference == prediction:
        return 1

    if reference in prediction or prediction in reference:
        return 1

    return 0


# -----------------------------
# 🚀 Build RAG System
# -----------------------------
print("🔄 Building RAG system...")

vector_db, bm25 = build_vector_database(
    pdf_path=PDF_PATH,
    chunk_size=CHUNK_SIZE,
    embedding_model=EMBEDDING_MODEL,
    db_type=DB_TYPE
)

print("✅ System Ready!")


# -----------------------------
# 📂 Load Dataset
# -----------------------------
df = pd.read_csv("data/qa_pairs/test_data.csv")

rouge_total = 0
bleu_total = 0
bert_total = 0
accuracy_total = 0

total = len(df)


# -----------------------------
# 🔄 Evaluation Loop
# -----------------------------
for index, row in df.iterrows():

    question = row["Question"]
    ground_truth = row["Ground Truth"]

    print("\n==============================")
    print("Question:", question)

    try:
        prediction, _ = generate_answer(
            query=question,
            vector_db=vector_db,
            bm25=bm25,
            llm_model=LLM_MODEL
        )

    except Exception as e:
        print("LLM Error:", e)
        prediction = ""

    print("Prediction:", prediction)
    print("Ground Truth:", ground_truth)

    rouge = compute_rouge(ground_truth, prediction)
    bleu = compute_bleu(ground_truth, prediction)
    bert = compute_bertscore(ground_truth, prediction)
    acc = compute_accuracy(ground_truth, prediction)

    print(f"ROUGE: {rouge:.2f}, BLEU: {bleu:.2f}, BERT: {bert:.2f}, ACC: {acc}")

    rouge_total += rouge
    bleu_total += bleu
    bert_total += bert
    accuracy_total += acc


# -----------------------------
# 📊 Final Results
# -----------------------------
print("\n===== FINAL RESULTS =====")
print("ROUGE:", round(rouge_total / total, 3))
print("BERTScore:", round(bert_total / total, 3))
print("Accuracy:", round(accuracy_total / total, 3))