# RAG Document QA System

A Retrieval-Augmented Generation (RAG) system for document-based question answering with both CLI and web dashboard interfaces.

## 📋 Features

- **Hybrid Retrieval**: Combines semantic search (vector embeddings) and keyword search (BM25)
- **Multiple Vector Databases**: Support for FAISS and Chroma
- **Flexible LLM Support**: Works with Llama 3, Gemma, and OpenAI models
- **Web Dashboard**: Interactive Streamlit interface for easy access
- **CLI Interface**: Command-line tool for batch processing
- **Evaluation Metrics**: ROUGE, BLEU, and BERTScore metrics for answer quality
- **Chat History**: Track conversation history in the dashboard

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PDF document(s) in `data/documents/` folder
- Ollama (for local LLMs) or OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_document_qa
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

Start the interactive web dashboard:
```bash
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501` and provides:
- 💬 Interactive Q&A interface
- 📊 System statistics and metrics
- 📈 Chat history tracking
- ⚙️ Configuration management

### Running the CLI

Use the command-line interface for batch processing:
```bash
python app.py
```

This will:
1. Load your PDF document
2. Create embeddings and vector database
3. Start an interactive query loop
4. Display answers and retrieved context

## 🏗️ Project Structure

```
rag_document_qa/
├── app.py                  # CLI application
├── dashboard.py            # Streamlit web dashboard
├── evaluate.py             # Evaluation script for metrics
├── requirements.txt        # Python dependencies
├── data/
│   ├── documents/         # PDF files to process
│   └── qa_pairs/
│       └── test_data.csv  # Test dataset
└── src/
    ├── bm25_retriever.py  # BM25 keyword search
    ├── chunking.py        # Document chunking logic
    ├── embeddings.py      # Embedding generation
    ├── evaluation.py      # Evaluation metrics
    ├── llm.py             # LLM interface
    ├── loader.py          # PDF loading
    ├── rag_pipeline.py    # RAG orchestration
    └── vectorstore.py     # Vector database management
```

## ⚙️ Configuration

### CLI (app.py)
Edit the configuration section at the top of `app.py`:
```python
PDF_PATH = "data/documents/N5.pdf"   # Your PDF file
CHUNK_SIZE = 500                      # Characters per chunk
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer
DB_TYPE = "faiss"                     # faiss or chroma
LLM_MODEL = "llama3"                  # llama3 or gemma
```

### Dashboard (dashboard.py)
Configure in the sidebar:
- PDF path
- Chunk size
- Embedding model
- Vector database type
- LLM model
- Number of documents to retrieve

## 📖 How It Works

1. **Loading**: PDFs are loaded and split into chunks
2. **Embeddings**: Each chunk is converted to embeddings using sentence transformers
3. **Storage**: Chunks and embeddings are stored in a vector database
4. **Indexing**: BM25 index is created for keyword-based retrieval
5. **Retrieval**: Query matches both semantic (vector) and keyword (BM25) results
6. **Generation**: Retrieved documents are fed to an LLM to generate answers

## 🤖 Supported Models

### Embedding Models
- `all-MiniLM-L6-v2` (default, fast, 384-dim)
- `all-mpnet-base-v2` (more accurate, 768-dim)

### LLM Models (via Ollama)
- `llama3` - High quality, larger model
- `gemma` - Lightweight alternative

### Alternative LLMs
- OpenAI's GPT models (via OpenAI API)
- Other Ollama-supported models

## 📊 Evaluation

Run evaluation on a test dataset:
```bash
python evaluate.py
```

This computes:
- **ROUGE**: n-gram overlap with reference answers
- **BLEU**: Bilingual evaluation understudy score
- **BERTScore**: Semantic similarity using BERT embeddings

## 🔧 Customization

### Add New Embedding Models
Edit `src/embeddings.py` to add more sentence transformer models.

### Add New Vector Databases
Implement new database wrapper in `src/vectorstore.py`.

### Change LLM Provider
Modify `src/llm.py` to support additional LLM APIs.

## 📝 Tips

- **Larger chunk size** → more context per retrieval, but may include irrelevant info
- **Smaller chunk size** → more specific results, but may lack context
- **Different embedding models** → experiment with `all-mpnet-base-v2` for better semantic understanding
- **More retrieval** → set k=5-10 for more context, but slower answers

## 🐛 Troubleshooting

**Ollama models not found:**
- Install Ollama from https://ollama.ai
- Pull models: `ollama pull llama3` and `ollama pull gemma`

**FAISS not available:**
- CPU version: `pip install faiss-cpu`
- GPU version: `pip install faiss-gpu`

**Streamlit connection issues:**
- Make sure port 8501 is not in use
- Try: `streamlit run dashboard.py --server.port 8502`

## 📄 License

MIT License

## 🤝 Contributing

Feel free to submit issues and pull requests!

---

**Built with**: LangChain • Streamlit • FAISS • Sentence Transformers • Ollama
