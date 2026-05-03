import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import os

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_pipeline import build_vector_database, generate_answer, retrieve_docs
from loader import load_pdf
from chunking import split_documents

# =========================
# 🎨 Page Configuration
# =========================
st.set_page_config(
    page_title="RAG Document QA Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .answer-box {
    color:black;
        background-color: #e3f2fd;
        padding: 15px;
        border-left: 4px solid #1976d2;
        border-radius: 5px;
        margin: 10px 0;
    }
    .context-box {
    color:black;
        background-color: #f5f5f5;
        padding: 10px;
        border-left: 4px solid #666;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 📊 Initialize Session State
# =========================
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "config" not in st.session_state:
    st.session_state.config = {
        "pdf_path": "data/documents/Policy.pdf",
        "chunk_size": 500,
        "embedding_model": "all-MiniLM-L6-v2",
        "db_type": "faiss",
        "llm_model": "llama3",
    }

# =========================
# 🔧 Sidebar Configuration
# =========================
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("System Settings")
    
    # PDF Path
    pdf_path = st.text_input(
        "📄 PDF Path",
        value=st.session_state.config["pdf_path"],
        help="Path to the PDF document"
    )
    
    # Chunk Size
    chunk_size = st.slider(
        "📏 Chunk Size",
        min_value=100,
        max_value=2000,
        value=st.session_state.config["chunk_size"],
        step=100,
        help="Number of characters per chunk"
    )
    
    # Embedding Model
    embedding_model = st.selectbox(
        "🧠 Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0 if st.session_state.config["embedding_model"] == "all-MiniLM-L6-v2" else 1,
        help="Sentence transformer model for embeddings"
    )
    
    # DB Type
    db_type = st.selectbox(
        "🗄️ Vector DB Type",
        ["faiss", "chroma"],
        index=0 if st.session_state.config["db_type"] == "faiss" else 1,
        help="Vector database backend"
    )
    
    # LLM Model
    llm_model = st.selectbox(
        "🤖 LLM Model",
        ["llama3", "gemma"],
        index=0 if st.session_state.config["llm_model"] == "llama3" else 1,
        help="Language model for answer generation"
    )
    
    # Retrieval Settings
    st.subheader("Retrieval Settings")
    k = st.slider(
        "📚 Number of Documents to Retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of context documents for answer generation"
    )
    
    # Build System Button
    if st.button("🔄 Build/Rebuild System", use_container_width=True):
        # Update config
        st.session_state.config = {
            "pdf_path": pdf_path,
            "chunk_size": chunk_size,
            "embedding_model": embedding_model,
            "db_type": db_type,
            "llm_model": llm_model,
        }
        
        with st.spinner("🔄 Building RAG system..."):
            try:
                vector_db, bm25 = build_vector_database(
                    pdf_path=pdf_path,
                    chunk_size=chunk_size,
                    embedding_model=embedding_model,
                    db_type=db_type
                )
                st.session_state.vector_db = vector_db
                st.session_state.bm25 = bm25
                st.session_state.chat_history = []  # Reset chat history
                st.success("✅ System built successfully!")
            except Exception as e:
                st.error(f"❌ Error building system: {str(e)}")
    
    # Clear History Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# =========================
# 📋 Main Content
# =========================
st.title("📚 RAG Document QA Dashboard")

# Check if system is initialized
if st.session_state.vector_db is None or st.session_state.bm25 is None:
    st.info("👈 Configure settings and click 'Build/Rebuild System' in the sidebar to get started!")
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A", "📊 Statistics", "📈 Chat History", "ℹ️ About"])
    
    # =========================
    # Tab 1: Q&A Interface
    # =========================
    with tab1:
        st.subheader("Ask Questions About Your Document")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_input(
                "Your Question",
                placeholder="e.g., What is the main topic of this document?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.button("🔍 Search", use_container_width=True, type="primary")
        
        if submit_button and user_query:
            with st.spinner("⏳ Generating answer..."):
                try:
                    answer, docs = generate_answer(
                        user_query,
                        st.session_state.vector_db,
                        st.session_state.bm25,
                        llm_model=st.session_state.config["llm_model"]
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "query": user_query,
                        "answer": answer,
                        "context_count": len(docs)
                    })
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
        
        # Display latest answer
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            
            st.markdown("---")
            st.markdown("### 💡 Answer")
            st.markdown(
                f'<div class="answer-box">{latest["answer"]}</div>',
                unsafe_allow_html=True
            )
            
            # Display retrieved documents
            with st.expander(f"📄 Retrieved Context ({latest['context_count']} documents)", expanded=True):
                try:
                    retrieved_docs = retrieve_docs(
                        latest["query"],
                        st.session_state.vector_db,
                        st.session_state.bm25,
                        k=k
                    )
                    
                    for i, doc in enumerate(retrieved_docs, 1):
                        with st.container():
                            st.markdown(f"**Document {i}:**")
                            st.markdown(
                                f'<div class="context-box">{doc.page_content[:500]}...</div>',
                                unsafe_allow_html=True
                            )
                except Exception as e:
                    st.warning(f"Could not retrieve context details: {str(e)}")
    
    # =========================
    # Tab 2: Statistics
    # =========================
    with tab2:
        st.subheader("System Statistics")
        
        # Get document info
        try:
            docs = load_pdf(st.session_state.config["pdf_path"])
            chunks = split_documents(docs, st.session_state.config["chunk_size"])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📄 Total Documents", len(docs))
            
            with col2:
                st.metric("📏 Total Chunks", len(chunks))
            
            with col3:
                st.metric("💬 Chat Messages", len(st.session_state.chat_history))
            
            with col4:
                total_chars = sum(len(chunk.page_content) for chunk in chunks)
                st.metric("📝 Total Characters", f"{total_chars:,}")
            
            # Configuration Display
            st.subheader("Current Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**System Settings:**")
                for key, value in st.session_state.config.items():
                    st.write(f"• {key}: `{value}`")
            
            with col2:
                st.write("**Retrieval Settings:**")
                st.write(f"• Number of documents to retrieve: `{k}`")
                st.write(f"• Embedding dimension: `384`")
                st.write(f"• DB backend: `{st.session_state.config['db_type']}`")
        
        except Exception as e:
            st.error(f"❌ Error loading statistics: {str(e)}")
    
    # =========================
    # Tab 3: Chat History
    # =========================
    with tab3:
        st.subheader("Conversation History")
        
        if not st.session_state.chat_history:
            st.info("No conversation history yet. Start by asking a question in the Q&A tab!")
        else:
            for idx, interaction in enumerate(st.session_state.chat_history, 1):
                with st.container():
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        st.write(f"**#{idx}**")
                    with col2:
                        timestamp = datetime.fromisoformat(interaction["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                        st.write(f"__{timestamp}__")
                    
                    st.markdown("**Query:**")
                    st.write(interaction["query"])
                    
                    st.markdown("**Answer:**")
                    st.markdown(
                        f'<div class="answer-box">{interaction["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.caption(f"Context documents retrieved: {interaction['context_count']}")
                    st.markdown("---")
    
    # =========================
    # Tab 4: About
    # =========================
    with tab4:
        st.subheader("About This Dashboard")
        
        st.markdown("""
        ### RAG Document QA System
        
        This dashboard provides a web interface for your Retrieval-Augmented Generation (RAG) system.
        
        **Features:**
        - 💬 **Interactive Q&A**: Ask questions about your documents
        - 📊 **Statistics**: View system metrics and configuration
        - 📈 **Chat History**: Review past interactions
        - ⚙️ **Configuration**: Customize system parameters
        
        **How it Works:**
        1. Upload or configure your PDF document
        2. The system builds embeddings and creates a vector database
        3. When you ask a question, it retrieves relevant documents using hybrid search
        4. An LLM generates an answer based on retrieved context
        
        **Technologies:**
        - **Vector DB**: FAISS or Chroma
        - **Embeddings**: Sentence Transformers
        - **LLM**: Llama 3 or Gemma (via Ollama)
        - **Framework**: Streamlit
        
        **Tips:**
        - Adjust chunk size for better context windows
        - Try different embedding models for better semantic search
        - Increase retrieval count for more context
        """)
        
        st.info("📚 For more details, check the README or source code in the repository.")

# =========================
# 🔧 Footer
# =========================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🚀 RAG Document QA System")
with col2:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    st.caption("Powered by Streamlit")
