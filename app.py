import streamlit as st
from src.pdf_proc import PDFProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.qa_engine import QAEngine

# Page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_models():
    """Load and cache models"""
    embedding_gen = EmbeddingGenerator()
    qa_engine = QAEngine()
    return embedding_gen, qa_engine

# Title
st.markdown('<h1 style="text-align: center; color: #1E88E5;">📚 PDF Question Answering System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""

# Sidebar
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                try:
                    # Initialize processors
                    pdf_processor = PDFProcessor()
                    embedding_gen, _ = load_models()
                    
                    # Extract text
                    text = pdf_processor.extract_text(uploaded_file)
                    st.session_state.pdf_text = text
                    
                    # Chunk text
                    chunks = pdf_processor.chunk_text(text)
                    st.success(f"✅ Extracted {len(chunks)} chunks from PDF")
                    
                    # Generate embeddings
                    embeddings = embedding_gen.generate_embeddings(chunks)
                    
                    # Store in vector database
                    vector_store = VectorStore()
                    vector_store.add_embeddings(embeddings, chunks)
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True
                    
                    st.success("✅ PDF processed successfully!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 📊 Document Stats")
    if st.session_state.pdf_processed:
        st.metric("Characters", len(st.session_state.pdf_text))
        st.metric("Chunks", len(st.session_state.vector_store.chunks))

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask Questions")
    
    if not st.session_state.pdf_processed:
        st.info("👈 Please upload and process a PDF document first")
    else:
        question = st.text_input("Enter your question:", placeholder="What is this document about?")
        
        if st.button("Get Answer") and question:
            with st.spinner("Generating answer..."):
                try:
                    # Load models
                    embedding_gen, qa_engine = load_models()
                    
                    # Generate query embedding
                    query_embedding = embedding_gen.generate_query_embedding(question)
                    
                    # Search for relevant chunks
                    results = st.session_state.vector_store.search(query_embedding, k=3)
                    
                    # Generate answer
                    answer = qa_engine.generate_answer(question, results)
                    
                    # Display answer
                    st.markdown("### 🎯 Answer:")
                    st.success(answer)
                    
                    # Show source chunks
                    with st.expander("📖 View Source Context"):
                        for i, (chunk, distance) in enumerate(results, 1):
                            st.markdown(f"**Chunk {i}** (Distance: {distance:.4f})")
                            st.text(chunk)
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.header("ℹ️ How it works")
    st.markdown("""
    1. **Upload PDF**: Choose your document
    2. **Process**: Text is extracted and embedded
    3. **Ask**: Type your question
    4. **Get Answer**: AI finds relevant content and answers
    
    **Features:**
    - 🔒 Runs completely locally
    - 🚀 No external APIs needed
    - 📊 Shows source context
    - ⚡ Fast similarity search
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        Built with Streamlit • Powered by Open Source AI
    </div>
""", unsafe_allow_html=True)