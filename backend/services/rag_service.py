import os
import logging
try:
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logger = logging.getLogger("sahayak.rag_service")

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medical_docs")
VECTOR_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faiss_index")

_vector_store = None

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_documents():
    """Loads PDFs from medical_docs directory and builds a local FAISS vector store."""
    if not RAG_AVAILABLE:
        logger.error("RAG dependencies missing. pip install langchain langchain-community sentence-transformers faiss-cpu pypdf")
        return False

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        logger.info(f"Created directory {DOCS_DIR}. Please place medical PDFs here.")
        return False
        
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        logger.warning(f"No PDFs found in {DOCS_DIR}.")
        return False
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    logger.info(f"Successfully processed {len(documents)} PDFs and saved FAISS index.")
    
    global _vector_store
    _vector_store = vector_store
    return True

def load_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store
        
    if not RAG_AVAILABLE:
        return None

    if os.path.exists(VECTOR_DB_PATH):
        embeddings = get_embeddings_model()
        # allow_dangerous_deserialization=True is safe here as we generated the index locally
        _vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return _vector_store
    return None

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieves relevant medical text from the local vector store to help the LLM answer."""
    store = load_vector_store()
    if not store:
        return ""
    
    docs = store.similarity_search(query, k=k)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    return context
