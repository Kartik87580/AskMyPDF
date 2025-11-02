import os
import hashlib
import pickle
from io import BytesIO
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


st.set_page_config(page_title="RAG PDF QA", layout="wide")
st.title("PDF Question Answering with RAG & Gemini")


# --- Helper utilities ---
def compute_file_hash(bytes_data: bytes) -> str:
    h = hashlib.sha256()
    h.update(bytes_data)
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    # fast extraction using PyPDF2; cached by file bytes hash
    # PyPDF2 expects a file-like object or path; wrap bytes in BytesIO
    reader = PdfReader(BytesIO(pdf_bytes))
    parts = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


@st.cache_data(show_spinner=False)
def split_text_to_chunks(text: str, chunk_size: int, chunk_overlap: int):
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str, device: str):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})


def faiss_index_path_for_hash(index_dir: str, file_hash: str) -> str:
    return os.path.join(index_dir, f"faiss_index_{file_hash}")


def build_or_load_faiss_index(index_dir: str, file_hash: str, chunks, embeddings):
    os.makedirs(index_dir, exist_ok=True)
    index_path = faiss_index_path_for_hash(index_dir, file_hash)
    # LangChain FAISS exposes save_local and load_local which use a folder path
    if os.path.exists(index_path):
        try:
            vector_store = FAISS.load_local(index_path, embeddings)
            return vector_store, True
        except Exception:
            # if loading fails, remove and rebuild
            pass
    # build and persist
    vector_store = FAISS.from_documents(chunks, embeddings)
    try:
        vector_store.save_local(index_path)
    except Exception:
        # best-effort; continue without failing
        pass
    return vector_store, False


# --- UI: sidebar options for performance tuning ---
with st.sidebar.expander("Settings", expanded=True):
    index_dir = st.text_input("Index directory", value="faiss_index")
    model_name = st.selectbox("Embedding model", options=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"], index=0)
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=500, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=50, step=10)
    top_k = st.number_input("Retriever k", min_value=1, max_value=10, value=3)


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = compute_file_hash(file_bytes)

    # Extract text (cached by identical bytes)
    try:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf_bytes(file_bytes)
    except Exception as e:
        st.error(f"Failed to extract PDF text: {e}")
        st.stop()

    if not text.strip():
        st.warning("No extractable text found in the uploaded PDF.")
        st.stop()

    st.info(f"Extracted {len(text)} characters from PDF.")

    # Split into chunks (cached by text and splitter params)
    chunks = split_text_to_chunks(text, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    st.info(f"Split into {len(chunks)} chunks.")

    # Embeddings (cached resource)
    embeddings = get_embeddings(model_name=model_name, device=device)

    # Build or load FAISS index persisted by file hash
    with st.spinner("Building / loading vector index (FAISS)..."):
        vector_store, loaded = build_or_load_faiss_index(index_dir, file_hash, chunks, embeddings)
    if loaded:
        st.success("Loaded existing vector index for this file (cached).")
    else:
        st.success("Built and cached a new vector index for this file.")

    # LLM setup: read API key from environment or Streamlit secrets (do NOT hardcode)
    google_api_key = os.environ.get("GOOGLE_API_KEY") or (
        st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") and isinstance(st.secrets, dict) else None
    )
    if not google_api_key:
        st.warning("Google API key not found. Set GOOGLE_API_KEY in environment or Streamlit secrets to enable Gemini LLM.")
    else:
        os.environ["GOOGLE_API_KEY"] = google_api_key

    # Build retriever and QA chain only if API key exists
    retriever = vector_store.as_retriever(search_kwargs={"k": int(top_k)})

    query = st.text_input("Ask a question about the PDF:")
    if query:
        if not google_api_key:
            st.error("Cannot run LLM without a configured Google API key.")
        else:
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
                with st.spinner("Generating answer..."):
                    response = qa_chain({"query": query})
                st.markdown(f"**Answer:** {response.get('result', '')}")
                if response.get("source_documents"):
                    st.markdown("**Source Chunks:**")
                    for i, doc in enumerate(response["source_documents"]):
                        st.code(doc.page_content, language="text")
            except Exception as e:
                st.error(f"LLM query failed: {e}")
