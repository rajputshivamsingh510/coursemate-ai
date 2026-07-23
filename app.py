import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind — PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f1116 0%, #14161d 100%);
        }

        [data-testid="stSidebar"] {
            background: #14161d;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #7c3aed 0%, #2563eb 100%);
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 30px rgba(37, 99, 235, 0.25);
        }
        .hero h1 {
            color: white;
            font-size: 1.9rem;
            margin: 0;
            font-weight: 700;
        }
        .hero p {
            color: rgba(255,255,255,0.85);
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }

        .stat-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
        }
        .stat-card .label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: rgba(255,255,255,0.5);
        }
        .stat-card .value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #a5b4fc;
        }

        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            border: 1px dashed rgba(255,255,255,0.15);
            border-radius: 16px;
            color: rgba(255,255,255,0.6);
        }
        .empty-state h3 {
            color: rgba(255,255,255,0.85);
        }

        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.4rem 0.2rem;
        }

        section[data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached resources (loaded once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatMistralAI(model="mistral-small-latest")


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.

            If the answer is not present in the context,
            say: "I could not find the answer in the document"
            """,
        ),
        (
            "human",
            """Context:
            {context}

            Question:
            {question}
            """,
        ),
    ]
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "messages": [],
    "vectorstore": None,
    "processed_file": None,
    "num_pages": 0,
    "num_chunks": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def process_pdf(uploaded_file):
    """Save the uploaded PDF, split it, embed it, and build a vectorstore."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = load_embeddings()
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

        return vectorstore, len(docs), len(chunks)
    finally:
        os.unlink(tmp_path)


def get_answer(query: str) -> str:
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.invoke({"context": context, "question": query})
    llm = load_llm()
    response = llm.invoke(final_prompt)
    return response.content


# --------------------------------------------------------------------------
# Sidebar — upload & document info
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📎 Upload a document")
    uploaded_file = st.file_uploader(
        "Drop a PDF here or browse",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        already_done = st.session_state.processed_file == uploaded_file.name
        process_clicked = st.button(
            "🔍 Process document",
            use_container_width=True,
            disabled=already_done,
        )
        if process_clicked:
            with st.spinner("Reading, chunking, and indexing your PDF..."):
                vectorstore, n_pages, n_chunks = process_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_file = uploaded_file.name
                st.session_state.num_pages = n_pages
                st.session_state.num_chunks = n_chunks
                st.session_state.messages = []
            st.success(f"'{uploaded_file.name}' is ready!")

    if st.session_state.processed_file:
        st.markdown("---")
        st.markdown("### 📊 Document info")
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="label">File</div>
                <div class="value" style="font-size:0.95rem;">{st.session_state.processed_file}</div>
            </div>
            <div class="stat-card">
                <div class="label">Pages</div>
                <div class="value">{st.session_state.num_pages}</div>
            </div>
            <div class="stat-card">
                <div class="label">Chunks indexed</div>
                <div class="value">{st.session_state.num_chunks}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Powered by LangChain · Chroma · Mistral")

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📄 DocMind</h1>
        <p>Upload a PDF and ask questions — answers grounded strictly in your document.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No document loaded yet</h3>
            <p>Upload a PDF from the sidebar and click <b>Process document</b> to start chatting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about your document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(query)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})