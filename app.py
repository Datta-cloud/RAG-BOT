"""
DocuMind AI - Streamlit UI for the RAG project.

Reuses the same pipeline that main.py / create_database.py already implement:
    PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> MistralAIEmbeddings
        -> Chroma -> MMR retriever -> Gemini 2.5 Flash -> answer

The only thing this file adds is a UI layer and turns the "run once on a
hardcoded PDF" script from create_database.py into a reusable function that
runs per-upload, in its own Chroma collection, so uploaded books never mix.

Run with:
    streamlit run app.py
"""

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

EMBEDDING_MODEL_NAME = "mistral-embed"
LLM_MODEL_NAME = "gemini-2.5-flash"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)

st.set_page_config(page_title="RAG BOT ", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "book_name": None,
    "retriever": None,
    "num_pages": None,
    "messages": [],       # list of {role, content, sources}
    "work_dir": None,     # temp dir holding this session's pdf + chroma db
    "pending_question": None,
}
for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Backend helpers (adapted from create_database.py / main.py)
# ---------------------------------------------------------------------------
def check_api_keys():
    missing = []
    if not os.getenv("MISTRAL_API_KEY"):
        missing.append("MISTRAL_API_KEY")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
    return missing


def get_llm():
    # ChatGoogleGenerativeAI looks for GOOGLE_API_KEY by default; your .env
    # uses GEMINI_API_KEY, so it's passed through explicitly here.
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0, google_api_key=api_key)


def process_pdf(uploaded_file):
    """Save -> load -> chunk -> embed -> store. One Chroma collection per
    session/document so books never mix."""
    work_dir = tempfile.mkdtemp(prefix="documind_")
    pdf_path = os.path.join(work_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise ValueError("Could not extract any text from this PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = MistralAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    persist_dir = os.path.join(work_dir, "chroma_db")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="uploaded_book",
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    return retriever, len(docs), work_dir


def answer_question(question: str):
    docs = st.session_state.retriever.invoke(question)
    if not docs:
        return "I could not find the answer in the document.", []

    context = "\n\n".join(d.page_content for d in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": question})
    llm = get_llm()
    response = llm.invoke(final_prompt)
    return response.content, docs


def remove_book():
    if st.session_state.work_dir and os.path.isdir(st.session_state.work_dir):
        shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
    for key in ["book_name", "retriever", "num_pages", "work_dir"]:
        st.session_state[key] = _DEFAULTS[key]
    st.session_state.messages = []


def render_sources(sources, book_name):
    with st.expander("📚 View Sources"):
        for i, src in enumerate(sources, 1):
            page = src.metadata.get("page")
            st.markdown(f"**📚 Source {i}**")
            st.markdown(f"**Book:** {book_name}")
            if page is not None:
                st.markdown(f"**Page:** {page + 1}")
            st.markdown(f"**Content:**\n> {src.page_content}")
            st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 RAG BOT")
    st.markdown("---")

    missing_keys = check_api_keys()
    if missing_keys:
        st.warning("Missing API key(s): " + ", ".join(missing_keys))

    st.subheader("📚 Upload Book")
    uploaded_file = st.file_uploader("Upload your book or PDF", type=["pdf"])

    if uploaded_file is not None:
        st.write(f"📄 {uploaded_file.name}")
        if st.button("Process Document", use_container_width=True, disabled=bool(missing_keys)):
            with st.spinner("Processing document..."):
                try:
                    if st.session_state.work_dir:
                        shutil.rmtree(st.session_state.work_dir, ignore_errors=True)

                    retriever, num_pages, work_dir = process_pdf(uploaded_file)
                    st.session_state.retriever = retriever
                    st.session_state.num_pages = num_pages
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.work_dir = work_dir
                    st.session_state.messages = []
                    st.success("✅ Document processed successfully")
                    st.caption(f"Pages: {num_pages}")
                except Exception as e:
                    st.error(f"⚠️ Failed to process document: {e}")

    st.markdown("---")

    if st.session_state.book_name:
        col1, col2 = st.columns(2)
        if col1.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if col2.button("🗑️ Remove Book", use_container_width=True):
            remove_book()
            st.rerun()

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("📚 RAG BOT")
st.caption("Chat with your books and documents using RAG.")

if not st.session_state.book_name:
    st.info(
        "📖 No book uploaded yet.\n\n"
        "Upload a PDF from the sidebar to start asking questions."
    )
else:
    st.subheader("📖 Current Book")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Book Name:**\n\n{st.session_state.book_name}")
    c2.markdown("**Status:**\n\n🟢 Ready")
    st.caption("You can now ask questions about this book.")

    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        examples = [
            "What is this book about?",
            "What are the main topics covered?",
            "Explain the important concepts in this book.",
        ]
        cols = st.columns(3)
        for col, example in zip(cols, examples):
            if col.button(example, use_container_width=True):
                st.session_state.pending_question = example

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"], st.session_state.book_name)

    typed_question = st.chat_input("Ask a question about your book...")
    question = st.session_state.pending_question or typed_question
    st.session_state.pending_question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = answer_question(question)
                except Exception:
                    answer = (
                        "⚠️ Something went wrong while generating the answer. "
                        "Please check your API keys and try again."
                    )
                    sources = []
            st.markdown(answer)
            if sources:
                render_sources(sources, st.session_state.book_name)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
