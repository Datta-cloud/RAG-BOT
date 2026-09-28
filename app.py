import os
import shutil
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "mistral-embed"

# Groq LLM
LLM_MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG BOT",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

You are answering questions about a document.

IMPORTANT RULES:

1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. Give a clear and simple answer.
4. Answer directly and naturally.
5. If the answer is not present in the context, say:

"I could not find the answer in the document."

Context:

{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# ============================================================
# SESSION STATE
# ============================================================

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "num_pages" not in st.session_state:
    st.session_state.num_pages = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "work_dir" not in st.session_state:
    st.session_state.work_dir = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# CHECK API KEYS
# ============================================================

def check_api_keys():

    missing = []

    # Mistral is used for embeddings
    if not os.getenv("MISTRAL_API_KEY"):
        missing.append("MISTRAL_API_KEY")

    # Groq is used for LLM
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")

    return missing


# ============================================================
# GET GROQ LLM
# ============================================================

def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file."
        )

    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=0,
        max_tokens=1024,
        groq_api_key=api_key
    )

    return llm


# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_response(response):

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content

    # Some models/providers may return a list
    if isinstance(content, list):

        text = ""

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text += item.get(
                        "text",
                        ""
                    )

            else:

                text += str(item)

        return text

    return str(content)


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    # --------------------------------------------------------
    # CREATE TEMPORARY DIRECTORY
    # --------------------------------------------------------

    work_dir = tempfile.mkdtemp(
        prefix="ragbot_"
    )

    # --------------------------------------------------------
    # SAVE UPLOADED PDF
    # --------------------------------------------------------

    pdf_path = os.path.join(
        work_dir,
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # --------------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    if not docs:

        shutil.rmtree(
            work_dir,
            ignore_errors=True
        )

        raise ValueError(
            "Could not extract text from this PDF."
        )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        docs
    )

    if not chunks:

        shutil.rmtree(
            work_dir,
            ignore_errors=True
        )

        raise ValueError(
            "Could not create document chunks."
        )

    # --------------------------------------------------------
    # MISTRAL EMBEDDINGS
    # --------------------------------------------------------

    embedding_model = MistralAIEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )

    # --------------------------------------------------------
    # CHROMA VECTOR DATABASE
    # --------------------------------------------------------

    persist_dir = os.path.join(
        work_dir,
        "chroma_db"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name="uploaded_book"
    )

    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return (
        retriever,
        len(docs),
        work_dir
    )


# ============================================================
# ASK QUESTION
# ============================================================

def answer_question(question):

    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENT CHUNKS
    # --------------------------------------------------------

    docs = st.session_state.retriever.invoke(
        question
    )

    if not docs:

        return (
            "I could not find the answer in the document.",
            []
        )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    final_prompt = PROMPT.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # --------------------------------------------------------
    # GROQ LLM
    # --------------------------------------------------------

    llm = get_llm()

    # --------------------------------------------------------
    # RETRY CONFIGURATION
    # --------------------------------------------------------

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = llm.invoke(
                final_prompt
            )

            answer = clean_response(
                response
            )

            return answer, docs

        except Exception as e:

            error_text = str(e)

            # ==================================================
            # GROQ RATE LIMIT - 429
            # ==================================================

            if (
                "429" in error_text
                or "rate_limit" in error_text.lower()
                or "rate limit" in error_text.lower()
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                if attempt < max_retries - 1:

                    # Wait before retrying
                    wait_time = 2 ** attempt

                    time.sleep(
                        wait_time
                    )

                    continue

                return (
                    "⚠️ Groq rate limit reached. "
                    "Please wait a little and try again.",
                    []
                )

            # ==================================================
            # GROQ SERVER ERROR - 500/502/503
            # ==================================================

            if (
                "500" in error_text
                or "502" in error_text
                or "503" in error_text
                or "UNAVAILABLE" in error_text
                or "Internal Server Error" in error_text
            ):

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    time.sleep(
                        wait_time
                    )

                    continue

                return (
                    "⚠️ Groq is temporarily unavailable. "
                    "Please try again in a few seconds.",
                    []
                )

            # ==================================================
            # OTHER ERROR
            # ==================================================

            return (
                f"⚠️ Groq error: {error_text}",
                []
            )

    return (
        "⚠️ Unable to generate an answer.",
        []
    )


# ============================================================
# REMOVE BOOK
# ============================================================

def remove_book():

    if (
        st.session_state.work_dir
        and os.path.isdir(
            st.session_state.work_dir
        )
    ):

        shutil.rmtree(
            st.session_state.work_dir,
            ignore_errors=True
        )

    st.session_state.book_name = None
    st.session_state.retriever = None
    st.session_state.num_pages = None
    st.session_state.work_dir = None
    st.session_state.messages = []
    st.session_state.pending_question = None


# ============================================================
# RENDER SOURCES
# ============================================================

def render_sources(
    sources,
    book_name
):

    with st.expander(
        "📚 View Sources"
    ):

        for i, src in enumerate(
            sources,
            1
        ):

            page = src.metadata.get(
                "page"
            )

            st.markdown(
                f"### 📚 Source {i}"
            )

            st.markdown(
                f"**Book:** {book_name}"
            )

            if page is not None:

                st.markdown(
                    f"**Page:** {page + 1}"
                )

            st.markdown(
                "**Content:**"
            )

            st.write(
                src.page_content
            )

            st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 RAG BOT")

    st.markdown("---")

    # --------------------------------------------------------
    # API KEY STATUS
    # --------------------------------------------------------

    missing_keys = check_api_keys()

    if missing_keys:

        st.error(
            "Missing API key(s): "
            + ", ".join(missing_keys)
        )

    else:

        st.success(
            "API keys detected ✅"
        )

        st.caption(
            "LLM: Groq GPT-OSS 120B"
        )

        st.caption(
            "Embeddings: Mistral"
        )

    # --------------------------------------------------------
    # UPLOAD BOOK
    # --------------------------------------------------------

    st.subheader(
        "📚 Upload Book"
    )

    uploaded_file = st.file_uploader(
        "Upload your book or PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.info(
            f"📄 {uploaded_file.name}"
        )

        process_button = st.button(
            "⚙️ Process Document",
            use_container_width=True,
            disabled=bool(missing_keys)
        )

        if process_button:

            with st.spinner(
                "Processing document..."
            ):

                try:

                    # ------------------------------------------------
                    # REMOVE PREVIOUS DOCUMENT
                    # ------------------------------------------------

                    if st.session_state.work_dir:

                        shutil.rmtree(
                            st.session_state.work_dir,
                            ignore_errors=True
                        )

                    # ------------------------------------------------
                    # PROCESS PDF
                    # ------------------------------------------------

                    (
                        retriever,
                        num_pages,
                        work_dir
                    ) = process_pdf(
                        uploaded_file
                    )

                    # ------------------------------------------------
                    # SAVE SESSION STATE
                    # ------------------------------------------------

                    st.session_state.retriever = retriever

                    st.session_state.num_pages = (
                        num_pages
                    )

                    st.session_state.book_name = (
                        uploaded_file.name
                    )

                    st.session_state.work_dir = (
                        work_dir
                    )

                    st.session_state.messages = []

                    st.success(
                        "✅ Document processed successfully!"
                    )

                    st.caption(
                        f"Pages: {num_pages}"
                    )

                except Exception as e:

                    st.error(
                        "❌ Error while processing document"
                    )

                    st.exception(e)

    # --------------------------------------------------------
    # CHAT CONTROLS
    # --------------------------------------------------------

    if st.session_state.book_name:

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🆕 New Chat",
                use_container_width=True
            ):

                st.session_state.messages = []

                st.rerun()

        with col2:

            if st.button(
                "🗑️ Remove Book",
                use_container_width=True
            ):

                remove_book()

                st.rerun()


# ============================================================
# MAIN PAGE
# ============================================================

st.title("📚 RAG BOT")

st.caption(
    "Chat with your books and documents using Retrieval Augmented Generation."
)


# ============================================================
# NO DOCUMENT
# ============================================================

if not st.session_state.book_name:

    st.info(
        """
        📖 **No document uploaded yet.**

        Upload a PDF from the sidebar to start chatting
        with your document.
        """
    )


# ============================================================
# DOCUMENT LOADED
# ============================================================

else:

    st.subheader(
        "📖 Current Document"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"**📄 Book:**\n\n"
            f"{st.session_state.book_name}"
        )

    with col2:

        st.markdown(
            f"**📑 Pages:**\n\n"
            f"{st.session_state.num_pages}"
        )

    with col3:

        st.markdown(
            "**Status:**\n\n"
            "🟢 Ready"
        )

    st.markdown("---")

    # ========================================================
    # EXAMPLE QUESTIONS
    # ========================================================

    if not st.session_state.messages:

        st.markdown(
            "### 💡 Try asking"
        )

        examples = [
            "What is this book about?",
            "What are the main topics covered?",
            "Explain the important concepts."
        ]

        cols = st.columns(3)

        for col, example in zip(
            cols,
            examples
        ):

            with col:

                if st.button(
                    example,
                    use_container_width=True
                ):

                    st.session_state.pending_question = (
                        example
                    )

                    st.rerun()

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):

            st.markdown(
                msg["content"]
            )

            if (
                msg["role"] == "assistant"
                and msg.get("sources")
            ):

                render_sources(
                    msg["sources"],
                    st.session_state.book_name
                )

    # ========================================================
    # CHAT INPUT
    # ========================================================

    typed_question = st.chat_input(
        "Ask a question about your book..."
    )

    question = (
        st.session_state.pending_question
        or typed_question
    )

    st.session_state.pending_question = None

    # ========================================================
    # PROCESS QUESTION
    # ========================================================

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                question
            )

        # ----------------------------------------------------
        # ASSISTANT MESSAGE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🔎 Searching document..."
            ):

                answer, sources = answer_question(
                    question
                )

            st.markdown(
                answer
            )

            if sources:

                render_sources(
                    sources,
                    st.session_state.book_name
                )

        # ----------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )
