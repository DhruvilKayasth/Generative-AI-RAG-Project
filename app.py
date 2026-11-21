import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------------
# App configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma-db"
COLLECTION_NAME = "rag_documents"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Knowledge Finder | RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 50% -10%, rgba(99, 102, 241, 0.10), transparent 34%),
            var(--background-color);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.16);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.3rem;
    }

    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin-bottom: 0.15rem;
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 11px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        margin-right: 8px;
        font-size: 17px;
    }

    .muted {
        color: var(--text-color);
        opacity: 0.62;
        font-size: 0.88rem;
    }

    .hero {
        text-align: center;
        padding: 3.2rem 0 1.4rem 0;
    }

    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 2.8rem);
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.65rem;
    }

    .hero p {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 1.02rem;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 12px;
        border: 1px solid rgba(99,102,241,0.28);
        background: rgba(99,102,241,0.08);
        color: #818cf8;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .stChatMessage {
        border: 1px solid rgba(128,128,128,0.12);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatInput"] {
        border-radius: 18px;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 0.98rem;
    }

    .sidebar-card {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 14px;
        padding: 13px;
        margin: 10px 0 16px 0;
        background: rgba(128,128,128,0.045);
    }

    .small-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.58;
        font-weight: 700;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.6rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.16);
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )


def get_api_key_status():
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def response_to_text(content):
    """Handle both string and block-based Gemini response content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def build_vector_database(uploaded_files):
    """Load uploaded PDFs, split them, embed chunks and create a fresh Chroma DB."""
    all_documents = []

    with st.spinner("Reading your PDF and preparing its content..."):
        temp_dir = BASE_DIR / ".rag_upload_tmp"
        temp_dir.mkdir(exist_ok=True)

        try:
            for uploaded_file in uploaded_files:
                safe_name = Path(uploaded_file.name).name
                pdf_path = temp_dir / safe_name
                pdf_path.write_bytes(uploaded_file.getvalue())

                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()

                # Preserve the uploaded filename in each chunk's metadata.
                for document in documents:
                    document.metadata["source"] = safe_name

                all_documents.extend(documents)

            if not all_documents:
                raise ValueError("No readable text was found in the uploaded PDF.")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(all_documents)

            if not chunks:
                raise ValueError("No text chunks could be created from these PDFs.")

            embeddings = get_embeddings()

            # Replace the old database so this upload becomes the active knowledge base.
            if DB_DIR.exists():
                shutil.rmtree(DB_DIR)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(DB_DIR),
            )

            return vectorstore, len(chunks)

        finally:
            # The source PDFs are represented in Chroma; temporary copies are no longer needed.
            shutil.rmtree(temp_dir, ignore_errors=True)


def answer_question(question):
    docs = st.session_state.retriever.invoke(question)

    context_parts = []
    for index, doc in enumerate(docs, start=1):
        source = Path(str(doc.metadata.get("source", "Uploaded PDF"))).name
        page = doc.metadata.get("page")
        page_text = f", page {int(page) + 1}" if isinstance(page, (int, float)) else ""
        context_parts.append(
            f"[Source {index}: {source}{page_text}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)


    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    You are an intelligent AI learning assistant
    powered by RAG and Gemini.

    You have access to retrieved PDF context and
    your general knowledge.

    Follow these rules:

    1. If relevant information is available in the PDF,
    use it as the primary source.

    2. If the user asks for examples, analogies,
    or further explanation, you may use your
    general knowledge to enhance the answer.

    3. If the PDF does not contain the answer,
    answer using your general knowledge and
    clearly mention:
    "This answer is based on general AI knowledge,
    not the uploaded PDF."

    4. Never claim that information comes from the
    PDF if it is not present in the retrieved context.

    5. Give detailed, beginner-friendly explanations.

    Retrieved Context:
    {context}
    """
        ),
        ("human", "{question}")
    ])

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = get_llm().invoke(final_prompt)
    return response_to_text(response.content), docs


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✦</span>Knowledge Finder</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="muted">Your personal document research assistant</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### 📚 Knowledge base")
    st.caption("Upload one or more PDF books to create your searchable knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="The uploaded PDFs are processed locally into text chunks and embeddings.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDF file(s) selected.")
        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    if st.button(
        "⚡ Process PDFs",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        try:
            vectorstore, chunk_count = build_vector_database(uploaded_files)
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5,
                },
            )
            st.session_state.indexed_files = [file.name for file in uploaded_files]
            st.session_state.chunk_count = chunk_count
            st.session_state.messages = []
            st.success(f"Ready! Indexed {chunk_count} chunks.")
        except Exception as error:
            st.error(f"Could not process the PDF(s): {error}")

    st.divider()

    st.markdown("#### 📊 Index status")
    if st.session_state.retriever is not None:
        st.success("Knowledge base ready")
        st.caption(f"{st.session_state.chunk_count} text chunks indexed")
        for name in st.session_state.indexed_files:
            st.caption(f"📄 {name}")
    else:
        st.info("No PDF indexed yet. Upload a PDF and select Process PDFs.")

    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑 Clear knowledge base", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.indexed_files = []
        st.session_state.chunk_count = 0
        st.session_state.messages = []
        if DB_DIR.exists():
            shutil.rmtree(DB_DIR, ignore_errors=True)
        st.success("Knowledge base cleared.")
        st.rerun()

    st.divider()
    st.markdown('<div class="small-label">How it works</div>', unsafe_allow_html=True)
    st.caption("PDF → Text chunks → Embeddings → ChromaDB → Relevant context → Gemini answer")

    if not get_api_key_status():
        st.warning(
            "Gemini API key not found. Add GOOGLE_API_KEY or GEMINI_API_KEY "
            "to your .env file."
        )


# -----------------------------
# Main chat area
# -----------------------------
main_left, main_center, main_right = st.columns([1, 5, 1])

with main_center:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="hero">
                <div class="status-pill">✦ AI-powered document search</div>
                <h1>What would you like to learn?</h1>
                <p>Ask questions about your uploaded books and documents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.retriever is None:
            st.info(
                "Start by uploading a PDF from the left sidebar and clicking "
                "**Process PDFs**. Your document will become the chatbot's knowledge base."
            )
        else:
            st.markdown("##### Try asking")
            suggestion_cols = st.columns(2)
            suggestions = [
                "Summarize the main topics in this book.",
                "Explain the most important concepts.",
                "What are the key definitions?",
                "Create revision notes from this document.",
            ]
            for index, suggestion in enumerate(suggestions):
                with suggestion_cols[index % 2]:
                    if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                        st.session_state.pending_question = suggestion
                        st.rerun()

    # Render previous messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(f"🔎 View retrieved sources ({len(message['sources'])})"):
                    for source_index, doc in enumerate(message["sources"], start=1):
                        source_name = Path(
                            str(doc.metadata.get("source", "Uploaded PDF"))
                        ).name
                        page = doc.metadata.get("page")
                        page_label = (
                            f" · Page {int(page) + 1}"
                            if isinstance(page, (int, float))
                            else ""
                        )
                        st.markdown(f"**Source {source_index}: {source_name}{page_label}**")
                        st.write(doc.page_content)
                        if source_index < len(message["sources"]):
                            st.divider()

    pending_question = st.session_state.pop("pending_question", None)
    question = st.chat_input(
        "Message Knowledge Finder...",
        disabled=st.session_state.retriever is None,
    )

    if pending_question:
        question = pending_question

    if question:
        if not get_api_key_status():
            st.error(
                "Gemini API key is missing. Add GOOGLE_API_KEY or GEMINI_API_KEY "
                "to your .env file, then restart the app."
            )
        else:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching your documents and generating an answer..."):
                    try:
                        answer, source_docs = answer_question(question)
                        st.markdown(answer)

                        with st.expander(
                            f"🔎 View retrieved sources ({len(source_docs)})"
                        ):
                            for source_index, doc in enumerate(source_docs, start=1):
                                source_name = Path(
                                    str(doc.metadata.get("source", "Uploaded PDF"))
                                ).name
                                page = doc.metadata.get("page")
                                page_label = (
                                    f" · Page {int(page) + 1}"
                                    if isinstance(page, (int, float))
                                    else ""
                                )
                                st.markdown(
                                    f"**Source {source_index}: {source_name}{page_label}**"
                                )
                                st.write(doc.page_content)
                                if source_index < len(source_docs):
                                    st.divider()

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": source_docs,
                            }
                        )
                    except Exception as error:
                        error_message = f"Sorry, something went wrong: {error}"
                        st.error(error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_message}
                        )

    if st.session_state.retriever is None:
        st.caption("Upload and process a PDF to enable the chat input.")
    else:
        st.caption("Answers are generated from retrieved document chunks. Always verify important information against the source.")
