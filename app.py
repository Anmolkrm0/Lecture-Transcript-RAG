import gc
import os
import time
import warnings
from typing import Generator, List, Set, Tuple

warnings.filterwarnings("ignore")

import streamlit as st

# Document parsing: PyMuPDF for PDF
try:
    import pymupdf as fitz
except ImportError:
    import fitz

# LangChain Imports with resilient fallback handling
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ==========================================
# Application Configuration & Constants
# ==========================================
APP_TITLE = "Lecture Transcript AI"
APP_ICON = "🎓"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "lecture_transcripts"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"

# Memory Optimization & Ingestion Constraints
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 32
BATCH_DELAY_SECONDS = 0.5
RETRIEVAL_K = 10  # Retrieve top 10 chunks from ChromaDB
RERANK_TOP_N = 5  # Top N highest-scoring chunks after reranking passed to LLM

SYSTEM_PROMPT = (
    "You are an expert AI Study Assistant for lecture transcripts.\n\n"
    "CRITICAL DOMAIN RESTRICTION:\n"
    "- Answer using ONLY the provided lecture context.\n"
    "- If the question asks about topics OUTSIDE the domain of the provided lecture context (e.g., general world trivia, celebrities, or unrelated subjects not mentioned in the context), strictly refuse by saying: 'I couldn't find this information in your uploaded lectures.' Do NOT guess or use outside knowledge.\n\n"
    "CAPABILITIES & FORMATTING INSTRUCTIONS:\n"
    "Deliver rich, meaningful, and pedagogically structured output based on whatever the user requests:\n"
    "1. HIGH-LEVEL MINDMAPS: When requested, generate a clear, hierarchical Markdown text-based mindmap tree (using └──, ├──, │) detailing core themes, subtopics, and key relationships.\n"
    "2. STUDY REPORTS: When asked for a summary, report, or overview, provide a structured study report with an Executive Summary, Key Pillars/Concepts, and Practical Takeaways.\n"
    "3. QUIZZES & MCQs: When asked for a quiz, test, or multiple-choice questions, generate challenging questions directly grounded in the lectures. For MCQs, provide 4 options (A, B, C, D), clearly indicate the correct answer, and include a brief explanation citing the context.\n"
    "4. FLASHCARDS: When asked for flashcards, format each card clearly with [Card N], **Front / Concept**, and **Back / Explanation / Definition**.\n"
    "5. IN-DEPTH EXPLANATIONS: For specific queries, deliver thorough, well-articulated explanations directly addressing the user's intent.\n\n"
    "MANDATORY CITATION:\n"
    "- Always end your response with an explicit citation listing the source document(s) used (e.g., 'Source: [Document Name]')."
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# Resource Caching (@st.cache_resource)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str = LLM_MODEL) -> ChatOllama:
    """Initialize and cache the Ollama LLM.
    Cached once per session to avoid re-allocating unified memory.
    """
    return ChatOllama(
        model=model_name,
        temperature=0.2,
    )


@st.cache_resource(show_spinner=False)
def load_embeddings(model_name: str = EMBEDDING_MODEL) -> OllamaEmbeddings:
    """Initialize and cache the Ollama Embeddings model.
    Cached once per session to prevent repeated model reloading.
    """
    return OllamaEmbeddings(
        model=model_name,
    )


@st.cache_resource(show_spinner=False)
def load_vector_store(_embeddings: OllamaEmbeddings, persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Initialize and cache the ChromaDB client and vector store.
    
    Leading underscore on `_embeddings` prevents Streamlit from attempting
    to hash the complex embedding object.
    """
    os.makedirs(persist_dir, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings,
        persist_directory=persist_dir,
    )


@st.cache_resource(show_spinner=False)
def load_reranker():
    """Initialize and cache the FlashRank local cross-encoder reranker.
    Lightweight (~4MB), runs via ONNX runtime on CPU/MPS with ~10ms latency.
    """
    from flashrank import Ranker
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2")


# ==========================================
# Document Parsing & Memory-Safe Ingestion
# ==========================================
def extract_and_chunk_pdf(
    file_bytes: bytes,
    file_name: str,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Parse PDF page-by-page using PyMuPDF (fitz) and chunk immediately to minimize RAM spikes."""
    chunks = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text()
            if text and text.strip():
                page_doc = Document(
                    page_content=text.strip(),
                    metadata={"source": file_name, "page": page_idx + 1},
                )
                page_chunks = splitter.split_documents([page_doc])
                chunks.extend(page_chunks)
                del page_doc
    finally:
        doc.close()
        del doc
        gc.collect()
    return chunks


def extract_and_chunk_txt(
    file_bytes: bytes,
    file_name: str,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Parse plain text files and chunk them with metadata."""
    text = file_bytes.decode("utf-8", errors="replace")
    if not text.strip():
        return []
    doc = Document(
        page_content=text.strip(),
        metadata={"source": file_name, "page": 1},
    )
    chunks = splitter.split_documents([doc])
    del text
    del doc
    gc.collect()
    return chunks


def parse_uploaded_files(
    uploaded_files,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """Process uploaded files one by one with immediate buffer cleanup."""
    all_chunks: List[Document] = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        if file_name.lower().endswith(".pdf"):
            file_chunks = extract_and_chunk_pdf(file_bytes, file_name, splitter)
        elif file_name.lower().endswith(".txt"):
            file_chunks = extract_and_chunk_txt(file_bytes, file_name, splitter)
        else:
            continue

        all_chunks.extend(file_chunks)
        del file_bytes
        gc.collect()

    return all_chunks


def ingest_chunks_memory_safe(
    vector_store: Chroma,
    chunks: List[Document],
    batch_size: int = BATCH_SIZE,
    delay: float = BATCH_DELAY_SECONDS,
    status_container=None,
) -> dict:
    """Embed chunks into ChromaDB in controlled batches with sleep delays and explicit gc.
    
    This prevents maxing out Apple Silicon Unified Memory and avoiding OS freezing.
    """
    total_chunks = len(chunks)
    if total_chunks == 0:
        return {"chunks": 0, "batches": 0, "elapsed_seconds": 0.0, "embeddings_generated": False, "saved_in_db": False}

    progress_bar = st.sidebar.progress(0.0)
    status_text = st.sidebar.empty()

    start_time = time.time()
    total_batches = (total_chunks + batch_size - 1) // batch_size

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        batch_count = len(batch)
        batch_num = (i // batch_size) + 1

        msg = f"Embedding batch {batch_num}/{total_batches} ({batch_count} chunks)..."
        status_text.caption(msg)
        if status_container:
            status_container.write(f"🔄 **Batch {batch_num}/{total_batches}:** Generating embeddings & writing {batch_count} chunks to ChromaDB...")

        # Ingest the batch
        vector_store.add_documents(documents=batch)

        # Update progress before cleanup
        progress = min((i + batch_count) / total_chunks, 1.0)
        progress_bar.progress(progress)

        # Mandatory sleep to allow Apple Silicon unified memory / GPU buffer to drain
        time.sleep(delay)

        # Explicit garbage collection after each batch
        del batch
        gc.collect()

    elapsed = time.time() - start_time
    status_text.empty()
    progress_bar.empty()

    return {
        "chunks": total_chunks,
        "batches": total_batches,
        "elapsed_seconds": round(elapsed, 2),
        "embeddings_generated": True,
        "saved_in_db": True,
    }


def get_indexed_metadata(vector_store: Chroma) -> Tuple[int, Set[str]]:
    """Retrieve total chunk count and distinct source names stored in ChromaDB."""
    try:
        count = vector_store._collection.count()
        if count == 0:
            return 0, set()
        data = vector_store._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        sources = {m.get("source") for m in metadatas if m and "source" in m}
        return count, sources
    except Exception:
        return 0, set()


def get_document_db_inspection(vector_store: Chroma, source_name: str) -> dict:
    """Retrieve full live inspection logs for a specific document stored in ChromaDB."""
    try:
        data = vector_store._collection.get(
            where={"source": source_name},
            include=["metadatas", "documents", "embeddings"],
        )
        ids = data.get("ids", [])
        chunk_count = len(ids)
        embeddings = data.get("embeddings")
        has_embeddings = embeddings is not None and len(embeddings) > 0
        embedding_dim = len(embeddings[0]) if has_embeddings and len(embeddings[0]) > 0 else 0
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        pages = sorted(list({m.get("page", 1) for m in metadatas if m and "page" in m})) if metadatas else [1]
        page_range = f"Page {pages[0]} to {pages[-1]}" if len(pages) > 1 else f"Page {pages[0]}" if pages else "N/A"

        sample_chunk = docs[0] if docs else ""
        sample_meta = metadatas[0] if metadatas else {}

        return {
            "source": source_name,
            "chunk_count": chunk_count,
            "has_embeddings": has_embeddings,
            "embedding_dim": embedding_dim,
            "embedding_model": EMBEDDING_MODEL,
            "saved_in_db": chunk_count > 0,
            "collection_name": COLLECTION_NAME,
            "db_path": CHROMA_PERSIST_DIR,
            "page_range": page_range,
            "sample_chunk": sample_chunk,
            "sample_meta": sample_meta,
        }
    except Exception as e:
        return {"error": str(e), "source": source_name, "saved_in_db": False, "chunk_count": 0}


def is_summary_request(query: str) -> bool:
    """Detect if a user query is asking for a general lecture summary, mindmap, report, quiz, or flashcards."""
    q = query.lower()
    summary_triggers = [
        "summarize", "summerize", "summary", "summery", "overview", "outline", "synopsis",
        "all the lectures", "all lectures", "what are the lectures about",
        "what did we learn", "topics covered", "review the lectures", "key takeaways",
        "tell me about the lectures", "what is covered", "explain the lecture", "explain all",
        "mindmap", "mind map", "concept map", "report", "study guide",
        "quiz", "quize", "mcq", "mcqs", "questions", "test", "exam",
        "flashcard", "flashcards"
    ]
    return any(trigger in q for trigger in summary_triggers)


def rerank_documents(
    ranker,
    query: str,
    documents: List[Document],
    top_n: int = RERANK_TOP_N,
) -> List[Tuple[Document, float]]:
    """Rerank retrieved chunks using FlashRank cross-encoder and return top_n with relevance scores."""
    if not documents:
        return []
    from flashrank import RerankRequest

    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(documents)
    ]
    rerank_req = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_req)

    reranked_docs = []
    for r in results[:top_n]:
        orig_doc = documents[r["id"]]
        score = float(r.get("score", 0.0))
        reranked_docs.append((orig_doc, score))
    return reranked_docs


# ==========================================
# Streamlit Session State & Init
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize resources
embeddings = load_embeddings()
vector_store = load_vector_store(embeddings)
llm = load_llm()
reranker = load_reranker()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

total_chunks_count, existing_sources = get_indexed_metadata(vector_store)

# ==========================================
# Sidebar UI: Ingestion & Memory Controls
# ==========================================
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("Local, memory-safe RAG assistant for **Apple Silicon Mac**.")
    
    st.markdown("---")
    st.subheader("📥 Add Lecture Transcripts")
    input_tab1, input_tab2 = st.tabs(["📁 Upload Files", "📝 Paste Text"])

    with input_tab1:
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT transcripts",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload lecture slides, notes, or transcripts (PDF or TXT).",
            key="file_uploader_widget",
        )

        if uploaded_files:
            # Check files that are not yet indexed in ChromaDB
            new_files = [f for f in uploaded_files if f.name not in existing_sources]
            
            if new_files:
                st.info(f"{len(new_files)} new file(s) ready to index.")
                if st.button("⚡ Ingest & Index Uploads", type="primary", use_container_width=True, key="ingest_files_btn"):
                    with st.status(f"⚡ Ingesting {len(new_files)} file(s)...", expanded=True) as status_box:
                        status_box.write("📄 **Step 1:** Extracting text and parsing documents...")
                        chunks = parse_uploaded_files(new_files, text_splitter)

                        if chunks:
                            status_box.write(f"🧩 **Step 2:** Created `{len(chunks)}` chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
                            status_box.write(f"🧠 **Step 3:** Generating embeddings via `{EMBEDDING_MODEL}` (Ollama, 768-dim) in batches of {BATCH_SIZE}...")
                            report = ingest_chunks_memory_safe(vector_store, chunks, batch_size=BATCH_SIZE, delay=BATCH_DELAY_SECONDS, status_container=status_box)
                            status_box.write(f"💾 **Step 4:** Saved `{report['chunks']}` chunks to ChromaDB collection `{COLLECTION_NAME}` in {report['elapsed_seconds']}s")
                            status_box.update(label=f"✅ Ingestion Complete! ({len(chunks)} chunks saved)", state="complete", expanded=False)
                            time.sleep(1.0)
                            st.rerun()
                        else:
                            status_box.update(label="⚠️ No readable text found", state="error")
                            st.warning("No readable text found in the selected file(s).")
            else:
                st.success("All selected files are already indexed in ChromaDB.")

    with input_tab2:
        st.caption("Directly paste lecture transcripts, YouTube subtitles, or meeting notes.")
        pasted_title = st.text_input(
            "Transcript Title / Source Name",
            value="",
            placeholder="e.g. Lecture 2 - Generative AI",
            help="This title will be cited in the answers.",
            key="pasted_title_input",
        )
        pasted_text = st.text_area(
            "Paste Transcript Text",
            height=200,
            placeholder="Paste your lecture transcript text here...",
            help="Paste text to chunk and index into ChromaDB.",
            key="pasted_text_input",
        )

        if st.button("⚡ Ingest Pasted Transcript", type="primary", use_container_width=True, key="ingest_paste_btn"):
            if not pasted_text.strip():
                st.warning("Please paste some transcript text first.")
            else:
                title = pasted_title.strip() or f"Pasted Transcript ({time.strftime('%b %d, %H:%M')})"
                with st.status(f"⚡ Ingesting '{title}'...", expanded=True) as status_box:
                    status_box.write("📄 **Step 1:** Parsing pasted transcript text...")
                    raw_doc = Document(
                        page_content=pasted_text.strip(),
                        metadata={"source": title, "page": 1},
                    )
                    chunks = text_splitter.split_documents([raw_doc])
                    del raw_doc
                    gc.collect()

                    if chunks:
                        status_box.write(f"🧩 **Step 2:** Created `{len(chunks)}` chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
                        status_box.write(f"🧠 **Step 3:** Generating embeddings via `{EMBEDDING_MODEL}` (Ollama, 768-dim) in batches of {BATCH_SIZE}...")
                        report = ingest_chunks_memory_safe(vector_store, chunks, batch_size=BATCH_SIZE, delay=BATCH_DELAY_SECONDS, status_container=status_box)
                        status_box.write(f"💾 **Step 4:** Saved `{report['chunks']}` chunks to ChromaDB collection `{COLLECTION_NAME}` in {report['elapsed_seconds']}s")
                        status_box.update(label=f"✅ Ingestion Complete! ({len(chunks)} chunks saved)", state="complete", expanded=False)
                        time.sleep(1.0)
                        st.rerun()
                    else:
                        status_box.update(label="⚠️ No valid text could be processed", state="error")
                        st.warning("No valid text could be processed.")

    st.markdown("---")
    st.subheader("📊 System & Index Status")
    st.metric(label="Total Chunks in ChromaDB", value=total_chunks_count)
    st.caption(f"**Embedding Model:** `{EMBEDDING_MODEL}` (Ollama)")
    st.caption(f"**LLM:** `{LLM_MODEL}` (~2.5GB RAM)")
    st.caption(f"**Reranker:** FlashRank (`ms-marco-TinyBERT`)")
    st.caption(f"**Retrieval Pipeline:** Top {RETRIEVAL_K} retrieved → Top {RERANK_TOP_N} reranked")
    st.caption(f"**Batch Size:** `{BATCH_SIZE}` | **Delay:** `{BATCH_DELAY_SECONDS}s`")

    if existing_sources:
        st.markdown("**📁 Ingested Documents (Click to inspect logs):**")
        for fn in sorted(existing_sources):
            with st.expander(f"📄 {fn}", expanded=False):
                doc_log = get_document_db_inspection(vector_store, fn)
                if doc_log.get("saved_in_db"):
                    st.success("✅ **Vector DB Status:** Saved in ChromaDB")
                    st.markdown(f"- 🧩 **Created Chunks:** `{doc_log['chunk_count']}` chunks")
                    if doc_log.get("has_embeddings"):
                        st.markdown(f"- 🧠 **Embeddings Generated:** ✅ Yes (`{doc_log['embedding_model']}`, {doc_log['embedding_dim']}-dim)")
                    else:
                        st.markdown("- 🧠 **Embeddings Generated:** ✅ Managed by Chroma")
                    st.markdown(f"- 💾 **Collection:** `{doc_log.get('collection_name')}`")
                    st.markdown(f"- 📍 **Pages / Sections:** `{doc_log.get('page_range')}`")
                    st.markdown(f"- 📂 **Storage Path:** `{doc_log.get('db_path')}`")

                    if doc_log.get("sample_chunk"):
                        st.caption("🔍 **First Chunk Content Preview:**")
                        st.text_area(
                            "Chunk #1",
                            value=doc_log["sample_chunk"],
                            height=110,
                            disabled=True,
                            key=f"preview_{abs(hash(fn))}",
                        )
                else:
                    st.error("❌ Document not found in vector database.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Reset DB", use_container_width=True, help="Deletes all vectors from the local database"):
            try:
                vector_store.delete_collection()
            except Exception:
                pass
            load_vector_store.clear()
            st.session_state.messages = []
            st.success("Database reset!")
            st.rerun()


# ==========================================
# Main Chat Interface & Retrieval
# ==========================================
st.header("💬 Study Assistant Chat")
st.caption("Ask questions about your uploaded lectures. Answers are strictly grounded in your materials.")

# Render previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Retrieved & Reranked Sources ({len(message['sources'])} chunks selected from top 10)"):
                for s_idx, src in enumerate(message["sources"], start=1):
                    page_str = f" | Page {src['page']}" if src.get("page") else ""
                    score_str = f" | 🎯 Rerank Score: `{src['score']:.4f}`" if "score" in src and src["score"] is not None else ""
                    st.markdown(f"**Rank [{s_idx}]:** `{src['source']}`{page_str}{score_str}")
                    st.info(src["content"])

# User input
user_query = st.chat_input("Ask a question about your lectures...")

if user_query:
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Check if vector database has any indexed content
        if total_chunks_count == 0:
            notice = (
                "⚠️ No lecture transcripts have been indexed yet. Please upload a PDF or TXT transcript "
                "in the sidebar and click **Ingest & Index Transcripts** first!"
            )
            st.warning(notice)
            st.session_state.messages.append({"role": "assistant", "content": notice, "sources": []})
        else:
            try:
                # 1. Retrieve top 10 chunks from ChromaDB (adaptive for summary vs targeted factual queries)
                if is_summary_request(user_query) and existing_sources:
                    initial_docs = []
                    chunks_per_source = max(1, RETRIEVAL_K // len(existing_sources))
                    for src in sorted(existing_sources):
                        try:
                            docs = vector_store.similarity_search(
                                f"{user_query} main points concepts overview topics discussion",
                                k=chunks_per_source,
                                filter={"source": src},
                            )
                            initial_docs.extend(docs)
                        except Exception:
                            pass
                    # If source-filtered search returned too few, supplement with global search
                    if len(initial_docs) < RETRIEVAL_K:
                        fallback_docs = vector_store.similarity_search(
                            f"{user_query} key concepts topics overview",
                            k=RETRIEVAL_K,
                        )
                        seen = {d.page_content for d in initial_docs}
                        for d in fallback_docs:
                            if d.page_content not in seen and len(initial_docs) < RETRIEVAL_K:
                                initial_docs.append(d)
                                seen.add(d.page_content)
                else:
                    # Standard query: Retrieve top 10 candidate chunks
                    initial_docs = vector_store.similarity_search(user_query, k=RETRIEVAL_K)

                # 2. Rerank the top 10 retrieved chunks using FlashRank cross-encoder
                reranked_pairs = rerank_documents(reranker, user_query, initial_docs, top_n=RERANK_TOP_N)

                # Format context for grounding using the top reranked chunks
                context_parts = []
                sources_data = []
                for idx, (doc, score) in enumerate(reranked_pairs, start=1):
                    src_name = doc.metadata.get("source", "Unknown Document")
                    page_num = doc.metadata.get("page", 1)
                    context_parts.append(
                        f"[Excerpt {idx}] Source: {src_name} (Page {page_num}, Relevance Score: {score:.3f})\nContent:\n{doc.page_content}"
                    )
                    sources_data.append({
                        "source": src_name,
                        "page": page_num,
                        "score": score,
                        "content": doc.page_content,
                    })

                context_str = "\n\n---\n\n".join(context_parts)

                # 3. Build Prompt adhering strictly to grounding rules
                system_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}"

                # Retain recent conversation history (last 4 turns) for context
                conversation_history = []
                for past_msg in st.session_state.messages[:-1][-4:]:
                    if past_msg["role"] == "user":
                        conversation_history.append(HumanMessage(content=past_msg["content"]))
                    elif past_msg["role"] == "assistant":
                        conversation_history.append(AIMessage(content=past_msg["content"]))

                messages = [
                    SystemMessage(content=system_content),
                    *conversation_history,
                    HumanMessage(content=user_query),
                ]

                # 4. Stream response chunk-by-chunk to the UI
                def generate_response_stream() -> Generator[str, None, None]:
                    for chunk in llm.stream(messages):
                        if hasattr(chunk, "content") and chunk.content:
                            yield chunk.content
                        elif isinstance(chunk, str) and chunk:
                            yield chunk

                full_response = st.write_stream(generate_response_stream())

                # 5. Display Citations Expander with Rerank Scores
                if sources_data:
                    with st.expander(f"📚 Retrieved & Reranked Sources (Top {len(sources_data)} selected from {len(initial_docs)} retrieved chunks)"):
                        for s_idx, src in enumerate(sources_data, start=1):
                            page_str = f" | Page {src['page']}" if src.get("page") else ""
                            score_str = f" | 🎯 Rerank Score: `{src['score']:.4f}`" if "score" in src and src["score"] is not None else ""
                            st.markdown(f"**Rank [{s_idx}]:** `{src['source']}`{page_str}{score_str}")
                            st.info(src["content"])

                # 5. Persist assistant response with sources in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources_data,
                })

            except Exception as e:
                error_msg = f"⚠️ An error occurred during retrieval or generation: {e}"
                if "connection" in str(e).lower() or "11434" in str(e):
                    error_msg += (
                        "\n\nPlease ensure Ollama is running locally (`ollama serve`) and the required models "
                        f"are pulled:\n- `ollama pull {EMBEDDING_MODEL}`\n- `ollama pull {LLM_MODEL}`"
                    )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})

    # Clear memory references
    gc.collect()
