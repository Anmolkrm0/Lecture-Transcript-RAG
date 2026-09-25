import gc
import os
import re
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

try:
    from evaluation_engine import load_eval_dataset, save_eval_dataset, run_full_evaluation
except ImportError:
    pass

# ==========================================
# Application Configuration & Constants
# ==========================================
# Feature Flags & Configuration
# ==========================================
ENABLE_CITATIONS = True
ENABLE_QUERY_REWRITING = True
ENABLE_RAG_TRACE = True
ENABLE_EVALUATION = True

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
    "EXACT SOURCE CITATION & VERIFICATION RULES:\n"
    "To make it effortless for users to verify and validate every answer against the original lecture transcripts:\n"
    "- IN-TEXT CITATIONS: Whenever stating a key point, definition, or explanation, immediately cite the exact source in brackets, e.g.: `[Source 1: Lecture 4, Page 12, Chunk L4-P12-C38]`.\n"
    "- VERIFIED SOURCES & EVIDENCE: Conclude your response with a dedicated section:\n"
    "  ### 📚 Sources\n"
    "  List each source referenced with:\n"
    "  * [Source N: <Lecture>, Page <Page>, Chunk <Chunk_ID>]: Short verbatim evidence quote from the lecture.\n"
    "- NEVER invent source titles, page numbers, or chunk IDs. Use ONLY the exact metadata given in the Context."
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
    lecture_title = file_name.rsplit(".", 1)[0]
    doc_prefix = re.sub(r"[^A-Za-z0-9]", "", lecture_title)[:6].upper() or "DOC"
    try:
        chunk_counter = 1
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text()
            if text and text.strip():
                page_doc = Document(
                    page_content=text.strip(),
                    metadata={
                        "source": file_name,
                        "lecture": lecture_title,
                        "page": page_idx + 1,
                    },
                )
                page_chunks = splitter.split_documents([page_doc])
                for p_chunk in page_chunks:
                    p_chunk.metadata["chunk_id"] = f"{doc_prefix}-P{page_idx + 1}-C{chunk_counter}"
                    p_chunk.metadata["lecture"] = lecture_title
                    chunks.append(p_chunk)
                    chunk_counter += 1
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
    lecture_title = file_name.rsplit(".", 1)[0]
    doc_prefix = re.sub(r"[^A-Za-z0-9]", "", lecture_title)[:6].upper() or "DOC"
    doc = Document(
        page_content=text.strip(),
        metadata={"source": file_name, "lecture": lecture_title, "page": 1},
    )
    raw_chunks = splitter.split_documents([doc])
    chunks = []
    for c_idx, ch in enumerate(raw_chunks, start=1):
        ch.metadata["chunk_id"] = f"{doc_prefix}-P1-C{c_idx}"
        ch.metadata["lecture"] = lecture_title
        chunks.append(ch)
    del text
    del doc
    del raw_chunks
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


def extract_timestamp(text: str) -> str:
    """Extract VTT/SRT transcript timestamps if present."""
    match = re.search(r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?) --> (\d{2}:\d{2}:\d{2}(?:\.\d+)?)", text)
    if match:
        return f"{match.group(1)} - {match.group(2)}"
    single = re.search(r"(\d{2}:\d{2}:\d{2})", text)
    if single:
        return single.group(1)
    return ""


# ==========================================
# Feature 2: Source Evidence Viewer Dialog
# ==========================================
@st.dialog("Source Evidence")
def show_source_evidence_dialog(item: dict):
    """Display verbatim retrieved chunk text in a focused modal dialog."""
    st.subheader(f"📄 {item.get('lecture', item.get('source', 'Lecture'))}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Page:** {item.get('page', 'N/A')}")
    with col2:
        st.write(f"**Chunk:** `{item.get('chunk_id', 'N/A')}`")
    with col3:
        if item.get("score") is not None:
            st.write(f"**Relevance:** `{item['score']:.4f}`")
    
    st.divider()
    st.markdown("**Exact Retrieved Passage:**")
    st.info(f"\"{item.get('content', '').strip()}\"")

    if item.get("source", "").lower().endswith(".pdf") or item.get("page"):
        st.caption(f"📖 *Open Page {item.get('page', 1)} in `{item.get('source')}`*")

    if st.button("Close", key=f"close_dialog_{item.get('chunk_id', 'btn')}"):
        st.rerun()


def render_citation_ui(sources_data: List[dict], msg_id: str = "cur"):
    """Render interactive citation badges and clickable source evidence viewer."""
    if not sources_data or not ENABLE_CITATIONS:
        return

    st.markdown("##### 📚 Sources")
    st.markdown("---")

    # Display structured source rows with [View Source] button
    for s_idx, src in enumerate(sources_data, start=1):
        lecture_name = src.get("lecture", src.get("source", "Lecture"))
        page_val = src.get("page", 1)
        chunk_val = src.get("chunk_id", f"C-{s_idx}")
        ts_val = src.get("timestamp")
        
        c_meta, c_btn = st.columns([3, 1])
        with c_meta:
            loc_label = f"Page {page_val}" + (f" · ⏱️ {ts_val}" if ts_val else "")
            st.markdown(f"**📄 {lecture_name}**  \n{loc_label} · Chunk `{chunk_val}`")
        with c_btn:
            if st.button("👁️ View Source", key=f"view_src_{msg_id}_{s_idx}_{chunk_val}"):
                show_source_evidence_dialog(src)
        
        if s_idx < len(sources_data):
            st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px solid #333;' />", unsafe_allow_html=True)

    # Detailed expandable view for all passages side-by-side
    with st.expander(f"🔍 Inspect All {len(sources_data)} Retrieved Passages", expanded=False):
        for s_idx, src in enumerate(sources_data, start=1):
            src_name = src.get("source", "Unknown Document")
            page = src.get("page")
            score = src.get("score")
            ts = src.get("timestamp")
            chunk_id = src.get("chunk_id", f"C-{s_idx}")

            badges = []
            if page:
                badges.append(f"📄 Page {page}")
            if ts:
                badges.append(f"⏱️ {ts}")
            if chunk_id:
                badges.append(f"🧩 `{chunk_id}`")
            if score is not None:
                badges.append(f"🎯 Relevance: `{score:.4f}`")

            st.markdown(f"**Rank [{s_idx}] — `{src_name}`** (*{' • '.join(badges)}*)")
            content = src.get("content", "").strip()
            cleaned_content = re.sub(r"^\d+\s*\n\d{2}:\d{2}:\d{2}.*?\n", "", content)
            st.info(f"“{cleaned_content}”")
            if s_idx < len(sources_data):
                st.divider()


# ==========================================
# Feature 3: Query Rewriter
# ==========================================
def needs_query_rewriting(query: str, history_len: int) -> bool:
    """Heuristic check to skip query rewriting when query is standalone."""
    if history_len == 0:
        return False
    q_low = query.lower().strip()
    pronoun_patterns = [
        r"\b(it|its|they|them|their|theirs|this|that|these|those)\b",
        r"\b(former|latter|above|previous|mentioned)\b",
        r"^(and|but|also|so|what about|how about|tell me more|explain more)\b",
    ]
    return any(re.search(pat, q_low) for pat in pronoun_patterns)


def rewrite_query(
    query: str,
    history: List[dict],
    llm: ChatOllama,
) -> Tuple[str, float]:
    """Lightweight conversational query rewriter before retrieval.
    
    Resolves conversational pronouns (e.g. 'What are its limitations?') into standalone queries.
    Falls back to original query if standalone or on error.
    """
    if not ENABLE_QUERY_REWRITING:
        return query, 0.0

    history_len = len([m for m in history if m.get("role") in ("user", "assistant")])
    if not needs_query_rewriting(query, history_len):
        return query, 0.0

    start_t = time.perf_counter()
    try:
        recent_turns = []
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200].replace("\n", " ")
            recent_turns.append(f"{role}: {content}")
        
        history_summary = "\n".join(recent_turns)
        rewrite_prompt = (
            "Given the chat history and follow-up question, rephrase the follow-up question into a "
            "clear, self-contained search query. If the question is already self-contained, return it unchanged. "
            "Output ONLY the rephrased query and nothing else.\n\n"
            f"Chat History:\n{history_summary}\n\n"
            f"Follow-up Question: {query}\n"
            "Standalone Query:"
        )

        resp = llm.invoke(rewrite_prompt)
        rewritten = resp.content.strip().strip('"').strip("'")
        elapsed = round(time.perf_counter() - start_t, 3)

        if rewritten and len(rewritten) > 3 and "\n" not in rewritten:
            return rewritten, elapsed
        return query, elapsed
    except Exception:
        return query, round(time.perf_counter() - start_t, 3)


# ==========================================
# Feature 5: RAG Trace / Debugger Card
# ==========================================
def render_rag_trace_card(trace: dict):
    """Render developer-facing execution trace with real timing, token counts, and candidate pipeline."""
    if not trace:
        return
    with st.expander("⚙️ RAG Execution Trace (Debug Mode)", expanded=True):
        st.markdown(f"**Query Received:** `{trace.get('timestamp')}`")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Original Query:** *\"{trace.get('original_query')}\"*")
        with c2:
            st.markdown(f"**Rewritten Query:** *\"{trace.get('rewritten_query')}\"* ({trace.get('query_rewrite_latency', 0.0)}s)")
        
        st.markdown("---")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            st.metric("Vector Search", f"{trace.get('retrieval_latency', 0.0)}s", f"{trace.get('retrieved_chunks_count', 0)} chunks")
        with t_col2:
            st.metric("FlashRank Rerank", f"{trace.get('rerank_latency', 0.0)}s", f"Top {trace.get('reranked_chunks_count', 0)}")
        with t_col3:
            st.metric("LLM Generation", f"{trace.get('generation_latency', 0.0)}s", f"~{trace.get('output_tokens', 0)} tokens")
        with t_col4:
            st.metric("Total Latency", f"{trace.get('total_latency', 0.0)}s")

        st.caption(
            f"**Embedding Model:** `{trace.get('embedding_model')}` | "
            f"**Reranker:** `{trace.get('reranker_model')}` | "
            f"**LLM:** `{trace.get('llm_model')}` | "
            f"**Context Tokens:** ~`{trace.get('context_tokens')}`"
        )
        if trace.get("sources"):
            st.caption(f"**Retrieved Sources:** {', '.join(trace.get('sources', []))}")


# ==========================================
# Feature 4: RAG Evaluation Dashboard
# ==========================================
def render_evaluation_page(vector_store, reranker, llm, embeddings):
    """Render the interactive RAG Evaluation Dashboard measuring retrieval, generation, citation, and latency."""
    st.header("📊 RAG Evaluation Dashboard")
    st.caption("Measure and benchmark your local RAG system against grounded lecture evaluation questions.")

    if not ENABLE_EVALUATION:
        st.info("Evaluation module is currently disabled via configuration flag `ENABLE_EVALUATION = False`.")
        return

    try:
        eval_data = load_eval_dataset("eval_dataset.json")
    except Exception:
        eval_data = []

    # Benchmark dataset manager
    with st.expander(f"📋 Benchmark Evaluation Dataset ({len(eval_data)} Questions Configured)", expanded=False):
        for item in eval_data:
            st.markdown(f"**[{item['id']}]** *\"{item['question']}\"*")
            st.caption(f"🎯 **Expected Sources:** {', '.join(item.get('expected_sources', []))}")
            st.divider()
        
        # Add new evaluation query form
        with st.form("add_eval_query_form"):
            st.subheader("➕ Add Benchmark Question")
            new_q = st.text_input("Benchmark Question", placeholder="e.g. What is LoRA in fine-tuning?")
            new_srcs = st.text_input("Expected Source Document(s) (comma-separated)", placeholder="e.g. Context Engineering - Session 2")
            submit_eval = st.form_submit_button("Add to Benchmark")
            if submit_eval:
                if new_q.strip() and new_srcs.strip():
                    src_list = [s.strip() for s in new_srcs.split(",") if s.strip()]
                    eval_data.append({
                        "id": f"eval_{len(eval_data) + 1:03d}",
                        "question": new_q.strip(),
                        "expected_sources": src_list,
                    })
                    save_eval_dataset(eval_data, "eval_dataset.json")
                    st.success(f"Added '{new_q.strip()}' to benchmark dataset!")
                    st.rerun()
                else:
                    st.warning("Please provide both a question and expected source(s).")

    # Action button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_eval_clicked = st.button("🚀 Run Full Evaluation", type="primary", use_container_width=True)
    with col_info:
        st.caption("Runs benchmark questions through active local ChromaDB + FlashRank + Qwen2.5:3b pipeline.")

    if run_eval_clicked:
        if not eval_data:
            st.warning("No benchmark questions found in `eval_dataset.json`.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_cb(current, total, q_text):
                progress_bar.progress(current / total)
                status_text.write(f"Evaluating ({current}/{total}): *\"{q_text[:60]}...\"*")

            with st.spinner("Executing benchmark evaluation across local models..."):
                results = run_full_evaluation(
                    dataset=eval_data,
                    vector_store=vector_store,
                    reranker=reranker,
                    llm=llm,
                    embeddings=embeddings,
                    retrieval_k=RETRIEVAL_K,
                    top_n=RERANK_TOP_N,
                    progress_callback=progress_cb,
                )
                st.session_state["eval_results"] = results
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Evaluation Complete!")

    # Display results
    eval_results = st.session_state.get("eval_results")

    st.markdown("### 📈 Evaluation Summary Metrics")
    st.markdown("---")

    if eval_results and "avg_recall" in eval_results:
        # 1. Retrieval Metrics
        st.subheader("🔍 Retrieval Metrics")
        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.metric("Recall@5", f"{eval_results['avg_recall'] * 100:.1f}%", help="Proportion of queries where expected source was retrieved")
        with r_c2:
            st.metric("Precision@5", f"{eval_results['avg_precision'] * 100:.1f}%", help="Proportion of retrieved chunks matching expected sources")
        with r_c3:
            st.metric("MRR", f"{eval_results['avg_mrr']:.3f}", help="Mean Reciprocal Rank of the first relevant chunk")

        st.markdown("---")

        # 2. Generation Metrics
        st.subheader("🧠 Generation Metrics")
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.metric("Faithfulness (Grounding)", f"{eval_results['avg_faithfulness'] * 100:.1f}%", help="Proportion of answer statements verified against retrieved context")
        with g_c2:
            st.metric("Answer Relevance", f"{eval_results['avg_answer_relevance'] * 100:.1f}%", help="Semantic similarity between question and generated answer")

        st.markdown("---")

        # 3. Citation Metrics
        st.subheader("📚 Citation Metrics")
        c_c1, c_c2 = st.columns(2)
        with c_c1:
            st.metric("Citation Coverage", f"{eval_results['avg_citation_coverage'] * 100:.1f}%", help="Proportion of answers containing explicit citations")
        with c_c2:
            st.metric("Citation Accuracy", f"{eval_results['avg_citation_accuracy'] * 100:.1f}%", help="Proportion of cited sources matching retrieved passages")

        st.markdown("---")

        # 4. Performance Metrics
        st.subheader("⚡ Hardware & Latency Performance")
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            st.metric("Avg Total Latency", f"{eval_results['avg_total_latency']:.2f}s")
        with p_c2:
            st.metric("Avg Retrieval Latency", f"{eval_results['avg_retrieval_latency']:.3f}s")
        with p_c3:
            st.metric("Avg Generation Latency", f"{eval_results['avg_generation_latency']:.2f}s")
        with p_c4:
            st.metric("Avg Context Tokens", f"~{eval_results['avg_context_tokens']}")

        # Detailed Results Table
        st.markdown("---")
        st.subheader("📝 Detailed Query-by-Query Results")
        for d in eval_results.get("detailed_results", []):
            with st.expander(f"Query: \"{d['question']}\" — Recall: {d['recall'] * 100:.0f}% | Total: {d['total_latency']}s"):
                st.markdown(f"**Expected Sources:** `{', '.join(d['expected_sources'])}`")
                st.markdown(f"**Retrieved Sources:** `{', '.join(d['retrieved_sources'])}`")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.write(f"**Recall@5:** {d['recall']}")
                with c2:
                    st.write(f"**Precision@5:** {d['precision']}")
                with c3:
                    st.write(f"**Faithfulness:** {d['faithfulness']}")
                with c4:
                    st.write(f"**Relevance:** {d['answer_relevance']}")
                st.markdown("**Generated Answer Preview:**")
                st.caption(d.get("answer_preview", ""))

    else:
        # Display "Not available" for uncalculated metrics as requested in specification
        st.subheader("🔍 Retrieval")
        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.metric("Recall@5", "Not available")
        with r_c2:
            st.metric("Precision@5", "Not available")
        with r_c3:
            st.metric("MRR", "Not available")

        st.subheader("🧠 Generation")
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.metric("Faithfulness", "Not available")
        with g_c2:
            st.metric("Answer Relevance", "Not available")

        st.subheader("📚 Citations")
        c_c1, c_c2 = st.columns(2)
        with c_c1:
            st.metric("Citation Coverage", "Not available")
        with c_c2:
            st.metric("Citation Accuracy", "Not available")

        st.subheader("⚡ Performance")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.metric("Average Latency", "Not available")
        with p_c2:
            st.metric("Context Tokens", "Not available")

        st.info("ℹ️ Click **🚀 Run Full Evaluation** above to calculate live benchmark metrics using your local models.")



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
                doc_prefix = re.sub(r"[^A-Za-z0-9]", "", title)[:6].upper() or "PASTE"
                with st.status(f"⚡ Ingesting '{title}'...", expanded=True) as status_box:
                    status_box.write("📄 **Step 1:** Parsing pasted transcript text...")
                    raw_doc = Document(
                        page_content=pasted_text.strip(),
                        metadata={"source": title, "lecture": title, "page": 1},
                    )
                    raw_chunks = text_splitter.split_documents([raw_doc])
                    chunks = []
                    for c_idx, ch in enumerate(raw_chunks, start=1):
                        ch.metadata["chunk_id"] = f"{doc_prefix}-P1-C{c_idx}"
                        ch.metadata["lecture"] = title
                        chunks.append(ch)
                    del raw_doc
                    del raw_chunks
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
    active_nav = st.radio(
        "Navigation",
        ["💬 Chat Assistant", "📊 RAG Evaluation"],
        index=0,
        key="main_navigation_choice",
    )
    enable_debugger = st.checkbox(
        "⚙️ Enable RAG Debugger",
        value=False,
        help="Display developer trace: latencies, candidate stages, and query rewriting.",
        key="enable_rag_debugger_toggle",
    )

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
# Main Interface: Chat Assistant vs Evaluation
# ==========================================
if active_nav == "📊 RAG Evaluation":
    render_evaluation_page(vector_store, reranker, llm, embeddings)
else:
    # ------------------------------------------
    # Main Chat Interface & Retrieval
    # ------------------------------------------
    st.header("💬 Study Assistant Chat")
    st.caption("Ask questions about your uploaded lectures. Answers are strictly grounded in your materials.")

    # Render previous messages from history
    for m_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_citation_ui(message["sources"], msg_id=f"hist_{m_idx}")
            if message["role"] == "assistant" and message.get("trace") and enable_debugger:
                render_rag_trace_card(message["trace"])

    # User input
    user_query = st.chat_input("Ask a question about your lectures...")

    if user_query:
        # Append and display user message (preserving original query in chat history)
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if total_chunks_count == 0:
                notice = (
                    "⚠️ No lecture transcripts have been indexed yet. Please upload a PDF or TXT transcript "
                    "in the sidebar and click **Ingest & Index Transcripts** first!"
                )
                st.warning(notice)
                st.session_state.messages.append({"role": "assistant", "content": notice, "sources": []})
            else:
                try:
                    turn_start_time = time.perf_counter()

                    # 0. Query Rewriting (Lightweight conversational resolution)
                    retrieval_query, rewrite_latency = rewrite_query(
                        query=user_query,
                        history=st.session_state.messages[:-1],
                        llm=llm,
                    )

                    # 1. Retrieval (ChromaDB top 10 candidates)
                    retrieval_start = time.perf_counter()
                    if is_summary_request(retrieval_query) and existing_sources:
                        initial_docs = []
                        chunks_per_source = max(1, RETRIEVAL_K // len(existing_sources))
                        for src in sorted(existing_sources):
                            try:
                                docs = vector_store.similarity_search(
                                    f"{retrieval_query} main points concepts overview topics discussion",
                                    k=chunks_per_source,
                                    filter={"source": src},
                                )
                                initial_docs.extend(docs)
                            except Exception:
                                pass
                        if len(initial_docs) < RETRIEVAL_K:
                            fallback_docs = vector_store.similarity_search(
                                f"{retrieval_query} key concepts topics overview",
                                k=RETRIEVAL_K,
                            )
                            seen = {d.page_content for d in initial_docs}
                            for d in fallback_docs:
                                if d.page_content not in seen and len(initial_docs) < RETRIEVAL_K:
                                    initial_docs.append(d)
                                    seen.add(d.page_content)
                    else:
                        initial_docs = vector_store.similarity_search(retrieval_query, k=RETRIEVAL_K)
                    retrieval_latency = time.perf_counter() - retrieval_start

                    # 2. FlashRank Cross-Encoder Reranking
                    rerank_start = time.perf_counter()
                    reranked_pairs = rerank_documents(reranker, retrieval_query, initial_docs, top_n=RERANK_TOP_N)
                    rerank_latency = time.perf_counter() - rerank_start

                    # Format context for grounding using the top reranked chunks
                    context_parts = []
                    sources_data = []
                    for idx, (doc, score) in enumerate(reranked_pairs, start=1):
                        src_name = doc.metadata.get("source", "Unknown Document")
                        lecture_name = doc.metadata.get("lecture") or src_name.rsplit(".", 1)[0]
                        page_num = doc.metadata.get("page", 1)
                        ts = extract_timestamp(doc.page_content)
                        chunk_id = doc.metadata.get("chunk_id")
                        if not chunk_id:
                            doc_prefix = re.sub(r"[^A-Za-z0-9]", "", str(lecture_name))[:6].upper() or "DOC"
                            chunk_id = f"{doc_prefix}-P{page_num}-C{idx}"

                        loc_desc = f"Page {page_num}"
                        if ts:
                            loc_desc += f", Time: {ts}"

                        context_parts.append(
                            f"--- [Source {idx}: Lecture: {lecture_name}, Page: {page_num}, Chunk: {chunk_id}, File: {src_name}] ---\n"
                            f"{doc.page_content}"
                        )
                        sources_data.append({
                            "source": src_name,
                            "lecture": lecture_name,
                            "page": page_num,
                            "chunk_id": chunk_id,
                            "score": score,
                            "timestamp": ts,
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
                    gen_start = time.perf_counter()
                    def generate_response_stream() -> Generator[str, None, None]:
                        for chunk in llm.stream(messages):
                            if hasattr(chunk, "content") and chunk.content:
                                yield chunk.content
                            elif isinstance(chunk, str) and chunk:
                                yield chunk

                    full_response = st.write_stream(generate_response_stream())
                    generation_latency = time.perf_counter() - gen_start
                    total_latency = time.perf_counter() - turn_start_time

                    # 5. Build structured execution trace
                    trace_data = {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "original_query": user_query,
                        "rewritten_query": retrieval_query,
                        "query_rewrite_latency": round(rewrite_latency, 3),
                        "embedding_model": EMBEDDING_MODEL,
                        "retrieval_latency": round(retrieval_latency, 3),
                        "retrieved_chunks_count": len(initial_docs),
                        "reranker_model": "FlashRank (ms-marco-TinyBERT)",
                        "rerank_latency": round(rerank_latency, 3),
                        "reranked_chunks_count": len(reranked_pairs),
                        "selected_chunks_count": len(sources_data),
                        "context_tokens": len(context_str) // 4,
                        "llm_model": LLM_MODEL,
                        "generation_latency": round(generation_latency, 3),
                        "output_tokens": len(full_response) // 4,
                        "total_latency": round(total_latency, 3),
                        "sources": [s.get("source") for s in sources_data],
                    }

                    # 6. Display Interactive Citations and Clickable Evidence Viewer
                    render_citation_ui(sources_data, msg_id="latest")

                    # 7. Display RAG Trace / Debugger if enabled
                    if enable_debugger:
                        render_rag_trace_card(trace_data)

                    # 8. Persist assistant response with sources and trace in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources_data,
                        "trace": trace_data,
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
