# 🎓 Lecture Transcript AI

An advanced, local Retrieval-Augmented Generation (RAG) study assistant engineered specifically for **Apple Silicon Macs** (M1/M2/M3/M4). It runs 100% locally with zero cloud API dependencies, optimized for memory management so it never freezes or hangs the operating system.

---

## 🌟 Key Features

- **⚡ Apple Silicon & Unified Memory Optimization:**
  - Enforces batch embedding (32 chunks/batch) with sleep delays and explicit garbage collection (`gc.collect()`).
  - Resource caching via `@st.cache_resource` to prevent redundant memory allocations.
  - Total system RAM footprint remains under **3.0 GB**.
- **📥 Dual-Mode Document Ingestion:**
  - **Upload Files:** Ingest multi-page PDFs and text files page-by-page using `PyMuPDF`.
  - **Direct Paste:** Paste raw transcripts, YouTube subtitles, or lecture notes directly with custom titles.
- **📋 Live Ingestion & Inspection Logs:**
  - Real-time step-by-step progress logs during ingestion (`st.status`).
  - Click-to-inspect document cards in the sidebar displaying created chunks, embedding verification (768-dim `nomic-embed-text`), ChromaDB storage status, and chunk previews.
- **🔍 Adaptive Dual-Route Retrieval:**
  - **Targeted Mode:** Standard top-4 vector similarity search for specific factual lookups.
  - **Global / Study Mode:** Automatically detects requests for summaries, mindmaps, reports, quizzes, and flashcards, sampling representative concepts across **all** uploaded lectures.
- **🧠 Rich Study Artifacts:**
  - **High-Level Mindmaps:** Hierarchical text trees (`├──`, `└──`).
  - **Study Reports:** Executive summaries, core pillars, and takeaways.
  - **Quizzes & MCQs:** Grounded 4-option multiple-choice questions with answer keys and explanations.
  - **Flashcards:** Front/Back format for rapid conceptual review.
- **🛡️ Strict Grounding & Anti-Hallucination:**
  - Strictly limited to provided lecture context.
  - Non-domain and external trivia questions are politely rejected with: *"I couldn't find this information in your uploaded lectures."*
- **🌊 Word-by-Word Response Streaming:**
  - Real-time token streaming via `st.write_stream` with collapsible citation panels.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | [Streamlit](https://streamlit.io/) v1.50 | Interactive UI with reactive chat, sidebar tabs, and status monitors |
| **Orchestration** | [LangChain](https://www.langchain.com/) v0.3 | Prompt engineering, message pipelines, and text splitters |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) v1.5.9 | Embedded local persistent vector database (`./chroma_db`) |
| **Document Parser** | [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`) | Page-by-page PDF extraction with fast C-level memory release |
| **Embeddings** | [Ollama](https://ollama.com/) `nomic-embed-text` | 768-dimensional dense vector embeddings (~274 MB RAM) |
| **LLM** | [Ollama](https://ollama.com/) `qwen2.5:3b` | 3.09B parameter instruction-tuned LLM (~2.0 GB RAM) |
| **Acceleration** | Apple Metal (MPS) | Native Apple Silicon GPU execution via `llama-server` |

---

## 🚀 Getting Started

### 1. Prerequisites
- macOS on an Apple Silicon Mac (M1/M2/M3/M4 recommended)
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running

### 2. Pull Required Models
Ensure the local Ollama server is running, then pull the required models:

```bash
# Pull the 768-dim embedding model (~274 MB)
ollama pull nomic-embed-text

# Pull the 3B instruction-tuned LLM (~2.0 GB)
ollama pull qwen2.5:3b
```

### 3. Installation
Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/<your-username>/lecture-transcript-ai.git
cd lecture-transcript-ai

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Application
Launch the Streamlit app:

```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 📂 Project Structure

```text
├── app.py                # Complete single-file application code
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes venv, cache, and local ChromaDB files
└── README.md             # Project documentation
```

---

## 💡 Example Queries to Try

- **Summarization:** *"Summarize all the lectures"*
- **Mindmap:** *"Create a high level mindmap of the sessions"*
- **Quiz:** *"Generate 3 MCQs with answers based on the lectures"*
- **Flashcards:** *"Create 4 flashcards for key terms"*
- **Targeted Question:** *"What is the GenAI value stack and what are its layers?"*
- **Out-of-Domain Guardrail:** *"What is the capital of France?"* $\rightarrow$ *"I couldn't find this information in your uploaded lectures."*

---

## 📄 License
MIT License
