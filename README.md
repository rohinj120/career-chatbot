# Career RAG System

An AI-powered career guidance chatbot using Retrieval-Augmented Generation. Ask natural language questions about careers, skills, education, and occupations — the system retrieves structured data from O\*NET, Canada dataset and ESCO, then generates a grounded answer using a local LLM running entirely on your machine.

---

## Architecture

```
User
 └─► React Frontend (Vite)
       └─► FastAPI Backend  (/chat-stream)
             └─► Query Router  (FAISS semantic similarity)
                   ├─► O*NET Retriever  (skills, tasks, education, related occupations)
                   ├─► ESCO Retriever   (European competences & occupations)
                   └─► NOC Retriever    (Canadian work descriptors & abilities)
                         └─► LLM  (Phi-3-mini-4k-instruct, runs locally)
                               └─► Streamed response  (word-by-word)
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Frontend | React | 19.2.6 |
| Frontend | Vite | 8.0.12 |
| Frontend | socket.io-client | 4.8.3 |
| Frontend | react-markdown | 10.1.0 |
| Backend API | FastAPI | 0.136.3 |
| Backend API | Uvicorn | 0.48.0 |
| Backend API | python-socketio | 5.12.1 |
| Semantic Router | FAISS | faiss-cpu 1.14.2 |
| Embeddings | sentence-transformers | 5.5.1 |
| LLM Framework | Transformers | 5.9.0 |
| LLM Framework | Torch | 2.12.0 |
| LLM Model | microsoft/Phi-3-mini-4k-instruct | local, offline |
| Datasets | O\*NET (US), ESCO (EU), NOC (Canada) | — |
| Observability | Langfuse | latest |
| Language | Python | 3.13 / 3.14 |
| Language | Node.js | 18+ |

---

## Project Structure

```
career_rag_system/
├── api.py                        # FastAPI app — /chat-stream endpoint
├── chatbot.py                    # Main pipeline: route → retrieve → generate
├── start.ps1                     # One-command launcher (backend + frontend)
│
├── router/
│   └── query_router.py           # Dynamic semantic router (FAISS-based, no hardcoded intents)
│
├── retrievers/
│   ├── onet_retriever.py         # O*NET data retrieval
│   ├── esco_retriever.py         # ESCO data retrieval
│   └── noc_retriever.py          # NOC (Canada) descriptors retrieval
│
├── llm/
│   ├── generate_response.py      # LLM inference + response formatting
│   └── education_data.py         # Education requirement lookups
│
├── embeddings/                   # Scripts to build FAISS indexes
├── indexes/                      # FAISS index files (ESCO + O*NET + NOC)
├── metadata/                     # Metadata pickles for index entries
├── models/
│   └── Phi-3-mini-4k-instruct/   # Local model weights (7.64 GB)
│
├── data/                         # Raw and cleaned dataset files
├── core/
│   └── embedder.py               # Shared sentence-transformer encoder
│
└── career-chatbot-frontend/      # React frontend
    └── src/
        └── App.jsx               # Chat UI with streaming support
```

---

## Prerequisites

- Python 3.13 or higher
- Node.js 18 or higher
- ~8 GB free disk space (for the Phi-3 model)
- A Hugging Face account (for the one-time model download)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rohinj120/career-chatbot.git
cd career_rag_system
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd career-chatbot-frontend
npm install
cd ..
```

### 4. Download the LLM

```bash
python download_phi3.py
```

This downloads `microsoft/Phi-3-mini-4k-instruct` (~7.64 GB) into `models/Phi-3-mini-4k-instruct/`. Only needs to be done once.

### 5. Configure backend environment variables

Create a `.env` file in the **project root**:

```env
# API server
API_HOST=127.0.0.1
API_PORT=8000

# Anthropic (if using Claude instead of local Phi-3)
ANTHROPIC_API_KEY=your_api_key_here

# Langfuse observability
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 6. Configure frontend environment variables

Create a `.env` file inside **`career-chatbot-frontend/`**:

```env
VITE_API_URL=http://localhost:8000
```

---

## Running the App

### Windows (recommended)

```powershell
.\start.ps1
```

This opens two terminal windows — one for the FastAPI backend and one for the React frontend.

### Manual start

**Backend:**
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

**Frontend:**
```bash
cd career-chatbot-frontend
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.

---

## How It Works

### Query Routing
The user's message is embedded using a sentence-transformer model. That embedding is compared against every FAISS index (O\*NET, ESCO, NOC) using cosine similarity. The highest-scoring source is selected — no hardcoded keywords or intent labels. To add a new dataset, add one entry to `INDEX_REGISTRY` in `router/query_router.py`.

### Retrieval
Depending on the routed source, the relevant retriever fetches structured career data — job tasks, required skills, education levels, related occupations, and competences.

### Generation
The retrieved context is passed to Phi-3-mini-4k-instruct, running fully locally via `transformers`. No external API calls are made at inference time.

### Streaming
FastAPI returns a `StreamingResponse` that yields words with a small delay. The React frontend consumes the stream and renders tokens in real time as they arrive.

---

## Datasets

| Dataset | Coverage | Data Includes |
|---|---|---|
| O\*NET | United States | Skills, tasks, abilities, work styles, education, wages |
| ESCO | European Union | Occupations, skills, competences, multilingual labels |
| NOC | Canada | Work descriptors, abilities, cognitive & physical skill categories |

---

## Known Limitations

- No conversation memory — each message is treated independently
- LLM runs on CPU by default; GPU support requires manual configuration
- ESCO data is English-only in the current build
- Response quality depends on whether the occupation exists in the indexed data

---

## License

This project is for educational and research purposes.
