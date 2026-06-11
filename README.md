# Career RAG System

An AI-powered career guidance chatbot using Retrieval-Augmented Generation. Ask natural language questions about careers, skills, education, and occupations — the system retrieves structured data from O\*NET and ESCO, then generates a grounded answer using a local LLM running entirely on your machine.

---

## Architecture

```
User
 └─► React Frontend (Vite)
       └─► FastAPI Backend  (/chat-stream)
             └─► Query Router  (FAISS semantic similarity)
                   ├─► O*NET Retriever  (skills, tasks, education, related occupations)
                   └─► ESCO Retriever   (European competences & occupations)
                         └─► LLM  (Phi-3-mini-4k-instruct, runs locally)
                               └─► Streamed response  (word-by-word)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite |
| Backend API | FastAPI + Uvicorn |
| Semantic Router | FAISS + Sentence Transformers |
| LLM | microsoft/Phi-3-mini-4k-instruct (local, offline) |
| Datasets | O\*NET (US), ESCO (European) |
| Embeddings | Sentence Transformers |
| Language | Python 3.10+ / Node.js 18+ |

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
│   └── esco_retriever.py         # ESCO data retrieval
│
├── llm/
│   ├── generate_response.py      # LLM inference + response formatting
│   └── education_data.py         # Education requirement lookups
│
├── embeddings/                   # Scripts to build FAISS indexes
├── indexes/                      # FAISS index files (ESCO + O*NET)
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

- Python 3.10 or higher
- Node.js 18 or higher
- ~8 GB free disk space (for the Phi-3 model)
- A Hugging Face account (for the one-time model download)

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
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

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
API_HOST=127.0.0.1
API_PORT=8000
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
The user's message is embedded using a sentence-transformer model. That embedding is compared against every FAISS index (O\*NET, ESCO) using L2 distance converted to a similarity score. The highest-scoring source is selected — no hardcoded keywords or intent labels. To add a new dataset, add one entry to `INDEX_REGISTRY` in `router/query_router.py`.

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

---

## Known Limitations

- No conversation memory — each message is treated independently
- LLM runs on CPU by default; GPU support requires manual configuration
- ESCO data is English-only in the current build
- Response quality depends on whether the occupation exists in the indexed data

---

## License

This project is for educational and research purposes.
