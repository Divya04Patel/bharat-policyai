# BharatPolicy AI - Samjho Budget. Paao Benefits. Har Din.

Hackathon prototype to help citizens understand budget updates and discover relevant schemes.

## Quick Start

1. Create and activate a Python 3.11 virtual environment.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `OLLAMA_MODEL` (`mistral` or `llama3` recommended).
4. Ensure Ollama is running locally and the model is pulled:

```bash
ollama serve
ollama pull mistral
```
5. Add budget PDFs to `data/raw/`.
6. Build vector index (local hash embeddings by default):

```bash
python -m backend.index_builder --input-dir data/raw --index-dir data/index
```

7. Run backend:

```bash
python -m backend.app
```

8. Run frontend:

```bash
streamlit run frontend/streamlit_app.py
```

## Notes

- The default embedding backend is `hash`, which runs locally without heavy model downloads.
- For higher-quality embeddings, set `EMBEDDING_BACKEND=sentence-transformers`.
- If `faiss-cpu` is installed, indexing and retrieval automatically use FAISS.
- Without FAISS, the app falls back to local NumPy vector retrieval (`vectors.npy`).
- Indexing writes `embedding_cache.npz` to avoid recomputing chunk embeddings across rebuilds.
- If Ollama is unavailable, backend returns extractive fallback answers from retrieved chunks.

## MVP Features

- Budget Q&A with citations
- Confidence-aware answers with low-confidence notice
- Profile-based scheme recommendations
- Eligibility diagnostics and action checklist per scheme
- Scheme comparison (top selections side-by-side)
- What-if simulator (profile scenario testing)
- Personalized budget impact summary
- Downloadable citizen benefit report (text export)
- Bilingual-friendly simple responses

## API Contract

All backend endpoints return a consistent envelope:

- Success: `{ "ok": true, ...data }`
- Error: `{ "ok": false, "error": { "code": "VALIDATION_ERROR", "message": "..." } }`

### Endpoints

- `GET /health`
	- Returns backend status, RAG readiness, indexed chunk count, and loaded scheme count.

- `POST /api/ask-budget`
	- Request: `{ "question": "...", "language": "en" | "hi" }`
	- Validation: `question` is required, `language` must be `en` or `hi`.

- `POST /api/recommend-schemes`
	- Request: `{ "profile": {...}, "top_n": 1-10 }`
	- Notes: if `profile` is omitted, top-level payload is treated as profile.
	- Includes `eligibility` diagnostics and `action_checklist` per scheme.

- `POST /api/compare-schemes`
	- Request: `{ "scheme_ids": ["SCH001", "SCH002"], "profile": {...} }`
	- Validation: 1-5 `scheme_ids` supported.
	- Returns side-by-side comparison payload with score and eligibility outcomes.

- `POST /api/impact-summary`
	- Request: `{ "profile": {...}, "recommendations": [...] }`
	- Validation: `recommendations` must be an array when provided.
	- Notes: if `recommendations` is missing, backend computes top matches.
	- Includes confidence metadata and export-ready `report` payload.

- `POST /api/reload-index`
	- Refreshes index artifacts from disk and reports readiness.

## Known Prototype Limits

- No authentication or rate limiting.
- Session data is in-memory on Streamlit side.
- Hindi mode is prompt-guided, not full translation/i18n.
