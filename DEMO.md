# BharatPolicy AI - Prototype Demo Walkthrough

This guide provides step-by-step instructions to run the complete prototype demo on a clean Windows machine.

## Judge Demo Scripts (Use One)

### 2-Minute Lightning Pitch
- Show backend is online (`/health`) and app is open.
- Ask one budget question on "Smart Budget Chat" and point to citations + confidence label.
- Load a preset profile and show top recommendations with eligibility status.
- Open "Budget Explanation" and click "Generate Impact Summary".
- Download the citizen report and close with: "Actionable, explainable, and fallback-ready."

### 5-Minute Core Demo (Recommended)
- Do everything in the 2-minute flow.
- In "Scheme Recommendations", compare 2 schemes using "Compare Selected".
- Run one What-If simulation by changing income or occupation.
- Highlight action checklists and blockers as policy transparency features.

### 10-Minute Deep Demo
- Do everything in the 5-minute flow.
- Click "Reload Index" in sidebar and explain index health/chunk count.
- Show one invalid-input behavior quickly (for example unsupported language) and explain validation envelope (`ok: false` with error code).
- Close with roadmap: auth, persistence, multilingual expansion.

## Prerequisites

- Python 3.11 installed
- Windows Command Prompt or PowerShell
- Internet connection (for pip installing dependencies)

## Pre-Demo Confidence Check (60 seconds)

Run these before judges join:

```bash
python validate.py
python -c "import requests; print(requests.get('http://127.0.0.1:5000/health', timeout=10).json())"
```

Expected: validation passes and health reports `rag_ready: True` with indexed chunks.

## Step 1: Setup Virtual Environment (2 minutes)

```bash
cd c:\Users\<YourUsername>\Desktop\Bharat_PolicyAI
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**Expected output:**
```
(.venv) C:\Users\...\Bharat_PolicyAI>
```

## Step 2: Install Dependencies (3 minutes)

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed Flask-3.1.0 flask-cors-5.0.0 streamlit-1.45.1 ... sentence-transformers-3.3.1
```

## Step 3: Configure Environment (1 minute)

```bash
copy .env.example .env
```

Edit `.env` and ensure local Ollama settings are valid:
```
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=mistral
```

Then in a separate terminal:
```bash
ollama serve
ollama pull mistral
```

## Step 4: Build Vector Index (1 minute)

```bash
python -m backend.index_builder --input-dir data/raw --index-dir data/index
```

**Expected output:**
```json
{
  "pdf_count": 2,
  "chunk_count": 235,
  "vector_count": 235,
  "storage": "numpy",
  "embedding_backend": "hash"
}
```

Verify index files created:
```bash
ls data\index\
```

Should see: `chunks_meta.json`, `vectors.npy`, `retrieval_config.json`

## Step 5: Start Backend Server (1 minute)

Open a new terminal with venv activated:

```bash
.venv\Scripts\python -m backend.app
```

**Expected output:**
```
INFO:backend.app:Backend ready on 127.0.0.1:5000
WARNING:... (optional deprecation warnings are fine)
```

Keep this terminal running in the background.

## Step 6: Start Frontend (1 minute)

Open another new terminal with venv activated:

```bash
.venv\Scripts\streamlit run frontend\streamlit_app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  URL: http://localhost:8501
```

Browser will auto-open to the Streamlit app.

## Step 7: Test the Demo (5 minutes)

### Page 1: Smart Budget Chat
- Click page "Smart Budget Chat"
- Type question: "What are the main farm schemes?"
- Select language: "English"
- Click "Ask Budget"
- **Expected**: Answer with citations, confidence score, and low-confidence notice when applicable

### Page 2: User Profile
- Click page "User Profile"
- Select preset: "Farmer - Maharashtra"
- Click "Load preset"
- **Expected**: Form fills with profile data (age, income, state, tags)

### Page 3: Scheme Recommendations
- Click page "Scheme Recommendations"
- Click "Find My Schemes"
- **Expected**: List of 3-5 schemes with match score, eligibility status, blockers, and action checklist
- Verify schemes like PM-KISAN, KCC appear for farmer profile

### Page 3A: Compare Schemes
- Select any 2 schemes in "Compare Schemes"
- Click "Compare Selected"
- **Expected**: Side-by-side comparison table with score, eligibility, documents, and links

### Page 3B: What-If Simulator
- Expand "What-If Simulator"
- Change one input (for example income from 140000 to 400000)
- Click "Run What-If"
- **Expected**: Simulated top schemes change based on profile shift

### Page 4: Budget Explanation
- Click page "Budget Explanation"
- Click "Generate Impact Summary"
- **Expected**: Personalized headline, summary, actions, budget context, confidence label, and report download button

### Page 4A: Citizen Report Export
- Click "Download Citizen Report"
- **Expected**: Text report with profile, top schemes, eligibility cues, and citations

### Bonus: Index Reload
- In sidebar, click "Reload Index" button
- **Expected**: "Index reloaded: ready, chunks: 235"

## Judge Talking Track (Short)

- "This is not just search. It gives explainable eligibility, blockers, and next actions."
- "Every answer is source-backed with confidence signaling to reduce misuse risk."
- "Citizens can compare options and run what-if scenarios before applying."
- "The report export creates immediate handoff value for users and field workers."

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` again |
| Backend port 5000 already in use | Change BACKEND_PORT in .env to 5001 |
| Streamlit command not found | Ensure venv is activated (see Step 1) |
| "No index found yet" message | Run Step 4 (build index) |
| No LLM responses (fallback extraction only) | Start Ollama and verify OLLAMA_MODEL is pulled locally |
| Confidence shows low often | Improve query specificity or verify index artifacts/chunk quality |

## No-Internet Fallback Plan (If Live API Fails)

- Keep `EMBEDDING_BACKEND=hash` (default) and run with local Ollama.
- Ask one question and explicitly call out extraction fallback behavior.
- Continue with recommendations, comparison, what-if, and report export (all local).
- Use this line: "Core citizen-assistance functions remain operational offline."

## Performance Notes

- **First load**: ~2-3 seconds (venv + module initialization)
- **Subsequent requests**: <1 second (hash embedding, local retrieval)
- **LLM responses**: 2-5 seconds when API key is provided

## What's Being Demonstrated

1. **RAG (Retrieval-Augmented Generation)**: Budget PDFs → indexed chunks → question embedding → retrieved context → LLM answer
2. **Rule-Based Recommendations**: User profile (age, income, occupation, state, tags) → matched schemes → ranked by score
3. **Eligibility Intelligence**: Per-scheme pass/fail checks, blockers, and action checklists
4. **Decision Support**: Scheme comparison and what-if simulation
5. **Personalized Summaries**: Profile + recommendations → budget-aware summary + exportable citizen report
6. **Graceful Fallbacks**: Works without FAISS, works without sentence-transformers, and falls back to extractive mode when Ollama is unavailable

## Next Steps After Demo

1. Check network connectivity for cloud deployment
2. Review API contract in README.md for integration
3. See ROADMAP.md for post-prototype features (auth, persistence, Hindi i18n)

---

**Demo Duration**: 2, 5, 10, or 15 minutes (choose by judging slot)  
**Backend**: Flask REST API (Python)  
**Frontend**: Streamlit (Python)  
**Retrieval**: Hash embeddings + NumPy (local)  
**LLM**: Ollama (local)
