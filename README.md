# Reliable Multimodal Math Mentor

Production-style Streamlit application for **JEE-style math solving** with:
- Multimodal input (**Text / Image / Audio**)
- Multi-agent reasoning pipeline
- Retrieval-Augmented Generation (RAG)
- Human-in-the-loop controls
- Persistent memory + similarity reuse

## Architecture

```text
Input (Text/Image/Audio)
  -> OCR/ASR extraction + user confirmation
  -> Parser Agent (structured JSON)
  -> Intent Router Agent (topic/strategy)
  -> Solver Agent (RAG + SymPy tool)
  -> Verifier Agent (checks + uncertainty)
  -> Explainer Agent (student-friendly steps)
  -> HITL gate + feedback
  -> SQLite memory + similarity reuse
```

## Project Structure

```text
math-mentor/
├── app.py
├── agents/
│   ├── parser_agent.py
│   ├── intent_router.py
│   ├── solver_agent.py
│   ├── verifier_agent.py
│   └── explainer_agent.py
├── rag/
│   ├── ingest.py
│   ├── retriever.py
│   └── vector_store.py
├── multimodal/
│   ├── image_ocr.py
│   └── audio_asr.py
├── memory/
│   ├── memory_store.py
│   └── similarity_search.py
├── tools/
│   └── python_math_tool.py
├── hitl/
│   └── hitl_manager.py
├── knowledge_base/
│   ├── algebra.md
│   ├── calculus.md
│   ├── probability.md
│   ├── linear_algebra.md
│   └── pitfalls.md
├── utils/
│   ├── prompts.py
│   └── logging.py
├── requirements.txt
└── README.md
```

## Key Features

### 1) Multimodal Input
- **Text**: direct problem input.
- **Image**: multi-pass OCR via PaddleOCR + Tesseract fallback/ensemble, editable extraction box.
- **Audio**: Whisper transcription + math phrase normalization + user confirmation.

### 2) Parser Agent Output
Produces structured JSON:

```json
{
  "problem_text": "...",
  "topic": "probability",
  "variables": ["x"],
  "constraints": ["x > 0"],
  "needs_clarification": false
}
```

### 3) RAG
- KB markdown docs → chunking → sentence-transformer embeddings → FAISS storage.
- Retrieval top-k = 4.
- UI shows retrieved sources and chunk content.
- If no retrieval, app explicitly states no source found (no fabricated citations).

### 4) Multi-agent Pipeline
1. Parser Agent
2. Intent Router Agent
3. Solver Agent
4. Verifier Agent
5. Explainer Agent

### 5) HITL Triggers
- Low OCR confidence
- Low ASR confidence
- Parser ambiguity
- Verifier uncertainty
- User can recheck via incorrect feedback path

### 6) Memory + Self-learning
SQLite stores:
- original_input
- parsed_problem
- retrieved_context
- solution
- verification_result
- user_feedback
- timestamp

Similarity search retrieves similar solved problems to reuse patterns.
OCR corrections are also stored.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: `sqlite3` is part of Python standard library and does not need separate installation.

### Optional system dependencies
- Tesseract binary (if using pytesseract fallback)
- FFmpeg (recommended for Whisper audio handling)

## Run

```bash
streamlit run app.py
```

## Deployment

Compatible with:
- Streamlit Cloud
- HuggingFace Spaces (Streamlit SDK)

For cloud deployment:
1. Push repository.
2. Set entrypoint to `app.py`.
3. Ensure `requirements.txt` installs successfully.

## Notes
- For best OCR/ASR quality, use clear images and noise-free audio.
- Solver has symbolic automation and fallback explanations; verifier + HITL protects reliability.
