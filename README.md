# Reliable Multimodal Math Mentor

A **production-style AI tutoring system** designed to solve **JEE-level mathematics problems** using **multimodal input**, **multi-agent reasoning**, **retrieval-augmented generation (RAG)**, and **symbolic mathematics tools**.

The system supports **text, image, and audio inputs**, processes them through a structured **AI reasoning pipeline**, verifies the solution, and generates **student-friendly explanations**.

---

# Overview

Reliable Multimodal Math Mentor is built to demonstrate a **robust AI architecture for math tutoring** rather than a simple LLM wrapper.

The system combines:

* Multimodal input processing
* Structured problem parsing
* Topic-aware reasoning
* Knowledge-grounded retrieval
* Symbolic math execution
* Automated verification
* Human-in-the-loop validation
* Persistent learning via memory

This architecture aims to **reduce hallucinations**, improve **solution reliability**, and provide **clear educational explanations**.

---

# System Architecture

```
User Input (Text / Image / Audio)
        │
        ▼
OCR / ASR Extraction
        │
        ▼
User Confirmation Layer
        │
        ▼
Parser Agent
(Structured Problem Representation)
        │
        ▼
Intent Router Agent
(Topic & Strategy Selection)
        │
        ▼
Solver Agent
(RAG + SymPy Math Tool)
        │
        ▼
Verifier Agent
(Logic + Consistency Checks)
        │
        ▼
Explainer Agent
(Student-Friendly Step-by-Step Explanation)
        │
        ▼
Human-in-the-Loop Gate
        │
        ▼
Memory Storage (SQLite)
        │
        ▼
Similarity Search & Pattern Reuse
```

---

# Project Structure

```
math-mentor/
│
├── app.py
│
├── agents/
│   ├── parser_agent.py
│   ├── intent_router.py
│   ├── solver_agent.py
│   ├── verifier_agent.py
│   └── explainer_agent.py
│
├── rag/
│   ├── ingest.py
│   ├── retriever.py
│   └── vector_store.py
│
├── multimodal/
│   ├── image_ocr.py
│   └── audio_asr.py
│
├── memory/
│   ├── memory_store.py
│   └── similarity_search.py
│
├── tools/
│   └── python_math_tool.py
│
├── hitl/
│   └── hitl_manager.py
│
├── knowledge_base/
│   ├── algebra.md
│   ├── calculus.md
│   ├── probability.md
│   ├── linear_algebra.md
│   ├── pitfalls.md
│   ├── combinatorics.md
│   ├── sequences_series.md
│   ├── trigonometry_basics.md
│   ├── domain_constraints.md
│   └── jee_problem_solving_patterns.md
│
├── utils/
│   ├── prompts.py
│   └── logging.py
│
├── requirements.txt
└── README.md
```

---

# Key Features

## Multimodal Problem Input

The system accepts math problems through multiple modalities.

### Text Input

Users can directly type a math problem into the interface.

### Image Input

* Multi-pass OCR using **PaddleOCR**
* Fallback/ensemble with **Tesseract**
* Extracted text is shown in an **editable confirmation box**

### Audio Input

* Speech-to-text using **Whisper**
* Math phrase normalization (example: "x squared" → `x^2`)
* User confirmation before processing

---

# Parser Agent

The parser converts raw input into a **structured mathematical problem representation**.

Example output:

```json
{
  "problem_text": "Solve x^2 + 2x = 0",
  "topic": "algebra",
  "variables": ["x"],
  "constraints": [],
  "needs_clarification": false
}
```

This structured format allows downstream agents to operate **reliably and consistently**.

---

# Retrieval-Augmented Generation (RAG)

The system retrieves relevant theory and strategies from a **local math knowledge base**.

### Knowledge Sources

The knowledge base includes curated markdown documents covering:

* Algebra
* Calculus
* Probability
* Linear Algebra
* Combinatorics
* Sequences and Series
* Trigonometry
* Domain Constraints
* Common Pitfalls
* JEE Problem-Solving Patterns

### Retrieval Pipeline

1. Markdown documents are chunked
2. Chunks are embedded using **sentence-transformers**
3. Embeddings are stored in a **FAISS vector database**
4. The top **4 most relevant chunks** are retrieved during solving

The UI displays retrieved sources to maintain **transparency and grounding**.

If no sources are retrieved, the system explicitly indicates that **no supporting source was found**.

---

# Multi-Agent Reasoning Pipeline

The solving pipeline consists of multiple specialized agents:

### Parser Agent

Transforms raw input into structured problem data.

### Intent Router Agent

Determines the **topic and solving strategy**.

### Solver Agent

Uses:

* retrieved knowledge (RAG)
* symbolic math execution via **SymPy**

### Verifier Agent

Checks:

* logical consistency
* symbolic correctness
* potential uncertainty

If confidence is low, the system triggers **human-in-the-loop review**.

### Explainer Agent

Transforms the solution into **clear, student-friendly steps** suitable for learning.

---

# Human-in-the-Loop (HITL)

The system includes safety gates that pause the pipeline when reliability may be compromised.

Triggers include:

* Low OCR confidence
* Low ASR transcription confidence
* Parser ambiguity
* Verifier uncertainty
* User marking an answer as incorrect

Users can edit extracted text or request re-evaluation.

---

# Memory and Self-Learning

Solved problems are stored in a **SQLite memory database**.

Stored data includes:

* original input
* parsed problem structure
* retrieved context
* generated solution
* verification results
* user feedback
* timestamp

### Similarity Search

When a new problem is submitted, the system searches for **similar past problems** to reuse solving patterns.

### OCR Correction Learning

User edits to OCR outputs are stored to improve future processing.

---

# Installation

Clone the repository and install dependencies.

```
git clone <repository-url>
cd math-mentor
```

Create a virtual environment:

```
python -m venv .venv
```

Activate it:

Mac/Linux:

```
source .venv/bin/activate
```

Windows:

```
.venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Note: `sqlite3` is included in Python and does not require installation.

---

# Optional System Dependencies

Some multimodal features may require additional system tools.

### Tesseract

Used as an OCR fallback.

Install from:

https://github.com/tesseract-ocr/tesseract

### FFmpeg

Recommended for audio preprocessing when using Whisper.

---

# Running the Application

Start the Streamlit interface:

```
streamlit run app.py
```

The app will open in your browser.

---

# Deployment

The application can be deployed on:

* Streamlit Cloud
* HuggingFace Spaces (Streamlit SDK)
* Any Python hosting environment

### Deployment Steps

1. Push the repository to GitHub
2. Ensure `requirements.txt` installs successfully
3. Set the entrypoint to:

```
app.py
```

---

# Design Goals

The project prioritizes:

* Reliability over raw generation
* Knowledge-grounded reasoning
* Clear educational explanations
* Modular AI architecture
* Extensibility for future research

---

# Future Improvements

Possible extensions include:

* LaTeX rendering for equations
* Interactive whiteboard input
* diagram interpretation
* step-by-step verification using symbolic proofs
* performance optimization for large-scale deployment
* improved UI for classroom use

---

# License

Add a license file if distributing publicly (MIT recommended for research projects).

---

# Acknowledgements

This project builds on several open-source tools:

* Streamlit
* Sentence Transformers
* FAISS
* PaddleOCR
* Tesseract
* Whisper
* SymPy
