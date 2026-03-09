"""Reliable Multimodal Math Mentor - Streamlit app entry point."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from agents.explainer_agent import ExplainerAgent
from agents.intent_router import IntentRouterAgent
from agents.parser_agent import ParserAgent
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from hitl.hitl_manager import HITLManager
from memory.memory_store import MemoryStore
from memory.similarity_search import MemorySimilarity
from multimodal.audio_asr import transcribe_audio
from multimodal.image_ocr import extract_text_from_image
from rag.ingest import ingest_knowledge_base


st.set_page_config(page_title="Reliable Multimodal Math Mentor", page_icon="🧠", layout="wide")
st.title("🧠 Reliable Multimodal Math Mentor")
st.caption("RAG + Multi-agent + Multimodal + HITL + Memory")


@st.cache_resource
def init_system():
    if not Path("rag/faiss.index").exists():
        ingest_knowledge_base("knowledge_base")
    return {
        "parser": ParserAgent(),
        "router": IntentRouterAgent(),
        "solver": SolverAgent(),
        "verifier": VerifierAgent(),
        "explainer": ExplainerAgent(),
        "hitl": HITLManager(),
        "memory": MemoryStore(),
        "similarity": MemorySimilarity(),
    }


system = init_system()

if "user_requested_recheck" not in st.session_state:
    st.session_state.user_requested_recheck = False

mode = st.sidebar.selectbox("Input Mode", ["Text", "Image", "Audio"])
st.sidebar.markdown("### Pipeline Trace")
trace_box = st.sidebar.empty()
trace_lines = []
if st.sidebar.button("🔁 Request Re-check (HITL)"):
    st.session_state.user_requested_recheck = True
    st.sidebar.info("Re-check request recorded. It will trigger HITL on next solve.")

raw_text = ""
ocr_conf = None
asr_conf = None

if mode == "Text":
    raw_text = st.text_area("Enter JEE-style math problem", height=140)

elif mode == "Image":
    img = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if img:
        st.image(img, caption="Uploaded Image", use_column_width=True)
        ocr_result = extract_text_from_image(img)
        ocr_conf = ocr_result.confidence
        trace_lines.append(f"OCR ({ocr_result.engine}) confidence: {ocr_result.confidence:.2f}")
        if ocr_result.error:
            st.error(f"OCR error: {ocr_result.error}")
        raw_text = st.text_area("OCR Extraction (editable)", value=ocr_result.text, height=160)

elif mode == "Audio":
    aud = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    if aud:
        st.audio(aud)
        asr_result = transcribe_audio(aud)
        asr_conf = asr_result.confidence
        trace_lines.append(f"ASR ({asr_result.engine}) confidence: {asr_result.confidence:.2f}")
        if asr_result.error:
            st.error(f"ASR error: {asr_result.error}")
        raw_text = st.text_area("Transcript (confirm/edit)", value=asr_result.transcript, height=160)

if st.button("Solve Problem", type="primary"):
    if not raw_text.strip():
        st.warning("Please provide a problem statement first.")
        st.stop()

    corrected_text, applied_rules = system["memory"].apply_correction_rules(raw_text)
    if applied_rules:
        trace_lines.append(f"Applied {len(applied_rules)} stored correction rule(s)")

    parsed = system["parser"].run(corrected_text).to_dict()
    trace_lines.append("Parser Agent → completed")

    route = system["router"].run(parsed["topic"])
    trace_lines.append(f"Intent Router → {route.topic} ({route.strategy})")

    similar = system["similarity"].find_similar(raw_text, top_k=2)

    solver_out = system["solver"].run(parsed, route.strategy)
    trace_lines.append(f"RAG retrieved {len(solver_out.retrieved_context)} docs")
    trace_lines.append("Solver Agent → completed")

    verify_out = system["verifier"].run(parsed, solver_out.final_answer)
    trace_lines.append("Verifier Agent → completed")

    hitl = system["hitl"].evaluate(
        ocr_conf=ocr_conf,
        asr_conf=asr_conf,
        parser_needs_clarification=parsed["needs_clarification"],
        verifier_uncertain=verify_out.needs_hitl,
        user_requested_recheck=st.session_state.user_requested_recheck,
    )

    explainer_out = system["explainer"].run(parsed, solver_out.plan, solver_out.steps, solver_out.final_answer)
    trace_lines.append("Explainer Agent → completed")
    trace_box.markdown("\n".join(f"- {line}" for line in trace_lines))

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Structured Parse")
        st.json(parsed)
        if applied_rules:
            st.caption("Applied correction rules before parsing:")
            for rule in applied_rules:
                st.write(f"- `{rule['incorrect_text']}` → `{rule['corrected_text']}`")

        st.subheader("Retrieved Context")
        if solver_out.retrieved_context:
            for item in solver_out.retrieved_context:
                with st.expander(f"{item['source']} | score={item['score']:.3f}"):
                    st.write(item["content"])
        else:
            st.info("No retrieval results found; no citations generated.")

        if similar:
            st.subheader("Similar Past Problems (Memory)")
            for rec in similar:
                st.write(f"- Similarity {rec['similarity']:.3f}: {rec['original_input'][:140]}")

    with col2:
        st.subheader("Final Answer")
        st.success(solver_out.final_answer)

        st.subheader("Step-by-step Explanation")
        st.markdown(f"```\n{explainer_out.explanation}\n```")

        st.metric("Confidence", f"{verify_out.confidence:.2f}")
        st.write("Verification checks:")
        for check in verify_out.checks:
            st.write(f"- {check}")

        if hitl.required:
            st.warning("HITL required before trusting final output.")
            st.write("Reasons:")
            for reason in hitl.reasons:
                st.write(f"- {reason}")

        st.subheader("Feedback")
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            ok = st.button("✅ Correct")
        with feedback_col2:
            bad = st.button("❌ Incorrect")
        comment = st.text_input("Optional correction / comment")

        feedback = ""
        if ok:
            feedback = "correct"
        if bad:
            feedback = f"incorrect: {comment}" if comment else "incorrect"

        if feedback:
            if mode == "Image" and raw_text and raw_text != parsed["problem_text"]:
                system["memory"].add_ocr_correction(raw_text, parsed["problem_text"])
            system["memory"].add_record(
                original_input=raw_text,
                parsed_problem=parsed,
                retrieved_context=solver_out.retrieved_context,
                solution=json.dumps(
                    {
                        "final_answer": solver_out.final_answer,
                        "plan": solver_out.plan,
                        "steps": solver_out.steps,
                    }
                ),
                verification_result={
                    "passed": verify_out.passed,
                    "confidence": verify_out.confidence,
                    "checks": verify_out.checks,
                    "needs_hitl": verify_out.needs_hitl,
                },
                user_feedback=feedback,
            )
            st.success("Feedback saved to memory.")

    if st.session_state.user_requested_recheck:
        st.info("Re-check request has been honored in this run.")
        st.session_state.user_requested_recheck = False

st.divider()
with st.expander("System Notes"):
    st.write("- Run `streamlit run app.py` to start locally.")
    st.write("- To rebuild KB vectors manually: `python -m rag.ingest` (or call ingest_knowledge_base).")
