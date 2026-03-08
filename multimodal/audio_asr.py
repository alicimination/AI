"""Audio speech-to-text utilities with Whisper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import tempfile


@dataclass
class ASRResult:
    transcript: str
    confidence: float
    engine: str
    error: Optional[str] = None


MATH_PHRASE_MAP = {
    "raised to power": "^",
    "to the power of": "^",
    "square root": "sqrt",
    "divided by": "/",
    "probability of": "P(",
}


def normalize_math_phrases(text: str) -> str:
    """Normalize spoken math phrases into symbolic hints."""
    normalized = text.lower()
    for phrase, token in MATH_PHRASE_MAP.items():
        normalized = normalized.replace(phrase, token)
    return normalized


def transcribe_audio(uploaded_file, model_size: str = "base") -> ASRResult:
    """Transcribe audio input with Whisper."""
    if uploaded_file is None:
        return ASRResult(transcript="", confidence=0.0, engine="none", error="No audio provided")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            audio_path = tmp.name

        import whisper

        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        transcript = result.get("text", "").strip()
        confidence = 0.8 if transcript else 0.0
        normalized = normalize_math_phrases(transcript)
        return ASRResult(transcript=normalized, confidence=confidence, engine="whisper")
    except Exception as exc:  # pragma: no cover
        return ASRResult(transcript="", confidence=0.0, engine="whisper", error=str(exc))
