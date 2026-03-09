"""Audio speech-to-text utilities with Whisper."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tempfile
import wave

import numpy as np

# --- FFMPEG INJECTION START ---
try:
    import imageio_ffmpeg
    
    # 1. Get the weirdly named ffmpeg executable from the package
    ffmpeg_original_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    # 2. Create a local folder in your project just for this binary
    local_bin_dir = os.path.join(os.getcwd(), "ffmpeg_bin")
    os.makedirs(local_bin_dir, exist_ok=True)
    
    # 3. Whisper strictly looks for "ffmpeg.exe" (or "ffmpeg" on Mac/Linux)
    is_windows = os.name == "nt"
    local_ffmpeg_exe = os.path.join(local_bin_dir, "ffmpeg.exe" if is_windows else "ffmpeg")
    
    # 4. Copy and rename the file so Whisper can find it
    if not os.path.exists(local_ffmpeg_exe):
        shutil.copy(ffmpeg_original_path, local_ffmpeg_exe)
        if not is_windows:
            os.chmod(local_ffmpeg_exe, 0o755)  # Make it executable for Mac/Linux users
            
    # 5. Add our custom folder to the VERY FRONT of the system PATH
    os.environ["PATH"] = f"{local_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    
except ImportError:
    pass  # If imageio_ffmpeg isn't installed, fall back to default behavior
# --- FFMPEG INJECTION END ---


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


def _decode_wav_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes without ffmpeg, returning float32 mono waveform."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with wave.open(tmp_path, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            pcm_bytes = wav_file.readframes(n_frames)

        if sample_width == 1:
            audio = np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        return audio

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def transcribe_audio(uploaded_file, model_size: str = "base") -> ASRResult:
    """Transcribe audio input with Whisper."""

    if uploaded_file is None:
        return ASRResult(
            transcript="",
            confidence=0.0,
            engine="none",
            error="No audio provided",
        )

    file_name = getattr(uploaded_file, "name", "") or ""
    suffix = Path(file_name).suffix.lower() or ".wav"
    file_bytes = uploaded_file.getvalue()

    try:
        import whisper

        model = whisper.load_model(model_size)

        # Fast path for direct microphone WAV (avoids ffmpeg dependency).
        if suffix == ".wav":
            try:
                waveform = _decode_wav_bytes(file_bytes)
                if not waveform.flags['C_CONTIGUOUS']:
                    waveform = np.ascontiguousarray(waveform)
                result = model.transcribe(waveform, fp16=False)
            except Exception:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file_bytes)
                    audio_path = tmp.name

                try:
                    result = model.transcribe(audio_path)
                finally:
                    Path(audio_path).unlink(missing_ok=True)

        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                audio_path = tmp.name

            try:
                result = model.transcribe(audio_path)
            finally:
                Path(audio_path).unlink(missing_ok=True)

        transcript = result.get("text", "").strip()
        confidence = 0.8 if transcript else 0.0
        normalized = normalize_math_phrases(transcript)

        return ASRResult(
            transcript=normalized,
            confidence=confidence,
            engine="whisper",
        )

    except FileNotFoundError as exc:  # pragma: no cover
        if "WinError 2" in str(exc) or "ffmpeg" in str(exc).lower():
            return ASRResult(
                transcript="",
                confidence=0.0,
                engine="whisper",
                error="FFmpeg is missing. Install FFmpeg or use direct microphone WAV recording.",
            )

        return ASRResult(
            transcript="",
            confidence=0.0,
            engine="whisper",
            error=str(exc),
        )

    except Exception as exc:  # pragma: no cover
        return ASRResult(
            transcript="",
            confidence=0.0,
            engine="whisper",
            error=str(exc),
        )
