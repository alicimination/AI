"""Image OCR utilities using PaddleOCR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    text: str
    confidence: float
    lines: List[str]
    engine: str
    error: Optional[str] = None


# ----------------------------------------------------
# Initialize PaddleOCR once (important for performance)
# ----------------------------------------------------
try:
    from paddleocr import PaddleOCR

    paddle_ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False
    )
except Exception as e:
    paddle_ocr = None
    paddle_init_error = str(e)


# ----------------------------------------------------
# Image preprocessing (improves OCR accuracy)
# ----------------------------------------------------
def preprocess_image(image: Image.Image):

    import cv2

    img = np.array(image)

    # convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # increase contrast
    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return img


# ----------------------------------------------------
# OCR extraction
# ----------------------------------------------------
def extract_text_from_image(uploaded_file) -> OCRResult:
    """Extract text from uploaded image using PaddleOCR."""

    if paddle_ocr is None:
        return OCRResult(
            text="",
            confidence=0.0,
            lines=[],
            engine="none",
            error="PaddleOCR failed to initialize. Check paddleocr installation."
        )

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        return OCRResult(
            text="",
            confidence=0.0,
            lines=[],
            engine="none",
            error=str(exc),
        )

    try:
        img = preprocess_image(image)

        result = paddle_ocr.ocr(img)

        lines = []
        confs = []

        for block in result or []:
            for item in block:
                text = item[1][0]
                score = float(item[1][1])

                lines.append(text)
                confs.append(score)

        final_text = "\n".join(lines).strip()
        confidence = sum(confs) / len(confs) if confs else 0.0

        return OCRResult(
            text=final_text,
            confidence=confidence,
            lines=lines,
            engine="paddleocr"
        )

    except Exception as exc:

        return OCRResult(
            text="",
            confidence=0.0,
            lines=[],
            engine="paddleocr",
            error=str(exc),
        )