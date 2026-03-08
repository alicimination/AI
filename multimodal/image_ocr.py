"""Image OCR utilities with robust free OCR backends and math-aware extraction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass
class OCRResult:
    text: str
    confidence: float
    lines: List[str]
    engine: str
    error: Optional[str] = None


paddle_ocr = None
paddle_init_error = None
try:
    from paddleocr import PaddleOCR

    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
except Exception as exc:  # noqa: BLE001
    paddle_init_error = str(exc)


pytesseract_mod = None
tesseract_init_error = None
try:
    import pytesseract

    pytesseract_mod = pytesseract
except Exception as exc:  # noqa: BLE001
    tesseract_init_error = str(exc)


rapid_ocr = None
rapid_init_error = None
try:
    from rapidocr_onnxruntime import RapidOCR

    rapid_ocr = RapidOCR()
except Exception as exc:  # noqa: BLE001
    rapid_init_error = str(exc)


def _roi_crops(image: Image.Image) -> List[Image.Image]:
    """Generate ROIs so equations lower in image are not missed."""

    w, h = image.size
    rois = [image]

    # lower area where equations are often placed
    rois.append(image.crop((0, int(0.28 * h), w, h)))
    # central equation strip
    rois.append(image.crop((0, int(0.38 * h), w, int(0.9 * h))))

    return rois


def _preprocess_variants(image: Image.Image) -> List[np.ndarray]:
    gray = ImageOps.grayscale(image)
    base = ImageOps.autocontrast(gray)
    denoised = base.filter(ImageFilter.MedianFilter(size=3))

    arr = np.array(denoised)

    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg = 0.0
    weight_bg = 0.0
    max_between = -1.0
    threshold = 127

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = t

    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    inverted = (255 - binary).astype(np.uint8)

    # dilation-like smoothing to reconnect thin math symbols after binarization
    binary_img = Image.fromarray(binary).filter(ImageFilter.MaxFilter(size=3))

    w, h = denoised.size
    upscaled = denoised.resize((int(w * 1.6), int(h * 1.6)), Image.Resampling.LANCZOS)

    return [
        np.array(base),
        np.array(denoised),
        np.array(binary_img),
        inverted,
        np.array(upscaled),
    ]


def _clean_lines(lines: Sequence[str]) -> List[str]:
    cleaned = [ln.strip() for ln in lines if isinstance(ln, str) and ln.strip()]
    return list(dict.fromkeys(cleaned))


def _is_equation_like(text: str) -> bool:
    if not text:
        return False
    operators = sum(ch in "=+-*/^()[]{}" for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    variables = sum(ch.isalpha() for ch in text)
    return operators >= 2 and (digits + variables) >= 3


def _quality_score(text: str, confidence: float) -> float:
    if not text:
        return 0.0

    visible_chars = sum(ch.isalnum() or ch in "+-*/=()[]{}^.,:<>" for ch in text)
    density = visible_chars / max(len(text), 1)
    length_bonus = min(len(text) / 140.0, 1.0)
    eq_bonus = 0.15 if _is_equation_like(text) else 0.0
    return (confidence * 0.58) + (density * 0.2) + (length_bonus * 0.07) + eq_bonus


def _ocr_with_paddle(variants: Sequence[np.ndarray]) -> Optional[OCRResult]:
    if paddle_ocr is None:
        return None

    best: Optional[OCRResult] = None
    best_score = -1.0

    for variant in variants:
        result = paddle_ocr.ocr(variant)
        lines: List[str] = []
        confs: List[float] = []

        for block in result or []:
            for item in block:
                text = item[1][0]
                score = float(item[1][1])
                lines.append(text)
                confs.append(score)

        lines = _clean_lines(lines)
        text = "\n".join(lines)
        confidence = sum(confs) / len(confs) if confs else 0.0
        score = _quality_score(text, confidence)

        if score > best_score:
            best_score = score
            best = OCRResult(text=text, confidence=confidence, lines=lines, engine="paddleocr")

    return best


def _ocr_with_tesseract(variants: Sequence[np.ndarray]) -> Optional[OCRResult]:
    if pytesseract_mod is None:
        return None

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 7",
        "--oem 3 --psm 13",
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()[]{}^.:, ",
    ]

    best: Optional[OCRResult] = None
    best_score = -1.0

    for variant in variants:
        for config in configs:
            # Build line-wise output (better for equations than word list)
            lines_by_row: defaultdict[Tuple[int, int, int], List[str]] = defaultdict(list)
            confs: List[float] = []

            try:
                data = pytesseract_mod.image_to_data(
                    variant,
                    output_type=pytesseract_mod.Output.DICT,
                    config=config,
                )
            except Exception:  # noqa: BLE001
                data = None

            if data:
                n = len(data.get("text", []))
                for i in range(n):
                    token = (data["text"][i] or "").strip()
                    if not token:
                        continue

                    key = (
                        int(data.get("block_num", [0])[i]),
                        int(data.get("par_num", [0])[i]),
                        int(data.get("line_num", [0])[i]),
                    )
                    lines_by_row[key].append(token)

                    try:
                        conf = float(data.get("conf", ["-1"])[i])
                    except (TypeError, ValueError):
                        conf = -1.0
                    if conf >= 0:
                        confs.append(conf / 100.0)

            line_joined = [" ".join(tokens) for _, tokens in sorted(lines_by_row.items())]

            # Additional plain OCR pass catches some missed math layouts.
            try:
                raw_text = pytesseract_mod.image_to_string(variant, config=config)
            except Exception:  # noqa: BLE001
                raw_text = ""

            raw_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            lines = _clean_lines([*line_joined, *raw_lines])

            text = "\n".join(lines)
            confidence = sum(confs) / len(confs) if confs else 0.0
            score = _quality_score(text, confidence)

            if score > best_score:
                best_score = score
                best = OCRResult(text=text, confidence=confidence, lines=lines, engine="tesseract")

    return best


def _ocr_with_rapidocr(variants: Sequence[np.ndarray]) -> Optional[OCRResult]:
    if rapid_ocr is None:
        return None

    best: Optional[OCRResult] = None
    best_score = -1.0

    for variant in variants:
        result, _ = rapid_ocr(variant)
        lines: List[str] = []
        confs: List[float] = []

        for item in result or []:
            if len(item) >= 3:
                text = str(item[1])
                try:
                    conf = float(item[2])
                except (TypeError, ValueError):
                    conf = 0.0
                if text.strip():
                    lines.append(text)
                    confs.append(conf)

        lines = _clean_lines(lines)
        text = "\n".join(lines)
        confidence = sum(confs) / len(confs) if confs else 0.0
        score = _quality_score(text, confidence)

        if score > best_score:
            best_score = score
            best = OCRResult(text=text, confidence=confidence, lines=lines, engine="rapidocr")

    return best


def _pick_best_result(candidates: Sequence[OCRResult]) -> OCRResult:
    ranked: List[Tuple[float, OCRResult]] = [(_quality_score(c.text, c.confidence), c) for c in candidates]
    ranked.sort(key=lambda x: x[0], reverse=True)
    best = ranked[0][1]

    # Preserve equation-like lines from other engines/ROIs if best misses them.
    eq_lines = []
    for _, cand in ranked:
        eq_lines.extend([ln for ln in cand.lines if _is_equation_like(ln)])
    eq_lines = _clean_lines(eq_lines)

    if eq_lines:
        merged_lines = _clean_lines([*best.lines, *eq_lines])
        return OCRResult(
            text="\n".join(merged_lines),
            confidence=best.confidence,
            lines=merged_lines,
            engine=best.engine,
        )

    if len(ranked) > 1:
        alt = ranked[1][1]
        if best.text and alt.text and best.text != alt.text:
            merged_lines = _clean_lines([*best.lines, *alt.lines])
            merged_text = "\n".join(merged_lines)
            if len(merged_text) > len(best.text):
                return OCRResult(
                    text=merged_text,
                    confidence=max(best.confidence, alt.confidence),
                    lines=merged_lines,
                    engine=f"{best.engine}+{alt.engine}",
                )

    return best


def extract_text_from_image(uploaded_file) -> OCRResult:
    """Extract text/equations from uploaded image with free OCR backends."""

    if paddle_ocr is None and rapid_ocr is None and pytesseract_mod is None:
        details = []
        if paddle_init_error:
            details.append(f"PaddleOCR: {paddle_init_error}")
        if tesseract_init_error:
            details.append(f"pytesseract: {tesseract_init_error}")
        if rapid_init_error:
            details.append(f"RapidOCR: {rapid_init_error}")
        error_msg = "No OCR engine available."
        if details:
            error_msg = f"{error_msg} {' | '.join(details)}"
        return OCRResult(text="", confidence=0.0, lines=[], engine="none", error=error_msg)

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return OCRResult(text="", confidence=0.0, lines=[], engine="none", error=str(exc))

    try:
        candidates: List[OCRResult] = []

        for roi in _roi_crops(image):
            variants = _preprocess_variants(roi)

            paddle_res = _ocr_with_paddle(variants)
            if paddle_res and paddle_res.text:
                candidates.append(paddle_res)

            rapid_res = _ocr_with_rapidocr(variants)
            if rapid_res and rapid_res.text:
                candidates.append(rapid_res)

            tesseract_res = _ocr_with_tesseract(variants)
            if tesseract_res and tesseract_res.text:
                candidates.append(tesseract_res)

        if not candidates:
            init_errors = []
            if paddle_init_error:
                init_errors.append(f"PaddleOCR init: {paddle_init_error}")
            if tesseract_init_error:
                init_errors.append(f"pytesseract init: {tesseract_init_error}")
            if rapid_init_error:
                init_errors.append(f"RapidOCR init: {rapid_init_error}")
            tail = f" {' | '.join(init_errors)}" if init_errors else ""
            return OCRResult(
                text="",
                confidence=0.0,
                lines=[],
                engine="none",
                error=f"OCR completed but no text was detected.{tail}",
            )

        return _pick_best_result(candidates)

    except Exception as exc:  # noqa: BLE001
        return OCRResult(text="", confidence=0.0, lines=[], engine="none", error=str(exc))
