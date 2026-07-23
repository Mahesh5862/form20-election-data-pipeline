"""
STEP 7 - ENHANCE FAILED ROWS (retry branch only - v1/v2/v3 belongs ONLY here)

Input   : data/failed_rows/{pdf_name}/page_N/booth_ID_raw.png
Output  : data/failed_rows/{pdf_name}/page_N/booth_ID_v1.png
          data/failed_rows/{pdf_name}/page_N/booth_ID_v2.png
          data/failed_rows/{pdf_name}/page_N/booth_ID_v3.png

Three different enhancement settings for the SAME row, so Step 8 can try
all three against Gemini and keep whichever passes validation. Cheap here
because it only runs on the ~100 rows that failed Step 5, never every row
of every page.

v1 - moderate contrast + upscale (works well for most faint-print cases)
v2 - aggressive binarization + larger upscale (works well for smudged/thick ink)
v3 - gentle grayscale-preserving enhancement (works well for uneven scan lighting)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import logging

from config import FAILED_ROWS_DIR, PDF_NAMES, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "07_enhance_failed_rows.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def upscale(img, scale_x, scale_y):
    return cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)


def variant_1(img_bgr):
    """Moderate contrast + upscale - good general-purpose default."""
    up = upscale(img_bgr, 3, 4)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)
    denoised = cv2.bilateralFilter(contrasted, d=5, sigmaColor=50, sigmaSpace=50)
    return denoised


def variant_2(img_bgr):
    """Aggressive binarization + larger upscale - good for smudged/thick ink."""
    up = upscale(img_bgr, 4, 5)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )
    return thresh


def variant_3(img_bgr):
    """Gentle grayscale-preserving enhancement - good for uneven scan lighting."""
    up = upscale(img_bgr, 3, 3)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    return clahe.apply(denoised)


def process_pdf(pdf_name: str):
    base_dir = FAILED_ROWS_DIR / pdf_name

    if not base_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 06_extract_failed_rows.py first")
        return

    logger.info(f"{'=' * 70}")
    logger.info(f"ENHANCING FAILED ROWS: {pdf_name}")
    logger.info(f"{'=' * 70}")

    raw_files = sorted(base_dir.glob("page_*/booth_*_raw.png"))
    total = 0

    for raw_path in raw_files:
        img_bgr = cv2.imread(str(raw_path))
        if img_bgr is None:
            logger.error(f"  Could not read {raw_path}")
            continue

        base_name = raw_path.stem.replace("_raw", "")

        for i, variant_fn in enumerate([variant_1, variant_2, variant_3], start=1):
            variant_img = variant_fn(img_bgr)
            out_path = raw_path.parent / f"{base_name}_v{i}.png"
            cv2.imwrite(str(out_path), variant_img)

        total += 1

    logger.info(f"[{pdf_name}] Done - {total} rows x 3 variants enhanced")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)