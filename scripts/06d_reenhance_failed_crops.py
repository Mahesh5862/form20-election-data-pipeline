"""
STEP 6D - AGGRESSIVE RE-ENHANCEMENT (only for still-failing booths)

Input   : data/validation_reports/{pdf_name}/manual_review_needed.json
          data/tables/{pdf_name}/page_N_table.png
          data/raw_json/{pdf_name}/page_N_raw.json (for booth order on page)
Output  : data/failed_rows/{pdf_name}/page_N/booth_ID_enhanced2.png
          logs/06d_reenhance_failed_crops.log

WHY THIS EXISTS: the shifted-crop variants (06c) fixed row-POSITION
problems and recovered 46 booths. Booths that got ZERO improvement from
shifting likely have a DIFFERENT problem - the crop position is already
correct, but the digits themselves are too faint/blurry to read. This
script applies stronger contrast (CLAHE) and sharpening to the same row
position, producing one additional variant that Step 8 will try and
self-verify exactly like v1/v2/v3 and the shift variants.

SPEED FIX: dropped fastNlMeansDenoising (extremely slow on wide table
images - was causing the script to appear stuck) and reduced upscale from
3x to 2x. CLAHE + sharpening alone still meaningfully boosts faint digits
without the denoising bottleneck.

ONLY processes booths currently in manual_review_needed.json.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import logging

from config import (
    VALIDATION_REPORTS_DIR, TABLES_DIR, RAW_JSON_DIR,
    FAILED_ROWS_DIR, PDF_NAMES, LOG_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "06d_reenhance_failed_crops.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def detect_row_boundaries(img_gray):
    thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_gray.shape[1] // 2, 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)

    row_sums = np.sum(horiz_lines, axis=1)
    threshold = row_sums.max() * 0.5
    line_ys = [y for y in range(len(row_sums)) if row_sums[y] > threshold]

    boundaries = []
    for y in line_ys:
        if not boundaries or y - boundaries[-1] > 5:
            boundaries.append(y)
    return boundaries


def get_ordered_booth_ids(pdf_name: str, page_num: int) -> list:
    raw_path = RAW_JSON_DIR / pdf_name / f"page_{page_num}_raw.json"
    if not raw_path.exists():
        return []
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(b["booth_id"]).strip() for b in data.get("booths", [])]


def base_row_bounds(boundaries, row_position: int, header_rows: int,
                     total_booths_on_page: int, img_h: int):
    row_index = header_rows + row_position
    if 0 <= row_index and row_index + 1 < len(boundaries):
        return boundaries[row_index], boundaries[row_index + 1]

    if total_booths_on_page > 0:
        header_height = boundaries[header_rows] if len(boundaries) > header_rows else 0
        usable_height = img_h - header_height
        row_height = usable_height / total_booths_on_page
        y1 = int(header_height + row_position * row_height)
        y2 = int(header_height + (row_position + 1) * row_height)
        return y1, y2

    return None, None


def aggressive_enhance(row_img):
    """Stronger contrast/sharpening than the standard Step 3/7 enhancement -
    aimed at genuinely faint digits, not boundary issues. No slow denoising
    step - tables are extremely wide, denoising at that scale was hanging."""
    gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)

    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)

    result = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def load_manual_review_list(pdf_name: str) -> list:
    path = VALIDATION_REPORTS_DIR / pdf_name / "manual_review_needed.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("needs_manual_review", [])


def process_pdf(pdf_name: str, header_rows: int = 1):
    review_list = load_manual_review_list(pdf_name)
    if not review_list:
        logger.info(f"[{pdf_name}] No manual_review_needed.json entries - skipping")
        return

    logger.info(f"{'=' * 70}")
    logger.info(f"AGGRESSIVE RE-ENHANCEMENT: {pdf_name} ({len(review_list)} booths)")
    logger.info(f"{'=' * 70}")

    by_page = {}
    for item in review_list:
        by_page.setdefault(item["page"], []).append(str(item["booth_id"]).strip())

    total_created = 0

    for page_num, booth_ids in by_page.items():
        table_path = TABLES_DIR / pdf_name / f"page_{page_num}_table.png"
        if not table_path.exists():
            logger.warning(f"  Missing table image for page {page_num}")
            continue

        table_img = cv2.imread(str(table_path))
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        boundaries = detect_row_boundaries(gray)
        img_h = table_img.shape[0]

        ordered_ids = get_ordered_booth_ids(pdf_name, page_num)
        total_booths_on_page = len(ordered_ids)

        out_dir = FAILED_ROWS_DIR / pdf_name / f"page_{page_num}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for booth_id in booth_ids:
            if booth_id not in ordered_ids:
                continue
            row_position = ordered_ids.index(booth_id)

            y1, y2 = base_row_bounds(boundaries, row_position, header_rows,
                                      total_booths_on_page, img_h)
            if y1 is None:
                continue

            y1 = max(0, y1)
            y2 = min(img_h, y2)
            if y2 <= y1:
                continue

            row_img = table_img[y1:y2, :]
            enhanced = aggressive_enhance(row_img)

            out_path = out_dir / f"booth_{booth_id}_enhanced2.png"
            cv2.imwrite(str(out_path), enhanced)
            total_created += 1

    logger.info(f"[{pdf_name}] Done - {total_created} aggressively re-enhanced crops created")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)