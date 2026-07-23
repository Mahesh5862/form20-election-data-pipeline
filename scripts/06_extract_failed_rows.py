"""
STEP 6 - EXTRACT FAILED ROWS (retry branch only)

Input   : data/failed_booths/{pdf_name}/page_N_failed.json
          data/raw_json/{pdf_name}/page_N_raw.json (for full booth order on the page)
          data/tables/{pdf_name}/page_N_table.png (the un-enhanced crop)
Output  : data/failed_rows/{pdf_name}/page_N/booth_ID_raw.png

booth_id is a STRING (e.g. "117A"). Row position on the page is found by
looking up where that booth_id sits in the page's full ordered booth list
(from raw_json), NOT by subtracting IDs - this works correctly even when
IDs include letter suffixes.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import logging

from config import (
    FAILED_BOOTHS_DIR, TABLES_DIR, RAW_JSON_DIR,
    FAILED_ROWS_DIR, PDF_NAMES, LOG_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "06_extract_failed_rows.log", encoding="utf-8"),
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
    """Returns booth_ids in the exact order they appear on the page (top to bottom),
    as strings - this is the source of truth for row position, since IDs like
    '117A' can't be used in arithmetic to find position."""
    raw_path = RAW_JSON_DIR / pdf_name / f"page_{page_num}_raw.json"
    if not raw_path.exists():
        return []
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(b["booth_id"]).strip() for b in data.get("booths", [])]


def crop_booth_row(table_img, boundaries, row_position: int, header_rows: int,
                    total_booths_on_page: int):
    """row_position is the 0-indexed position of this booth in the page's
    ordered booth list (from get_ordered_booth_ids), NOT the booth_id itself."""
    row_index = header_rows + row_position
    img_h = table_img.shape[0]

    if 0 <= row_index and row_index + 1 < len(boundaries):
        y1 = boundaries[row_index]
        y2 = boundaries[row_index + 1]
        if y2 > y1:
            return table_img[y1:y2, :]

    # Fallback: uniform row height across the page's actual booth count
    if total_booths_on_page > 0:
        header_height = boundaries[header_rows] if len(boundaries) > header_rows else 0
        usable_height = img_h - header_height
        row_height = usable_height / total_booths_on_page
        y1 = int(header_height + row_position * row_height)
        y2 = int(header_height + (row_position + 1) * row_height)
        y1 = max(0, y1)
        y2 = min(img_h, y2)
        if y2 > y1:
            return table_img[y1:y2, :]

    return None


def process_pdf(pdf_name: str, header_rows: int = 1):
    failed_dir = FAILED_BOOTHS_DIR / pdf_name
    table_dir = TABLES_DIR / pdf_name
    out_base = FAILED_ROWS_DIR / pdf_name

    if not failed_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 05_validate.py first")
        return

    logger.info(f"{'=' * 70}")
    logger.info(f"EXTRACTING FAILED ROWS: {pdf_name}")
    logger.info(f"{'=' * 70}")

    failed_files = sorted(failed_dir.glob("page_*_failed.json"),
                           key=lambda p: int(p.stem.split("_")[1]))

    total_cropped = 0
    total_failed_to_crop = 0

    for failed_path in failed_files:
        with open(failed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        failed_booths = data.get("failed_booths", [])
        if not failed_booths:
            continue

        page_num = data["page"]
        table_path = table_dir / f"page_{page_num}_table.png"

        if not table_path.exists():
            logger.error(f"  Missing table image for page {page_num}")
            continue

        table_img = cv2.imread(str(table_path))
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        boundaries = detect_row_boundaries(gray)

        ordered_ids = get_ordered_booth_ids(pdf_name, page_num)
        total_booths_on_page = len(ordered_ids)

        out_dir = out_base / f"page_{page_num}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for booth in failed_booths:
            booth_id = str(booth["booth_id"]).strip()

            if booth_id not in ordered_ids:
                logger.warning(f"  Booth {booth_id} not found in page {page_num}'s "
                                f"ordered booth list - cannot determine row position")
                total_failed_to_crop += 1
                continue

            row_position = ordered_ids.index(booth_id)
            row_img = crop_booth_row(table_img, boundaries, row_position,
                                      header_rows, total_booths_on_page)

            if row_img is None or row_img.size == 0:
                logger.warning(f"  Could not isolate row for booth {booth_id} on page {page_num}")
                total_failed_to_crop += 1
                continue

            out_path = out_dir / f"booth_{booth_id}_raw.png"
            cv2.imwrite(str(out_path), row_img)
            total_cropped += 1

    logger.info(f"[{pdf_name}] Done - {total_cropped} rows cropped, "
                f"{total_failed_to_crop} could not be isolated")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)