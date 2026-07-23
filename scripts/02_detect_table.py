"""
STEP 2 - DETECT & CROP TABLE (v2 - skips title/header block above the table)

Input   : data/pages_rotated/{pdf_name}/page_N.png
Output  : data/tables/{pdf_name}/page_N_table.png

The page often has a title block (FORM 20 / FINAL RESULT SHEET / ...) sitting
inside the SAME outer border as the actual data table, so a simple "biggest
bordered region" crop grabs both. This version finds where the real column
grid starts by counting how many vertical divider lines cross each
horizontal band - title rows only have 2 (the outer left/right border),
the real table has one per column (15+ for Form-20).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import PAGES_ROTATED_DIR, TABLES_DIR, PDF_NAMES

MIN_COLUMN_LINES_FOR_TABLE_ROW = 8  # title rows have ~2, real table rows have 15+


def get_line_masks(gray):
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    return horiz_lines, vert_lines


def count_vertical_lines_in_band(vert_lines, y1, y2):
    """Counts distinct vertical line x-positions crossing this horizontal band."""
    band = vert_lines[y1:y2, :]
    col_sums = np.sum(band, axis=0)
    active_cols = col_sums > (band.shape[0] * 0.5 * 255)

    # collapse consecutive active columns into single line counts
    count = 0
    prev = False
    for is_active in active_cols:
        if is_active and not prev:
            count += 1
        prev = is_active
    return count


def get_horizontal_line_positions(horiz_lines):
    row_sums = np.sum(horiz_lines, axis=1)
    threshold = row_sums.max() * 0.5
    line_ys = [y for y in range(len(row_sums)) if row_sums[y] > threshold]

    positions = []
    for y in line_ys:
        if not positions or y - positions[-1] > 5:
            positions.append(y)
    return positions


def detect_table_bbox(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    horiz_lines, vert_lines = get_line_masks(gray)

    horiz_positions = get_horizontal_line_positions(horiz_lines)
    if len(horiz_positions) < 2:
        return None

    # Find the first horizontal band where vertical line count jumps up -
    # this is where the real column grid (the table) begins.
    table_top = None
    for i in range(len(horiz_positions) - 1):
        y1, y2 = horiz_positions[i], horiz_positions[i + 1]
        if y2 - y1 < 5:
            continue
        col_count = count_vertical_lines_in_band(vert_lines, y1, y2)
        if col_count >= MIN_COLUMN_LINES_FOR_TABLE_ROW:
            table_top = y1
            break

    if table_top is None:
        return None  # couldn't confidently find the table start

    # Full width of the bordered region, from table_top down to the last line
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    contours, _ = cv2.findContours(
        cv2.add(horiz_lines, vert_lines), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    img_h, img_w = gray.shape
    if w < img_w * 0.3:
        return None

    # Use the outer box for x/width, but override the top with our detected table_top
    new_y = max(y, table_top)
    new_h = (y + h) - new_y

    margin = 8
    x = max(0, x - margin)
    new_y = max(0, new_y - margin)
    w = min(img_w - x, w + 2 * margin)
    new_h = min(img_h - new_y, new_h + 2 * margin)

    return x, new_y, w, new_h


def process_page(src_path: Path, dst_path: Path):
    img_bgr = cv2.imread(str(src_path))
    bbox = detect_table_bbox(img_bgr)

    if bbox is None:
        print(f"  WARNING: table not confidently detected in {src_path.name} - using full page")
        cropped = img_bgr
    else:
        x, y, w, h = bbox
        cropped = img_bgr[y:y + h, x:x + w]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), cropped)


def process_pdf(pdf_name: str):
    src_dir = PAGES_ROTATED_DIR / pdf_name
    dst_dir = TABLES_DIR / pdf_name

    if not src_dir.exists():
        print(f"[{pdf_name}] SKIP - run 00_deskew_rotate.py first")
        return

    pages = sorted(src_dir.glob("page_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))

    print(f"\n{'=' * 70}\nDETECTING TABLES: {pdf_name} ({len(pages)} pages)\n{'=' * 70}")

    for page_path in pages:
        page_num = page_path.stem.split("_")[1]
        dst_path = dst_dir / f"page_{page_num}_table.png"
        process_page(page_path, dst_path)
        print(f"Cropped: {dst_path.name}")

    print(f"[{pdf_name}] Done")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)