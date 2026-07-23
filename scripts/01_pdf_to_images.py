"""
STEP 1 - PDF -> PAGE IMAGES

Input   : input_pdfs/{pdf_name}.pdf
Output  : data/pages_raw/{pdf_name}/page_1.png ... page_N.png

Renders every page at 600 DPI. Nothing else - no rotation, no cropping here.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pdf2image import convert_from_path
from config import INPUT_PDFS_DIR, PAGES_RAW_DIR, PDF_NAMES, RENDER_DPI


def process_pdf(pdf_name: str):
    pdf_path = INPUT_PDFS_DIR / f"{pdf_name}.pdf"

    if not pdf_path.exists():
        print(f"[{pdf_name}] SKIP - PDF not found at {pdf_path}")
        return

    out_dir = PAGES_RAW_DIR / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\nRENDERING: {pdf_name}\n{'=' * 70}")

    pages = convert_from_path(str(pdf_path), dpi=RENDER_DPI)

    for i, page in enumerate(pages, start=1):
        out_path = out_dir / f"page_{i}.png"
        page.save(out_path)
        print(f"Saved: {out_path.name}")

    print(f"[{pdf_name}] Done - {len(pages)} pages rendered at {RENDER_DPI} DPI")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)