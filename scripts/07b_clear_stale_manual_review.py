"""
STEP 7B - CLEAR STALE MANUAL-REVIEW RESULTS

Input   : data/validation_reports/{pdf_name}/manual_review_needed.json
Output  : deletes data/re_extracted/{pdf_name}/page_N/booth_ID.json
          for booths still in manual_review

WHY THIS EXISTS: Step 8 skips any booth whose output file already exists
(resume logic). Booths still needing manual review already have a result
file written (the best-effort failed attempt) - so re-running Step 8 would
just skip them again, even with the new PaddleOCR crop available.

This script deletes ONLY those specific stale files, for the specific
booths still listed in manual_review_needed.json - it does not touch any
already-passed booth's result, and does not touch raw_json, failed_rows,
or anything upstream.

Run this AFTER 06b_paddleocr_row_detect.py and BEFORE re-running
08_retry_gemini_extract.py.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging

from config import VALIDATION_REPORTS_DIR, RE_EXTRACTED_DIR, PDF_NAMES, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "07b_clear_stale_manual_review.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def process_pdf(pdf_name: str):
    review_path = VALIDATION_REPORTS_DIR / pdf_name / "manual_review_needed.json"
    if not review_path.exists():
        logger.info(f"[{pdf_name}] No manual_review_needed.json - skipping")
        return

    with open(review_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    review_list = data.get("needs_manual_review", [])
    if not review_list:
        logger.info(f"[{pdf_name}] No booths in manual review - skipping")
        return

    deleted = 0
    not_found = 0

    for item in review_list:
        page_num = item["page"]
        booth_id = str(item["booth_id"]).strip()
        out_path = RE_EXTRACTED_DIR / pdf_name / f"page_{page_num}" / f"booth_{booth_id}.json"

        if out_path.exists():
            out_path.unlink()
            deleted += 1
        else:
            not_found += 1

    logger.info(f"[{pdf_name}] Cleared {deleted} stale results "
                f"({not_found} already absent) - ready for re-retry")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)