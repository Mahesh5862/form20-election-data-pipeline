"""
STEP 9 - VALIDATE RE-EXTRACTED BOOTHS

Input   : data/re_extracted/{pdf_name}/page_N/booth_ID.json
Output  : data/validation_reports/{pdf_name}/manual_review_needed.json
          logs/09_validate_reextracted.log

Collects every booth still flagged manual_review into one report per PDF,
so you know exactly what to check by hand instead of searching folders.

booth_id = the Serial No. column value from the original PDF table. Every
booth_id in re_extracted/ has already been self-verified by Step 8 (Gemini
confirmed the Serial No. it read matches the requested booth_id) - so the
label-to-data pairing here is trustworthy. Only the arithmetic (does the
sum add up) may still be wrong for manual_review entries.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from config import RE_EXTRACTED_DIR, VALIDATION_REPORTS_DIR, PDF_NAMES, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "09_validate_reextracted.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "booth_id", "candidate_votes", "total_valid_votes",
    "rejected_votes", "nota", "total_votes", "tendered_votes",
]


def process_pdf(pdf_name: str):
    base_dir = RE_EXTRACTED_DIR / pdf_name

    if not base_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 08_retry_gemini_extract.py first")
        return

    out_dir = VALIDATION_REPORTS_DIR / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 70}")
    logger.info(f"VALIDATING RE-EXTRACTED: {pdf_name}")
    logger.info(f"{'=' * 70}")

    passed = 0
    manual_review = []

    for booth_path in sorted(base_dir.glob("page_*/booth_*.json")):
        with open(booth_path, "r", encoding="utf-8") as f:
            booth = json.load(f)

        missing = [k for k in REQUIRED_KEYS if k not in booth]
        if missing:
            logger.error(f"  {booth_path}: missing keys {missing}")
            continue

        # Real page number comes from the folder name (e.g. "page_8" -> 8),
        # since individual booth JSONs don't store their own page number.
        page_num = int(booth_path.parent.name.split("_")[1])

        status = booth.get("validation_status", "unknown")
        if status == "retried_passed":
            passed += 1
        else:
            manual_review.append({
                "page": page_num,
                "booth_id": booth["booth_id"],
                "file": str(booth_path.relative_to(base_dir.parent.parent)),
                "candidate_votes": booth["candidate_votes"],
                "total_votes": booth["total_votes"],
                "extraction_source": booth.get("extraction_source", "unknown"),
                "note": "Label (booth_id) is verified correct. Only the arithmetic "
                        "did not match - verify the numbers against the original PDF page.",
            })

    report_path = out_dir / "manual_review_needed.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "pdf_name": pdf_name,
            "retried_and_passed": passed,
            "needs_manual_review_count": len(manual_review),
            "needs_manual_review": manual_review,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"[{pdf_name}] Retried and passed: {passed}")
    logger.info(f"[{pdf_name}] Needs manual review: {len(manual_review)} -> {report_path}")
    for item in manual_review:
        logger.info(f"    - page {item['page']}, booth {item['booth_id']}")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)