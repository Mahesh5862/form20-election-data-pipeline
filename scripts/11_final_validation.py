"""
STEP 11 - FINAL VALIDATION

Input   : data/corrected_json/{pdf_name}/page_N_corrected.json
Output  : data/validation_reports/{pdf_name}/page_N_validation.json
          logs/11_final_validation.log

Re-checks arithmetic on every merged booth (belt-and-braces after the merge)
and flags missing/duplicate booth IDs. Also reports a final accuracy
percentage per PDF (booths passing arithmetic / total booths).
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from config import CORRECTED_JSON_DIR, VALIDATION_REPORTS_DIR, PDF_NAMES, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "11_final_validation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def process_pdf(pdf_name: str):
    src_dir = CORRECTED_JSON_DIR / pdf_name
    dst_dir = VALIDATION_REPORTS_DIR / pdf_name

    if not src_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 10_merge_corrected.py first")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 70}")
    logger.info(f"FINAL VALIDATION: {pdf_name}")
    logger.info(f"{'=' * 70}")

    total_still_failing = 0
    total_duplicates = 0
    total_booths = 0

    for path in sorted(src_dir.glob("page_*_corrected.json"),
                        key=lambda p: int(p.stem.split("_")[1])):
        with open(path, "r", encoding="utf-8") as f:
            page_data = json.load(f)

        page_num = page_data["page"]
        results = []
        seen_ids = []

        for booth in page_data["booths"]:
            calc = sum(booth["candidate_votes"]) + booth["rejected_votes"] + booth["nota"]
            ok = calc == booth["total_votes"]
            seen_ids.append(booth["booth_id"])
            results.append({
                "booth_id": booth["booth_id"],
                "arithmetic_ok": ok,
                "calculated_total": calc,
                "reported_total": booth["total_votes"],
                "validation_status": booth.get("validation_status", "unknown"),
            })

        duplicates = [b for b in set(seen_ids) if seen_ids.count(b) > 1]
        still_failing = [r["booth_id"] for r in results if not r["arithmetic_ok"]]

        total_still_failing += len(still_failing)
        total_duplicates += len(duplicates)
        total_booths += len(page_data["booths"])

        report = {
            "pdf_name": pdf_name,
            "page": page_num,
            "booth_count": len(page_data["booths"]),
            "duplicates": duplicates,
            "still_failing_arithmetic": still_failing,
            "needs_manual_review": [
                r["booth_id"] for r in results
                if r["validation_status"] == "manual_review"
            ],
            "results": results,
        }

        out_path = dst_dir / f"page_{page_num}_validation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        flag = "OK" if not still_failing and not duplicates else "NEEDS ATTENTION"
        logger.info(f"Page {page_num}: {flag} "
                    f"(still_failing={len(still_failing)}, duplicates={len(duplicates)})")

        if still_failing:
            logger.warning(f"  Still failing arithmetic: {still_failing}")
        if duplicates:
            logger.warning(f"  Duplicate booth_ids: {duplicates}")

    passed = total_booths - total_still_failing
    accuracy = (passed / total_booths * 100) if total_booths else 0

    logger.info(f"[{pdf_name}] TOTAL: {total_still_failing} still failing arithmetic, "
                f"{total_duplicates} duplicates across all pages")
    logger.info(f"[{pdf_name}] FINAL ACCURACY: {passed}/{total_booths} booths correct "
                f"({accuracy:.2f}%)")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)