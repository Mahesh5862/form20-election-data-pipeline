"""
STEP 5 - VALIDATE

Input   : data/raw_json/{pdf_name}/page_N_raw.json
Output  : data/failed_booths/{pdf_name}/page_N_failed.json   (only failing booths)
          data/corrected_json/{pdf_name}/page_N_passed.json  (passing booths)
          logs/05_validate.log

VALIDATION FORMULA (per booth):
    sum(candidate_votes) + rejected_votes + nota == total_votes
    AND
    the two extraction passes from Step 4 must have agreed (consistency_check
    != "mismatched_between_passes")

booth_id is a STRING (e.g. "117" or "117A" for split/auxiliary polling
stations) - it is a label only, never included in the arithmetic sum.

WHY THE CONSISTENCY CHECK MATTERS: a booth can have numbers that sum
correctly (passes arithmetic) even when the two independent Gemini passes
disagreed on what those numbers actually are. Without this check, such a
booth would silently pass even though we already have direct evidence the
extraction wasn't reliable. Routing these into the retry pipeline instead
catches errors arithmetic alone cannot see.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from config import RAW_JSON_DIR, FAILED_BOOTHS_DIR, CORRECTED_JSON_DIR, PDF_NAMES, LOG_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "05_validate.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def validate_booth(booth: dict):
    """Returns (calculated_total, reported_total, difference, is_ok).

    is_ok requires BOTH:
      1. arithmetic matches (calculated == reported)
      2. the two Step 4 extraction passes agreed on this booth
    """
    candidate_votes = booth["candidate_votes"]
    calculated = sum(candidate_votes) + booth["rejected_votes"] + booth["nota"]
    reported = booth["total_votes"]
    diff = calculated - reported

    arithmetic_ok = (diff == 0)
    consistency_ok = booth.get("consistency_check", "matched") != "mismatched_between_passes"

    is_ok = arithmetic_ok and consistency_ok
    return calculated, reported, diff, is_ok


def process_page(json_path: Path, pdf_name: str, page_num: int):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    passed_booths = []
    failed_booths = []
    seen_ids = []

    for booth in data["booths"]:
        # Ensure booth_id is always a string, regardless of what Step 4 stored
        booth["booth_id"] = str(booth["booth_id"]).strip()
        seen_ids.append(booth["booth_id"])

        calc, reported, diff, is_ok = validate_booth(booth)

        if is_ok:
            passed_booths.append(booth)
        else:
            reason = []
            if diff != 0:
                reason.append("arithmetic_mismatch")
            if booth.get("consistency_check") == "mismatched_between_passes":
                reason.append("passes_disagreed")

            failed_booths.append({
                "pdf_name": pdf_name,
                "page": page_num,
                "booth_id": booth["booth_id"],
                "candidate_votes": booth["candidate_votes"],
                "candidate_votes_by_name": booth.get("candidate_votes_by_name", {}),
                "candidate_count_used_in_sum": len(booth["candidate_votes"]),
                "total_valid_votes": booth["total_valid_votes"],
                "rejected_votes": booth["rejected_votes"],
                "nota": booth["nota"],
                "reported_total": reported,
                "calculated_total": calc,
                "difference": diff,
                "failure_reason": reason,
                "status": "FAILED",
            })

    duplicates = {bid for bid in seen_ids if seen_ids.count(bid) > 1}
    if duplicates:
        logger.warning(f"[{pdf_name}] page {page_num}: DUPLICATE booth_id(s) "
                        f"in raw data: {duplicates} - check original PDF for "
                        f"split/auxiliary stations")

    return passed_booths, failed_booths


def process_pdf(pdf_name: str):
    src_dir = RAW_JSON_DIR / pdf_name
    failed_dir = FAILED_BOOTHS_DIR / pdf_name
    corrected_dir = CORRECTED_JSON_DIR / pdf_name

    if not src_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 04_gemini_extract_table.py first")
        return

    failed_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("page_*_raw.json"),
                    key=lambda p: int(p.stem.split("_")[1]))

    logger.info(f"{'=' * 70}")
    logger.info(f"VALIDATING: {pdf_name} ({len(files)} pages)")
    logger.info(f"{'=' * 70}")

    total_pass = 0
    total_fail = 0
    total_consistency_only_fail = 0

    for json_path in files:
        page_num = int(json_path.stem.split("_")[1])
        passed, failed = process_page(json_path, pdf_name, page_num)

        total_pass += len(passed)
        total_fail += len(failed)
        total_consistency_only_fail += sum(
            1 for f in failed
            if f["failure_reason"] == ["passes_disagreed"]
        )

        with open(corrected_dir / f"page_{page_num}_passed.json", "w", encoding="utf-8") as f:
            json.dump({"page": page_num, "booths": passed}, f, indent=2, ensure_ascii=False)

        with open(failed_dir / f"page_{page_num}_failed.json", "w", encoding="utf-8") as f:
            json.dump({"page": page_num, "failed_booths": failed}, f, indent=2, ensure_ascii=False)

        status = "all passed" if len(failed) == 0 else f"{len(failed)} failed"
        logger.info(f"Page {page_num}: {len(passed)} passed, {status}")

    total = total_pass + total_fail
    accuracy = (total_pass / total * 100) if total else 0
    logger.info(f"[{pdf_name}] Passed {total_pass}/{total} booths ({accuracy:.2f}%)")
    if total_consistency_only_fail:
        logger.info(f"[{pdf_name}] Of those failures, {total_consistency_only_fail} were "
                    f"caught ONLY by the consistency check (arithmetic alone would have "
                    f"missed these)")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)