"""
STEP 10 - MERGE: passed + retried + manual corrections -> final page JSON

Input   : data/corrected_json/{pdf_name}/page_N_passed.json  (from Step 5)
          data/re_extracted/{pdf_name}/page_N/booth_ID.json  (from Step 8/9)
          data/manual_corrections/{pdf_name}.json            (your verified fixes)
Output  : data/corrected_json/{pdf_name}/page_N_corrected.json
          logs/10_merge_corrected.log

Manual corrections ALWAYS win - applied last, overriding anything else for
that exact page + booth_id combination. If a booth_id in manual_corrections
doesn't match any page/booth from the automated steps, it's still added -
this lets you add booths the automation missed entirely (e.g. booth 243).
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from config import (
    CORRECTED_JSON_DIR, RE_EXTRACTED_DIR, MANUAL_CORRECTIONS_DIR,
    PDF_NAMES, LOG_DIR, natural_sort_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "10_merge_corrected.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_manual_corrections(pdf_name: str) -> dict:
    """Returns {page_num: {booth_id: correction_dict}}"""
    path = MANUAL_CORRECTIONS_DIR / f"{pdf_name}.json"
    if not path.exists():
        logger.warning(f"[{pdf_name}] No manual_corrections file found at {path} - "
                        f"proceeding without manual overrides")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_page = {}
    for c in data.get("corrections", []):
        page = c["page"]
        c["booth_id"] = str(c["booth_id"]).strip()

        candidates = list(c["candidate_votes_by_name"].keys())
        c["candidate_votes"] = [c["candidate_votes_by_name"][n] for n in candidates]

        c["validation_status"] = "manually_corrected"
        c["extraction_source"] = "manual_correction"
        c.pop("_VERIFY_AGAINST_PDF", None)  # drop the reminder flag if still present
        c.pop("_STATUS", None)              # drop the fast/slow-check tag if still present

        by_page.setdefault(page, {})[c["booth_id"]] = c

    total = sum(len(v) for v in by_page.values())
    logger.info(f"[{pdf_name}] Loaded {total} manual corrections across {len(by_page)} pages")
    return by_page


def process_pdf(pdf_name: str):
    passed_dir = CORRECTED_JSON_DIR / pdf_name
    re_extracted_dir = RE_EXTRACTED_DIR / pdf_name

    if not passed_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 05_validate.py first")
        return

    manual_by_page = load_manual_corrections(pdf_name)

    logger.info(f"{'=' * 70}")
    logger.info(f"MERGING: {pdf_name}")
    logger.info(f"{'=' * 70}")

    passed_files = sorted(passed_dir.glob("page_*_passed.json"),
                           key=lambda p: int(p.stem.split("_")[1]))

    # Track which pages exist from the automated pipeline, so we can also
    # handle manual corrections for pages that had ZERO passed booths
    known_pages = {int(p.stem.split("_")[1]) for p in passed_files}
    all_pages = known_pages | set(manual_by_page.keys())

    for page_num in sorted(all_pages):
        passed_path = passed_dir / f"page_{page_num}_passed.json"
        booths_by_id = {}

        if passed_path.exists():
            with open(passed_path, "r", encoding="utf-8") as f:
                passed_data = json.load(f)
            for booth in passed_data.get("booths", []):
                booth["booth_id"] = str(booth["booth_id"]).strip()
                booth.setdefault("validation_status", "passed")
                booth.setdefault("extraction_source", "table")
                booths_by_id[booth["booth_id"]] = booth

        retry_page_dir = re_extracted_dir / f"page_{page_num}"
        if retry_page_dir.exists():
            for booth_path in retry_page_dir.glob("booth_*.json"):
                with open(booth_path, "r", encoding="utf-8") as f:
                    retried = json.load(f)
                retried["booth_id"] = str(retried["booth_id"]).strip()
                booths_by_id[retried["booth_id"]] = retried

        # Manual corrections ALWAYS win - applied last, added even if the
        # booth_id wasn't present from any automated step (e.g. booth 243)
        for bid, corrected in manual_by_page.get(page_num, {}).items():
            if bid not in booths_by_id:
                logger.info(f"  Page {page_num} booth {bid}: added from manual "
                            f"corrections (was not present in automated results)")
            booths_by_id[bid] = corrected

        merged = [booths_by_id[k] for k in sorted(booths_by_id, key=natural_sort_key)]

        out_data = {
            "pdf_name": pdf_name,
            "page": page_num,
            "booth_count": len(merged),
            "booths": merged,
        }

        out_path = passed_dir / f"page_{page_num}_corrected.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Page {page_num}: merged {len(merged)} booths -> {out_path.name}")

    logger.info(f"[{pdf_name}] Done")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)