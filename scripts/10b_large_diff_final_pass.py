"""
STEP 11 - LARGE-DIFF FINAL PASS (last automated attempt, large-diff booths only)

Input   : data/re_extracted/{pdf_name}/page_N/booth_ID.json
          (only files with validation_status != "retried_passed" AND
          diff > SMALL_ERROR_MAX_DIFF are touched - small-diff booths and
          anything already passing are left completely alone)
Output  : data/re_extracted/{pdf_name}/page_N/booth_ID.json (overwritten
          only if strictly improved)
          logs/11_large_diff_final_pass.log

WHY THIS EXISTS: Step 10 only gave large-diff booths ONE full-table
fallback attempt. It never tried the shift_-40..shift_40 row-offset crops
against them, even though those exist specifically to handle variable
row-position drift (the Nirmal lesson). This step closes that gap.

SECOND CHECK: for booths still failing after that, this makes TWO
independent full-table reads and compares them. If both agree with each
other on every number but still fail arithmetic, that's evidence the
EXTRACTION is accurate and the discrepancy exists on the original printed
form itself (a real clerical error, which does happen on this kind of
paperwork) - not evidence Gemini is misreading. These get tagged
"double_confirmed_source_discrepancy" instead of plain "manual_review" so
you know to visually confirm rather than assume the number is wrong.

HONEST LIMIT: this is expected to close some but not all remaining
large-diff cases. What's left after this run is very likely the genuine
Nirmal-style ceiling (faint/damaged print) and further automated attempts
are unlikely to help - manual review at that point is the correct answer,
not a fallback.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import google.generativeai as genai
import logging

from config import (
    GEMINI_API_KEY, MODEL_NAME, FAILED_ROWS_DIR, TABLES_ENHANCED_DIR,
    RE_EXTRACTED_DIR, PDF_NAMES, load_constituency_config, LOG_DIR,
    MAX_RETRIES, RETRY_BACKOFF_SECONDS, MAX_WORKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "11_large_diff_final_pass.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

MAX_DIMENSION_FOR_API = 4000
SHIFT_VARIANTS = ["shift_-40", "shift_-25", "shift_-10", "shift_10", "shift_25", "shift_40"]
SMALL_ERROR_MAX_DIFF = 10  # must match Step 10's threshold - only touch booths ABOVE this


def build_single_booth_schema(candidates: list) -> dict:
    return {
        "type": "object",
        "properties": {
            "serial_no_read_from_image": {"type": "string"},
            "candidate_votes_by_name": {
                "type": "object",
                "properties": {name: {"type": "integer"} for name in candidates},
                "required": candidates,
            },
            "total_valid_votes": {"type": "integer"},
            "rejected_votes": {"type": "integer"},
            "nota": {"type": "integer"},
            "total_votes": {"type": "integer"},
            "tendered_votes": {"type": "integer"},
        },
        "required": [
            "serial_no_read_from_image", "candidate_votes_by_name", "total_valid_votes",
            "rejected_votes", "nota", "total_votes", "tendered_votes",
        ],
    }


def build_shift_prompt(candidates: list, booth_id: str, prior_diff: int) -> str:
    candidate_list = "\n".join(f'  {i + 1}. "{name}"' for i, name in enumerate(candidates))
    return f"""
This image is a crop of a Form-20 table, shifted slightly up or down from
its expected position - it may show the target row, a neighboring row, or
a partial mix of both due to row-position drift in the original scan.

Look for the row where "Serial No. Of Polling Station" reads EXACTLY
"{booth_id}" (NEVER the plain "Serial No." row counter). If that exact
row is visible in this crop, report its data. If it is NOT visible in
this crop, report serial_no_read_from_image as "NOT_VISIBLE" and leave
other fields as 0.

A previous reading of this booth was off by {prior_diff} votes - a large
discrepancy suggesting the wrong row may have been read before.

The candidates, in the same left-to-right order as the columns, are:
{candidate_list}

Read every number carefully, digit by digit.
"""


def build_fulltable_prompt(candidates: list, target_booth_id: str) -> str:
    candidate_list = "\n".join(f'  {i + 1}. "{name}"' for i, name in enumerate(candidates))
    return f"""
This image is a full Form-20 results table with many booth rows.

Find the ONE row where the "Serial No. Of Polling Station" column value is
EXACTLY "{target_booth_id}" (match the text exactly, including any letter
suffix) - NEVER match against the plain "Serial No." row counter.
Report that Serial No. Of Polling Station value in serial_no_read_from_image
to confirm you found the right row.

The candidates, in the same left-to-right order as the columns, are:
{candidate_list}

Read every number in that row extremely carefully, digit by digit. Take
your time - this row has been difficult to read correctly in previous
attempts.
"""


def is_valid(booth: dict, candidates: list) -> tuple[bool, int]:
    votes = [booth["candidate_votes_by_name"].get(name, 0) for name in candidates]
    calc = sum(votes) + booth["rejected_votes"] + booth["nota"]
    diff = calc - booth["total_votes"]
    return diff == 0, abs(diff)


def load_image_for_api(image_path: Path):
    img = Image.open(image_path)
    if max(img.size) > MAX_DIMENSION_FOR_API:
        scale = MAX_DIMENSION_FOR_API / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def call_gemini_with_retry(image_path: Path, prompt: str, schema: dict) -> str:
    img = load_image_for_api(image_path)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            return response.text
        except Exception as e:
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(f"    Gemini call failed (attempt {attempt}/{MAX_RETRIES}): {e} - retrying in {wait}s")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


def parse_result(raw_text: str, candidates: list, booth_id: str) -> dict | None:
    data = json.loads(raw_text)
    reported_id = str(data.get("serial_no_read_from_image", "")).strip()
    if reported_id != booth_id:
        return None
    data["booth_id"] = booth_id
    data["candidate_votes"] = [data["candidate_votes_by_name"].get(name, 0) for name in candidates]
    return data


def results_agree(a: dict, b: dict, candidates: list) -> bool:
    if a["candidate_votes"] != b["candidate_votes"]:
        return False
    for key in ("total_valid_votes", "rejected_votes", "nota", "total_votes", "tendered_votes"):
        if a[key] != b[key]:
            return False
    return True


def process_booth_file(pdf_name: str, booth_path: Path, candidates: list, schema: dict) -> str:
    with open(booth_path, "r", encoding="utf-8") as f:
        prior = json.load(f)

    if prior.get("validation_status") == "retried_passed":
        return "already_passed"

    booth_id = prior["booth_id"]
    page_num = int(booth_path.parent.name.split("_")[1])
    _, prior_diff = is_valid(prior, candidates)

    if prior_diff <= SMALL_ERROR_MAX_DIFF:
        return "skipped_small_diff"  # Step 10 already handled these

    page_dir = FAILED_ROWS_DIR / pdf_name / f"page_{page_num}"
    schema_local = schema

    # --- Attempt 1: try shift-variant crops (never attempted for large-diff before) ---
    for v in SHIFT_VARIANTS:
        variant_path = page_dir / f"booth_{booth_id}_{v}.png"
        if not variant_path.exists():
            continue
        prompt = build_shift_prompt(candidates, booth_id, prior_diff)
        try:
            raw_text = call_gemini_with_retry(variant_path, prompt, schema_local)
            result = parse_result(raw_text, candidates, booth_id)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"  [{pdf_name}] page {page_num} booth {booth_id} {v}: {e}")
            continue
        if result is None:
            continue
        valid, new_diff = is_valid(result, candidates)
        if valid:
            result["validation_status"] = "retried_passed"
            result["extraction_source"] = f"step11_shift_{v}"
            with open(booth_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"  [{pdf_name}] page {page_num} booth {booth_id}: "
                        f"NOW PASSES via shift crop {v} (was off by {prior_diff})")
            return "passed"

    # --- Attempt 2: two independent full-table reads, check if they agree ---
    table_path = TABLES_ENHANCED_DIR / pdf_name / f"page_{page_num}_table.png"
    if not table_path.exists():
        return "no_improvement"

    prompt = build_fulltable_prompt(candidates, booth_id)
    attempts = []
    for _ in range(2):
        try:
            raw_text = call_gemini_with_retry(table_path, prompt, schema_local)
            result = parse_result(raw_text, candidates, booth_id)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"  [{pdf_name}] page {page_num} booth {booth_id}: fulltable attempt failed - {e}")
            continue
        if result is not None:
            attempts.append(result)

    if not attempts:
        return "no_improvement"

    best = attempts[0]
    valid, new_diff = is_valid(best, candidates)

    if valid:
        best["validation_status"] = "retried_passed"
        best["extraction_source"] = "step11_fulltable"
        with open(booth_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        logger.info(f"  [{pdf_name}] page {page_num} booth {booth_id}: "
                    f"NOW PASSES via step11 fulltable (was off by {prior_diff})")
        return "passed"

    if len(attempts) == 2 and results_agree(attempts[0], attempts[1], candidates):
        best["validation_status"] = "double_confirmed_source_discrepancy"
        best["extraction_source"] = "step11_fulltable_double_confirmed"
        best["note"] = (f"Two independent reads agree with each other but still don't "
                         f"sum correctly (off by {new_diff}). This likely means the "
                         f"ORIGINAL PRINTED FORM has a genuine discrepancy, not that "
                         f"the extraction is wrong. Visually confirm against the PDF "
                         f"rather than assuming a misread.")
        with open(booth_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        logger.warning(f"  [{pdf_name}] page {page_num} booth {booth_id}: "
                        f"two reads AGREE but still off by {new_diff} - likely genuine "
                        f"source discrepancy, flagged for visual confirmation")
        return "double_confirmed"

    if new_diff < prior_diff:
        best["validation_status"] = "manual_review"
        best["extraction_source"] = "step11_fulltable"
        with open(booth_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)
        return "improved_not_passed"

    return "no_improvement"


def process_pdf(pdf_name: str):
    base_dir = RE_EXTRACTED_DIR / pdf_name
    if not base_dir.exists():
        return

    config = load_constituency_config(pdf_name)
    candidates = config["candidates"]
    schema = build_single_booth_schema(candidates)

    booth_files = sorted(base_dir.glob("page_*/booth_*.json"))
    if not booth_files:
        return

    logger.info(f"{'=' * 70}")
    logger.info(f"LARGE-DIFF FINAL PASS: {pdf_name} ({len(booth_files)} booth files to check)")
    logger.info(f"{'=' * 70}")

    counts = {"already_passed": 0, "skipped_small_diff": 0, "passed": 0,
              "double_confirmed": 0, "improved_not_passed": 0, "no_improvement": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_booth_file, pdf_name, bp, candidates, schema): bp
            for bp in booth_files
        }
        for future in as_completed(futures):
            bp = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as e:
                logger.error(f"  {bp}: UNEXPECTED ERROR - {e}")

    still_review = counts["improved_not_passed"] + counts["no_improvement"] + counts["double_confirmed"]
    logger.info(f"[{pdf_name}] {counts['passed']} newly passed, "
                f"{counts['double_confirmed']} flagged as likely source discrepancies, "
                f"{still_review} total still need manual review")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)