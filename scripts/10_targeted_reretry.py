"""
STEP 10 - TARGETED RE-RETRY (second-chance pass for manual_review booths only)

Input   : data/re_extracted/{pdf_name}/page_N/booth_ID.json
          (only files where validation_status != "retried_passed" are touched -
          anything already passing is never opened, so it cannot regress)
Output  : data/re_extracted/{pdf_name}/page_N/booth_ID.json (overwritten IN PLACE
          only if the new attempt actually passes or gets strictly closer)
          logs/10_targeted_reretry.log

STRATEGY: Step 8 already proved label-verification works (booth_id is
trustworthy in every manual_review entry - Step 9's own docstring confirms
this). What's left is purely a digit-reading problem. Two different error
sizes point to two different likely causes, so they get two different
retry strategies instead of one generic re-ask:

  SMALL diff (<= SMALL_ERROR_MAX_DIFF): most likely a single misread
  digit. Retry with a prompt that explicitly names commonly-confused
  digit pairs (1/7, 3/8, 0/6, 5/6, 4/9) and asks Gemini to specifically
  re-examine against those pairs, tried on whichever enhanced row-crop
  variants already exist on disk (no new image processing needed).

  LARGE diff (> SMALL_ERROR_MAX_DIFF): more likely a wrong column read
  (e.g. rejected_votes and nota swapped) or a genuinely different row
  than expected. Retry via the full-table fallback with an explicit
  request to re-verify column alignment, not just re-read digits.

Booths that still fail after this get left as manual_review, unchanged -
this script does not claim to close 100% of the gap, only to shrink it.
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
        logging.FileHandler(LOG_DIR / "10_targeted_reretry.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

MAX_DIMENSION_FOR_API = 4000
VARIANT_LABELS = ["v1", "v2", "v3", "shift_-40", "shift_-25", "shift_-10",
                  "shift_10", "shift_25", "shift_40", "enhanced2"]

SMALL_ERROR_MAX_DIFF = 10  # <=10 votes off: treat as single-digit misread


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


def build_small_diff_prompt(candidates: list, booth_id: str, prior_diff: int,
                             prior: dict) -> str:
    candidate_list = "\n".join(f'  {i + 1}. "{name}"' for i, name in enumerate(candidates))
    return f"""
This image contains exactly ONE election booth row from a Form-20 table -
Serial No. Of Polling Station "{booth_id}".

A previous reading of this row did NOT sum correctly - it was off by
{prior_diff} vote(s). This is a small discrepancy, most likely caused by
ONE misread digit somewhere in this row. Look extremely carefully at
EVERY digit, especially checking for these commonly confused pairs in
printed/scanned tables:
  1 <-> 7      3 <-> 8      0 <-> 6      5 <-> 6      4 <-> 9
Also check for a digit that may have been missed entirely (e.g. reading
"10" instead of "100") or an extra digit accidentally added.

The candidates, in the same left-to-right order as the columns, are:
{candidate_list}

Read the "Serial No. Of Polling Station" value first and report it in
serial_no_read_from_image to confirm you are looking at the right row -
NEVER the plain "Serial No." row counter.

Report your BEST, most careful reading of every number in this row.
"""


def build_large_diff_prompt(candidates: list, booth_id: str, prior_diff: int) -> str:
    candidate_list = "\n".join(f'  {i + 1}. "{name}"' for i, name in enumerate(candidates))
    return f"""
This image is a full Form-20 results table with many booth rows.

Find the ONE row where the "Serial No. Of Polling Station" column value
is EXACTLY "{booth_id}" (match the text exactly, including any letter
suffix) - NEVER match against the plain "Serial No." row counter.

A previous reading of this row was off by {prior_diff} votes - a LARGE
discrepancy, which usually means either:
  (a) a column was misread or shifted (e.g. "No. of rejected votes" and
      "NOTA" values swapped, or a candidate column shifted by one), or
  (b) the wrong row was read entirely.

Carefully re-verify you have the correct row (double-check the Serial
No. Of Polling Station value matches exactly), then carefully re-read
EVERY column for that row in its correct left-to-right position:
{candidate_list}

Report the Serial No. Of Polling Station in serial_no_read_from_image
to confirm the row, and report your BEST, most careful reading of every
number.
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
        return None  # still didn't find the right row - don't use this attempt
    data["booth_id"] = booth_id
    data["candidate_votes"] = [data["candidate_votes_by_name"].get(name, 0) for name in candidates]
    return data


def retry_small_diff(pdf_name: str, page_num: int, page_dir: Path, booth_id: str,
                      candidates: list, schema: dict, prior: dict, prior_diff: int) -> dict | None:
    best = None
    best_diff = prior_diff
    for v in VARIANT_LABELS:
        variant_path = page_dir / f"booth_{booth_id}_{v}.png"
        if not variant_path.exists():
            continue
        prompt = build_small_diff_prompt(candidates, booth_id, prior_diff, prior)
        try:
            raw_text = call_gemini_with_retry(variant_path, prompt, schema)
        except Exception as e:
            logger.error(f"  [{pdf_name}] page {page_num} booth {booth_id} {v}: call failed - {e}")
            continue
        try:
            result = parse_result(raw_text, candidates, booth_id)
        except json.JSONDecodeError:
            continue
        if result is None:
            continue
        valid, new_diff = is_valid(result, candidates)
        if valid:
            result["validation_status"] = "retried_passed"
            result["extraction_source"] = f"step10_small_diff_{v}"
            return result
        if new_diff < best_diff:
            result["validation_status"] = "manual_review"
            result["extraction_source"] = f"step10_small_diff_{v}"
            best, best_diff = result, new_diff
    return best


def retry_large_diff(pdf_name: str, page_num: int, booth_id: str,
                      candidates: list, schema: dict, prior_diff: int) -> dict | None:
    table_path = TABLES_ENHANCED_DIR / pdf_name / f"page_{page_num}_table.png"
    if not table_path.exists():
        return None
    prompt = build_large_diff_prompt(candidates, booth_id, prior_diff)
    try:
        raw_text = call_gemini_with_retry(table_path, prompt, schema)
    except Exception as e:
        logger.error(f"  [{pdf_name}] page {page_num} booth {booth_id}: fallback call failed - {e}")
        return None
    try:
        result = parse_result(raw_text, candidates, booth_id)
    except json.JSONDecodeError:
        return None
    if result is None:
        return None
    valid, new_diff = is_valid(result, candidates)
    result["validation_status"] = "retried_passed" if valid else "manual_review"
    result["extraction_source"] = "step10_large_diff_fulltable"
    if valid or new_diff < prior_diff:
        return result
    return None


def process_booth_file(pdf_name: str, booth_path: Path, candidates: list, schema: dict) -> str:
    with open(booth_path, "r", encoding="utf-8") as f:
        prior = json.load(f)

    if prior.get("validation_status") == "retried_passed":
        return "already_passed"  # never touch passing booths

    booth_id = prior["booth_id"]
    page_num = int(booth_path.parent.name.split("_")[1])
    _, prior_diff = is_valid(prior, candidates)

    page_dir = FAILED_ROWS_DIR / pdf_name / f"page_{page_num}"

    if prior_diff <= SMALL_ERROR_MAX_DIFF:
        new_result = retry_small_diff(pdf_name, page_num, page_dir, booth_id,
                                       candidates, schema, prior, prior_diff)
    else:
        new_result = retry_large_diff(pdf_name, page_num, booth_id,
                                       candidates, schema, prior_diff)

    if new_result is None:
        return "no_improvement"

    with open(booth_path, "w", encoding="utf-8") as f:
        json.dump(new_result, f, indent=2, ensure_ascii=False)

    if new_result["validation_status"] == "retried_passed":
        logger.info(f"  [{pdf_name}] page {page_num} booth {booth_id}: "
                     f"NOW PASSES (was off by {prior_diff})")
        return "passed"
    else:
        logger.info(f"  [{pdf_name}] page {page_num} booth {booth_id}: "
                     f"improved but still off (was {prior_diff}, still manual_review)")
        return "improved_not_passed"


def process_pdf(pdf_name: str):
    base_dir = RE_EXTRACTED_DIR / pdf_name
    if not base_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 08_retry_gemini_extract.py first")
        return

    config = load_constituency_config(pdf_name)
    candidates = config["candidates"]
    schema = build_single_booth_schema(candidates)

    booth_files = sorted(base_dir.glob("page_*/booth_*.json"))
    if not booth_files:
        return

    logger.info(f"{'=' * 70}")
    logger.info(f"TARGETED RE-RETRY: {pdf_name} ({len(booth_files)} booth files to check)")
    logger.info(f"{'=' * 70}")

    counts = {"already_passed": 0, "passed": 0, "improved_not_passed": 0, "no_improvement": 0}

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

    newly_passed = counts["passed"]
    still_review = counts["improved_not_passed"] + counts["no_improvement"]
    logger.info(f"[{pdf_name}] {newly_passed} newly passed, {still_review} still need "
                f"manual review ({counts['already_passed']} were already passing, untouched)")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)