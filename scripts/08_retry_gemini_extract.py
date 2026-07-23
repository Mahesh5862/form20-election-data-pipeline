"""
STEP 8 - RETRY GEMINI EXTRACTION (self-verifying, row-safe, string booth_id, parallel)

Input   : data/failed_rows/{pdf_name}/page_N/booth_ID_v1.png (v2, v3, shift_*)
          data/tables_enhanced/{pdf_name}/page_N_table.png (fallback only)
Output  : data/re_extracted/{pdf_name}/page_N/booth_ID.json
          logs/08_retry_gemini_extract.log

FIX (this version): both prompts were asking Gemini to read the plain
"Serial No." row-counter column (1, 2, 3, 4... sequential) and then
comparing that against expected_booth_id, which comes from the DIFFERENT
"Serial No. Of Polling Station" column - the real booth identifier. These
two columns are different by design (Step 4 established this rule
explicitly). Comparing them was causing near-constant false "mismatch"
discards, correct only by coincidence when a booth's row position happened
to equal its polling-station ID. Both prompts now correctly ask for
"Serial No. Of Polling Station".
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
    GEMINI_API_KEY, MODEL_NAME, FAILED_BOOTHS_DIR, FAILED_ROWS_DIR,
    TABLES_ENHANCED_DIR, RE_EXTRACTED_DIR, PDF_NAMES,
    load_constituency_config, LOG_DIR, MAX_RETRIES, RETRY_BACKOFF_SECONDS,
    MAX_WORKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "08_retry_gemini_extract.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

MAX_DIMENSION_FOR_API = 4000

VARIANT_LABELS = ["v1", "v2", "v3", "shift_-40", "shift_-25", "shift_-10",
                  "shift_10", "shift_25", "shift_40", "enhanced2"]

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


def build_crop_prompt(candidates: list) -> str:
    candidate_list = "\n".join(f'  {i + 1}. "{name}"' for i, name in enumerate(candidates))
    return f"""
This image contains exactly ONE election booth row from a Form-20 table.

This table has TWO similarly-named columns near the left edge:
"Serial No." (a simple row counter, always sequential - 1, 2, 3, 4...)
and "Serial No. Of Polling Station" (the real polling station identifier,
which can include a letter suffix like "117A" for a split/auxiliary station).

Read the "Serial No. Of Polling Station" value printed in this row -
NEVER the plain "Serial No." row counter - and report it in
serial_no_read_from_image as TEXT, exactly as printed, including any
letter suffix. Do not convert it to a plain number or guess.

The candidates, in the same left-to-right order as the columns, are:
{candidate_list}

Read every number carefully, digit by digit.
"""


def build_fulltable_prompt(candidates: list, target_booth_id: str) -> str:
    candidate_list = "\n".join(f'  {i + 1}. "{name}"' for i, name in enumerate(candidates))
    return f"""
This image is a full Form-20 results table with many booth rows.

This table has TWO similarly-named columns near the left edge:
"Serial No." (a simple row counter, always sequential - 1, 2, 3, 4...)
and "Serial No. Of Polling Station" (the real polling station identifier,
which can include a letter suffix like "117A" for a split/auxiliary station).

Find the ONE row where the "Serial No. Of Polling Station" column value is
EXACTLY "{target_booth_id}" (match the text exactly, including any letter
suffix) - NEVER match against the plain "Serial No." row counter.
Report that Serial No. Of Polling Station value in serial_no_read_from_image
to confirm you found the right row.

The candidates, in the same left-to-right order as the columns, are:
{candidate_list}

Read every number in that row carefully, digit by digit.
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


def try_crop_variant(variant_path: Path, candidates: list, schema: dict,
                      expected_booth_id: str) -> dict | None:
    prompt = build_crop_prompt(candidates)
    raw_text = call_gemini_with_retry(variant_path, prompt, schema)
    data = json.loads(raw_text)

    reported_id = str(data.get("serial_no_read_from_image", "")).strip()
    if reported_id != expected_booth_id:
        logger.warning(f"    Crop mismatch: expected booth {expected_booth_id}, "
                        f"image itself reads Serial No. Of Polling Station '{reported_id}' - discarding this crop")
        return None

    data["booth_id"] = expected_booth_id
    data["candidate_votes"] = [data["candidate_votes_by_name"].get(name, 0) for name in candidates]
    return data


def try_fulltable_fallback(pdf_name: str, page_num: int, candidates: list,
                            schema: dict, booth_id: str) -> dict | None:
    table_path = TABLES_ENHANCED_DIR / pdf_name / f"page_{page_num}_table.png"
    if not table_path.exists():
        return None

    prompt = build_fulltable_prompt(candidates, booth_id)
    try:
        raw_text = call_gemini_with_retry(table_path, prompt, schema)
        data = json.loads(raw_text)
    except Exception as e:
        logger.error(f"    Full-table fallback failed for booth {booth_id}: {e}")
        return None

    reported_id = str(data.get("serial_no_read_from_image", "")).strip()
    if reported_id != booth_id:
        logger.error(f"    Full-table fallback also mismatched for booth {booth_id} "
                      f"(got '{reported_id}') - keeping anyway with mismatch flag")
        data["label_mismatch"] = True

    data["booth_id"] = booth_id
    data["candidate_votes"] = [data["candidate_votes_by_name"].get(name, 0) for name in candidates]
    return data


def process_booth(pdf_name: str, page_dir: Path, page_num: int, booth_id: str,
                   candidates: list, crop_schema: dict, out_dir: Path) -> str:
    out_path = out_dir / f"booth_{booth_id}.json"
    if out_path.exists():
        return "skipped"

    best_result = None
    best_error = None

    for v in VARIANT_LABELS:
        variant_path = page_dir / f"booth_{booth_id}_{v}.png"
        if not variant_path.exists():
            continue

        try:
            result = try_crop_variant(variant_path, candidates, crop_schema, booth_id)
        except Exception as e:
            logger.error(f"  Page {page_num} booth {booth_id} {v} FAILED: {e}")
            continue

        if result is None:
            continue

        valid, error_size = is_valid(result, candidates)
        result["extraction_source"] = f"row_retry_{v}_verified"

        if valid:
            result["validation_status"] = "retried_passed"
            best_result = result
            logger.info(f"  Page {page_num} booth {booth_id}: PASSED on {v} (label verified)")
            break

        if best_error is None or error_size < best_error:
            result["validation_status"] = "manual_review"
            best_result = result
            best_error = error_size

    if best_result is None or best_result.get("validation_status") != "retried_passed":
        fallback = try_fulltable_fallback(pdf_name, page_num, candidates, crop_schema, booth_id)
        if fallback is not None:
            valid, error_size = is_valid(fallback, candidates)
            fallback["extraction_source"] = "fulltable_fallback"
            if valid and not fallback.get("label_mismatch"):
                fallback["validation_status"] = "retried_passed"
                best_result = fallback
                logger.info(f"  Page {page_num} booth {booth_id}: PASSED via full-table fallback")
            elif best_error is None or error_size < best_error:
                fallback["validation_status"] = "manual_review"
                best_result = fallback
                best_error = error_size

    if best_result is None:
        logger.error(f"  Page {page_num} booth {booth_id}: no verified result from any method")
        return "no_result"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)

    if best_result["validation_status"] == "retried_passed":
        return "passed"
    else:
        logger.warning(f"  Page {page_num} booth {booth_id}: label verified correct, "
                        f"still off by {best_error} - kept for manual review")
        return "manual_review"


def collect_all_booth_tasks(pdf_name: str, failed_dir: Path, rows_dir: Path):
    tasks = []
    for failed_path in sorted(failed_dir.glob("page_*_failed.json"),
                               key=lambda p: int(p.stem.split("_")[1])):
        with open(failed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        failed_booth_ids = [str(b["booth_id"]).strip() for b in data.get("failed_booths", [])]
        if not failed_booth_ids:
            continue

        page_num = data["page"]
        page_dir = rows_dir / f"page_{page_num}"
        out_dir = RE_EXTRACTED_DIR / pdf_name / f"page_{page_num}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for booth_id in failed_booth_ids:
            tasks.append((page_num, booth_id, page_dir, out_dir))

    return tasks


def process_pdf(pdf_name: str):
    failed_dir = FAILED_BOOTHS_DIR / pdf_name
    rows_dir = FAILED_ROWS_DIR / pdf_name

    if not failed_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 05_validate.py first")
        return

    config = load_constituency_config(pdf_name)
    candidates = config["candidates"]
    crop_schema = build_single_booth_schema(candidates)

    logger.info(f"{'=' * 70}")
    logger.info(f"RETRY EXTRACTION (self-verifying, parallel): {pdf_name}")
    logger.info(f"{'=' * 70}")

    tasks = collect_all_booth_tasks(pdf_name, failed_dir, rows_dir)
    logger.info(f"[{pdf_name}] {len(tasks)} booths to retry, {MAX_WORKERS} workers")

    counts = {"passed": 0, "manual_review": 0, "no_result": 0, "skipped": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_booth, pdf_name, page_dir, page_num, booth_id,
                             candidates, crop_schema, out_dir): (page_num, booth_id)
            for page_num, booth_id, page_dir, out_dir in tasks
        }
        for future in as_completed(futures):
            page_num, booth_id = futures[future]
            try:
                status = future.result()
                counts[status] += 1
            except Exception as e:
                logger.error(f"  Page {page_num} booth {booth_id}: UNEXPECTED ERROR - {e} "
                              f"- skipping, will retry on next run")
                counts["no_result"] += 1

    logger.info(f"[{pdf_name}] Retry summary: {counts['passed']} passed, "
                f"{counts['manual_review']} need manual review, "
                f"{counts['no_result']} produced no result, "
                f"{counts['skipped']} already done")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)