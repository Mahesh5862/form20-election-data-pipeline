"""
STEP 4 - GEMINI EXTRACTION (two-pass consistency check, string booth_id, parallel calls)

Input   : data/tables_enhanced/{pdf_name}/page_N_table.png
          constituency_config/{pdf_name}.json (candidate names)
Output  : data/raw_json/{pdf_name}/page_N_raw.json
          logs/04_gemini_extract_table.log
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
    GEMINI_API_KEY, MODEL_NAME, TABLES_ENHANCED_DIR, RAW_JSON_DIR,
    PDF_NAMES, load_constituency_config, LOG_DIR,
    MAX_RETRIES, RETRY_BACKOFF_SECONDS, MAX_WORKERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "04_gemini_extract_table.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

MAX_DIMENSION_FOR_API = 4000


def build_response_schema(candidates):
    return {
        "type": "object",
        "properties": {
            "page": {"type": "integer"},
            "booth_count": {"type": "integer"},
            "booths": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "booth_id": {"type": "string"},
                        "candidate_votes_by_name": {
                            "type": "object",
                            "properties": {n: {"type": "integer"} for n in candidates},
                            "required": candidates,
                        },
                        "total_valid_votes": {"type": "integer"},
                        "rejected_votes": {"type": "integer"},
                        "nota": {"type": "integer"},
                        "total_votes": {"type": "integer"},
                        "tendered_votes": {"type": "integer"},
                    },
                    "required": [
                        "booth_id", "candidate_votes_by_name", "total_valid_votes",
                        "rejected_votes", "nota", "total_votes", "tendered_votes",
                    ],
                },
            },
        },
        "required": ["page", "booth_count", "booths"],
    }


def build_prompt(page_num, candidates):
    candidate_list = "\n".join(f'  {i + 1}. "{n}"' for i, n in enumerate(candidates))
    return f"""You are extracting election Form-20 booth-wise voting data from a table image.

Candidates, in the same left-to-right order as the table columns:
{candidate_list}

Extract every booth row visible on page {page_num}.

Rules:
1. Use the exact candidate names given above as JSON keys.
2. Preserve every digit exactly as printed - do not round or guess.
3. If a cell is blank/unreadable, use 0 - do not skip the booth.
4. This table has TWO similarly-named columns near the left edge:
   "Serial No." (a simple row counter, always sequential - 1, 2, 3, 4...)
   and "Serial No. Of Polling Station" (the real polling station
   identifier, which can include a letter suffix like "117A" for a
   split/auxiliary station).
5. booth_id MUST come from "Serial No. Of Polling Station", NEVER from the
   plain "Serial No." row counter.
6. booth_id is text: plain numbers ("42") or a number with one letter
   suffix ("117A").
7. Do NOT include summary/footer rows such as "TOTAL", "Total votes
   polled", or any grand-total row - only individual polling booth rows."""


def add_votes_array(booth, candidates):
    by_name = booth.get("candidate_votes_by_name", {})
    booth["candidate_votes"] = [by_name.get(n, 0) for n in candidates]
    return booth


def load_image_for_api(image_path):
    img = Image.open(image_path)
    if max(img.size) > MAX_DIMENSION_FOR_API:
        scale = MAX_DIMENSION_FOR_API / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def call_gemini_with_retry(image_path, prompt, schema):
    img = load_image_for_api(image_path)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                [prompt, img],
                generation_config={
                    "temperature": 0, "top_p": 0.1, "top_k": 1,
                    "response_mime_type": "application/json", "response_schema": schema,
                },
            )
            return response.text
        except Exception as e:
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(f"  Gemini call failed (attempt {attempt}/{MAX_RETRIES}): {e} - retrying in {wait}s")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


def normalize_booths(data, candidates):
    result = {}
    for booth in data.get("booths", []):
        bid = str(booth["booth_id"]).strip()
        booth["booth_id"] = bid
        add_votes_array(booth, candidates)
        result[bid] = booth
    return result


def booths_match(b1, b2):
    return (
        b1["candidate_votes"] == b2["candidate_votes"]
        and b1["total_valid_votes"] == b2["total_valid_votes"]
        and b1["rejected_votes"] == b2["rejected_votes"]
        and b1["nota"] == b2["nota"]
        and b1["total_votes"] == b2["total_votes"]
        and b1["tendered_votes"] == b2["tendered_votes"]
    )


def process_page(pdf_name, table_path, page_num, candidates, schema, out_dir):
    out_path = out_dir / f"page_{page_num}_raw.json"
    if out_path.exists():
        logger.info(f"[{pdf_name}] page {page_num}: already done, skipping")
        return "skipped"

    prompt = build_prompt(page_num, candidates)
    try:
        with ThreadPoolExecutor(max_workers=2) as pair_executor:
            f1 = pair_executor.submit(call_gemini_with_retry, table_path, prompt, schema)
            f2 = pair_executor.submit(call_gemini_with_retry, table_path, prompt, schema)
            raw_text_1, raw_text_2 = f1.result(), f2.result()
    except Exception as e:
        logger.error(f"[{pdf_name}] page {page_num}: API call failed permanently - {e}")
        return "api_failed"

    try:
        data_1, data_2 = json.loads(raw_text_1), json.loads(raw_text_2)
    except json.JSONDecodeError as e:
        logger.error(f"[{pdf_name}] page {page_num}: JSON parse failed - {e}")
        return "parse_failed"

    booths_1 = normalize_booths(data_1, candidates)
    booths_2 = normalize_booths(data_2, candidates)
    all_ids = set(booths_1) | set(booths_2)
    final_booths, inconsistent_ids = [], []

    # THE FIX: booths present in only ONE pass now get an explicit tag
    # instead of silently defaulting to "matched" in Step 5.
    for bid in sorted(all_ids):
        b1, b2 = booths_1.get(bid), booths_2.get(bid)
        if b1 is None or b2 is None:
            chosen = b1 or b2
            chosen["consistency_check"] = "id_missing_in_one_pass"
            inconsistent_ids.append(bid)
            final_booths.append(chosen)
            continue
        if booths_match(b1, b2):
            b1["consistency_check"] = "matched"
            final_booths.append(b1)
        else:
            b1["consistency_check"] = "mismatched_between_passes"
            inconsistent_ids.append(bid)
            final_booths.append(b1)

    if inconsistent_ids:
        logger.warning(f"[{pdf_name}] page {page_num}: {len(inconsistent_ids)} booth(s) "
                        f"disagreed between passes: {inconsistent_ids}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"page": page_num, "booth_count": len(final_booths), "booths": final_booths},
                   f, indent=2, ensure_ascii=False)
    logger.info(f"[{pdf_name}] page {page_num}: OK - {len(final_booths)} booths "
                f"({len(inconsistent_ids)} flagged inconsistent)")
    return "ok"


def process_pdf(pdf_name):
    src_dir = TABLES_ENHANCED_DIR / pdf_name
    out_dir = RAW_JSON_DIR / pdf_name
    if not src_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 03_enhance_table.py first")
        return

    config = load_constituency_config(pdf_name)
    candidates = config["candidates"]
    schema = build_response_schema(candidates)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = sorted(src_dir.glob("page_*_table.png"), key=lambda p: int(p.stem.split("_")[1]))
    logger.info(f"{'=' * 70}")
    logger.info(f"GEMINI EXTRACTION: {pdf_name} ({len(tables)} pages, {MAX_WORKERS} workers)")
    logger.info(f"{'=' * 70}")

    results = {"ok": 0, "skipped": 0, "api_failed": 0, "parse_failed": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_page, pdf_name, tp, int(tp.stem.split("_")[1]),
                             candidates, schema, out_dir): tp
            for tp in tables
        }
        for future in as_completed(futures):
            results[future.result()] += 1

    logger.info(f"[{pdf_name}] Summary: {results}")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)