"""
BUILD PRE-FILLED MANUAL CORRECTIONS TEMPLATES

Input   : data/validation_reports/{pdf_name}/manual_review_needed.json
          data/re_extracted/{pdf_name}/page_N/booth_ID.json (full detail source)
Output  : data/manual_corrections/{pdf_name}.json (pre-filled, ready to edit)

For every booth still needing manual review, pulls the FULL best-guess
result (candidate_votes_by_name, nota, rejected_votes, etc.) from
re_extracted/ and writes it into manual_corrections/{pdf_name}.json in the
exact format Step 10 expects.

If manual_corrections/{pdf_name}.json ALREADY EXISTS, this script does NOT
overwrite it - it skips that PDF so nothing you've already done is lost.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import VALIDATION_REPORTS_DIR, RE_EXTRACTED_DIR, MANUAL_CORRECTIONS_DIR, PDF_NAMES


def load_manual_review_list(pdf_name: str) -> list:
    path = VALIDATION_REPORTS_DIR / pdf_name / "manual_review_needed.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("needs_manual_review", [])


def build_template(pdf_name: str):
    out_path = MANUAL_CORRECTIONS_DIR / f"{pdf_name}.json"
    if out_path.exists():
        print(f"[{pdf_name}] SKIP - manual_corrections file already exists, "
              f"not overwriting (in case you already fixed some booths there)")
        return

    review_list = load_manual_review_list(pdf_name)
    if not review_list:
        print(f"[{pdf_name}] No manual review needed - skipping")
        return

    corrections = []
    missing = []

    for item in review_list:
        page_num = item["page"]
        booth_id = str(item["booth_id"]).strip()

        re_extracted_path = RE_EXTRACTED_DIR / pdf_name / f"page_{page_num}" / f"booth_{booth_id}.json"
        if not re_extracted_path.exists():
            missing.append((page_num, booth_id))
            continue

        with open(re_extracted_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        corrections.append({
            "page": page_num,
            "booth_id": booth_id,
            "candidate_votes_by_name": data.get("candidate_votes_by_name", {}),
            "total_valid_votes": data.get("total_valid_votes", 0),
            "rejected_votes": data.get("rejected_votes", 0),
            "nota": data.get("nota", 0),
            "total_votes": data.get("total_votes", 0),
            "tendered_votes": data.get("tendered_votes", 0),
            "_VERIFY_AGAINST_PDF": "PRE-FILLED GUESS - check every number against the original PDF page before trusting this row"
        })

    MANUAL_CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"corrections": corrections}, f, indent=2, ensure_ascii=False)

    print(f"[{pdf_name}] Created template with {len(corrections)} booths -> {out_path}")
    if missing:
        print(f"[{pdf_name}] WARNING - {len(missing)} booths had no re_extracted file, "
              f"skipped: {missing}")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        build_template(pdf_name)