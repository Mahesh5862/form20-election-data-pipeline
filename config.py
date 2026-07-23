import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_NAME = "models/gemini-2.5-flash"

BASE_DIR = Path(__file__).resolve().parent
INPUT_PDFS_DIR = BASE_DIR / "input_pdfs"
CONSTITUENCY_CONFIG_DIR = BASE_DIR / "constituency_config"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PAGES_RAW_DIR = DATA_DIR / "pages_raw"
PAGES_ROTATED_DIR = DATA_DIR / "pages_rotated"
TABLES_DIR = DATA_DIR / "tables"
TABLES_ENHANCED_DIR = DATA_DIR / "tables_enhanced"
RAW_JSON_DIR = DATA_DIR / "raw_json"
FAILED_BOOTHS_DIR = DATA_DIR / "failed_booths"
FAILED_ROWS_DIR = DATA_DIR / "failed_rows"
RE_EXTRACTED_DIR = DATA_DIR / "re_extracted"
CORRECTED_JSON_DIR = DATA_DIR / "corrected_json"
VALIDATION_REPORTS_DIR = DATA_DIR / "validation_reports"
EXCEL_REPORTS_DIR = DATA_DIR / "excel_reports"
MASTER_REPORT_DIR = DATA_DIR / "master_report"
MANUAL_CORRECTIONS_DIR = DATA_DIR / "manual_corrections"

# ============================================================
# BATCH 4: Jagtial/Karimnagar/Peddapalle district constituencies
# (21, 23-30) plus 114-Madhira(SC) (Khammam district). 22 skipped
# per available source PDFs. Batch 3 (11-16,19-20,115,118,119) is
# validated and archived in data/master_report/All_Constituencies_Master.xlsx
# Do NOT reuse old batch names here - 13_master_excel.py appends
# by matching PDF_NAMES, so stale entries duplicate/skip sheets.
# ============================================================
PDF_NAMES = [
    "21-Jagtial",
    "23-Ramagundam",
    "24-Manthani",
    "25-Peddapalle",
    "27-Choppadandi",
    "28-Vemulawada",
    "29-Sircilla",
    "30-Manakondur(SC)",
    "111-Yellandu",
    "114-Madhira",
]

RENDER_DPI = 700  # baseline; raise selectively only for PDFs the
                   # quality check flags as POOR on SHARPNESS specifically

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
MAX_WORKERS = 5


def load_constituency_config(pdf_name: str) -> dict:
    import json
    path = CONSTITUENCY_CONFIG_DIR / f"{pdf_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def natural_sort_key(booth_id: str):
    """Sorts '2' before '10', and '117' before '117A'."""
    import re
    match = re.match(r"(\d+)([A-Za-z]*)", str(booth_id))
    if match:
        return (int(match.group(1)), match.group(2))
    return (999999, str(booth_id))