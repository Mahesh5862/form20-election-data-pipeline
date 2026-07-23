# Form-20 Election Data Extraction Pipeline

An automated, self-verifying pipeline that extracts booth-level voting data from scanned Telangana Legislative Assembly Form-20 election result PDFs — converting unstructured scanned government documents into validated, structured Excel data.

Built to solve a real problem: official Form-20 result sheets exist only as scanned PDFs, with no digital source of truth. Manual transcription at this scale (hundreds of polling booths per constituency) is slow and error-prone. This pipeline automates extraction while never silently trusting a wrong answer.

---

## Results

| Metric | Value |
|---|---|
| Constituencies processed | 10 |
| Polling booth records extracted | 2,252 |
| Final validated accuracy | **98.9%** |
| Unresolved records reduced (automated recovery only) | 528 → 24 (95%+ reduction, before any manual correction) |

---

## Why This Exists

Standard OCR tools extract raw characters with no understanding of table structure. These forms contain **two similarly-named columns** — a plain row counter ("Serial No.") and the actual polling station identifier ("Serial No. Of Polling Station") — which plain OCR cannot reliably disambiguate. A vision-language model that can be given schema context and reasoning ability was a better fit than traditional OCR for this specific problem.

Because this is official election data, **accuracy matters more than speed**. The pipeline is built around the principle that the system should know what it doesn't know — every extracted value passes through multiple independent checks before being trusted.

---

## Pipeline Architecture

A 14-stage, checkpointed pipeline — each stage reads only from the previous stage's output and writes to its own folder. This means any single stage can be fixed and re-run in isolation without repeating earlier work.

```
Raw PDF
  │
  ▼
[00] Deskew & Rotate ──────────► Corrects page-level rotation
  │
  ▼
[01] PDF → Images ─────────────► Renders pages to PNG at fixed DPI
  │
  ▼
[Quality Check — Pass 1] ──────► Sharpness & contrast verdict (GOOD/MODERATE/POOR)
  │
  ▼
[02] Table Detection ──────────► Crops just the vote-count table from each page
  │
  ▼
[03] Image Enhancement ────────► CLAHE, adaptive threshold, denoise, quality-aware profiles
  │
  ▼
[04] Gemini Extraction ────────► Two independent AI reads per page, structured JSON output
  │
  ▼
[Quality Check — Pass 2] ──────► Structural row-alignment verdict
  │
  ▼
[05] Validation ───────────────► Arithmetic check + two-pass agreement check
  │
  ├──► PASSED ──────────────────────────────────┐
  │                                              │
  ▼                                              │
[06] Extract Failed Rows                         │
  │                                              │
  ▼                                              │
[07] Enhance Failed Rows (3 variants)            │
  │                                              │
  ▼                                              │
[08] Retry Extraction (label-verified)           │
  │                                              │
  ▼                                              │
[09] Validate Re-extracted                       │
  │                                              │
  ▼                                              │
[Custom] Targeted Re-retry                       │
  (small-digit-error vs large-error strategies)  │
  │                                              │
  ▼                                              │
[Manual] Correction Templates ───► Remaining discrepancies, pre-filled for review
  │                                              │
  ▼                                              ▼
[10] Merge Corrected ◄────────────────────────────
  │
  ▼
[11] Final Validation ─────────► Per-constituency accuracy %
  │
  ▼
[12] Generate Excel ───────────► Per-constituency report
  │
  ▼
[13] Master Excel ─────────────► Combined workbook, all constituencies
```

---

## Key Engineering Decisions

**Two-pass consistency checking.** Every page is read independently, twice, in parallel. If both reads agree on every field, the record passes. If they disagree, it's flagged automatically — this catches a class of error that arithmetic validation alone misses: a systematically misread value can still sum correctly while being factually wrong.

**Quality-aware image enhancement.** Rather than applying one fixed enhancement profile to every page, pages flagged weak by the quality-check step automatically get a stronger correction pass (higher CLAHE clip limit, more permissive thresholding, mild deblurring) — while already-clean pages aren't over-processed.

**Cascading, self-verifying retries.** Failed extractions aren't just retried blindly. Each retry attempt re-confirms the row label ("Serial No. Of Polling Station") before trusting any numbers from that attempt — preventing a confident-but-wrong read of a neighboring row. If cropped-row retries fail, the system falls back to showing the *entire table* and asking the model to locate the exact row by label match, sidestepping pixel-position math entirely.

**Error-magnitude-aware recovery.** Two custom recovery strategies were built beyond the pipeline's original design:
- **Small discrepancies** (likely a single misread digit) get a retry prompt that explicitly names commonly-confused digit pairs (1↔7, 3↔8, 0↔6, 5↔6, 4↔9).
- **Large discrepancies** get a full-table re-verification, and if two independent re-reads *agree with each other* but still fail arithmetic, that's treated as evidence the discrepancy exists on the **original printed form** — not as an extraction error — and flagged for a different kind of manual check (visual confirmation, not digit-hunting).

---

## Bugs Found & Fixed During Development

Documented for transparency — these were real issues diagnosed and resolved mid-project:

- **False-positive quality checks:** the scan-quality script was including each PDF's blank cover page in its sample, dragging every verdict to POOR regardless of actual table quality. Root cause found by noticing a suspiciously uniform failure pattern across otherwise-unrelated PDFs.
- **Structural-alignment check failure (twice):** first, gridline detection was reading un-enhanced images and undercounting rows; after switching to enhanced images, a second bug caused near-zero detection from double-thresholding already-binary data. Both fixed; the check was ultimately deprioritized in favor of the more reliable two-pass agreement signal.
- **Column-confusion bug in retry logic:** the retry script was asking the model to read the plain "Serial No." row-counter column instead of the real "Serial No. Of Polling Station" identifier, then comparing the two — causing near-constant false mismatches that only coincidentally "passed" when a booth's row position equaled its polling station number.
- **File mix-up:** a template-generation script was accidentally saved over the pipeline's merge script, causing an import error — diagnosed directly from the traceback and corrected.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| AI / Extraction | Google Gemini API (`gemini-2.5-flash`), structured JSON output via schema-constrained generation |
| Image Processing | OpenCV (deskew, CLAHE, adaptive thresholding, Hough line detection, morphological ops, Laplacian sharpness scoring), Pillow |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` |
| Data Interchange | JSON |
| Output | Excel (`.xlsx`) via openpyxl/pandas |
| Config / Environment | python-dotenv |
| Logging | Python `logging` module — per-stage timestamped logs |

---

## Project Structure

```
config.py                              # Central config: paths, API key, constants
constituency_config/                   # Candidate names + ballot order, per constituency
scripts/
├── 00_deskew_rotate.py
├── 01_pdf_to_images.py
├── 02_detect_table.py
├── 03_enhance_table.py
├── 04_gemini_extract_table.py
├── 05_validate.py
├── 06_extract_failed_rows.py
├── 07_enhance_failed_rows.py
├── 08_retry_gemini_extract.py
├── 09_validate_reextracted.py
├── 10_merge_corrected.py
├── 11_final_validation.py
├── 12_generate_excel.py
└── 13_master_excel.py
build_manual_corrections_templates.py  # Pre-fills manual review templates
assess_scan_quality_combined.py        # Quality pre-flight checker
requirements.txt
```

*Source PDFs and intermediate pipeline outputs (`data/`, `logs/`, `input_pdfs/`) are excluded from this repository to keep it lightweight and code-focused — see `.gitignore`.*

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

Update `PDF_NAMES` in `config.py` with your target constituencies, then run the pipeline stages in order (00 → 13).

---

## Disclaimer

This project processes publicly available official election result documents for data extraction and analysis purposes.
