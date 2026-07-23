"""
STEP 3 - ENHANCE TABLE (ADAPTIVE, QUALITY-AWARE, PRODUCTION GRADE)

Input   : data/tables/{pdf_name}/page_N_table.png
          scan_quality_report.json (optional - if present, pages flagged
          MODERATE/POOR get a stronger enhancement profile automatically)
Output  : data/tables_enhanced/{pdf_name}/page_N_table.png
          logs/03_enhance_table.log

HONEST LIMIT: this script cannot fully compensate for a genuinely faint
source scan - that lesson from Nirmal still holds. What it CAN do is
apply proportionally stronger correction to pages already flagged weak
by assess_scan_quality_combined.py, instead of treating every page
identically. This should shrink, not eliminate, the manual-correction
queue. Genuinely POOR pages still need Step 5-7's retry/manual flow.

PIPELINE ORDER (fixed, each step exists for a specific reason):

  1. Grayscale          - color carries no information for a B/W scanned table
  2. Upscale             - Gemini reads small/faint digits far more reliably
                           at higher resolution; done BEFORE denoising so the
                           smoothing step has more real pixels to work with
  3. Fine deskew         - straightens residual tilt using the table's own
                           grid lines (a few degrees at most)
  4. Global contrast stretch (NEW) - percentile-based rescale to 0-255.
                           CLAHE alone only boosts LOCAL contrast; a
                           uniformly faint page (low global contrast, like
                           Jukkal page 2) needs this first or CLAHE has
                           little dynamic range to work with.
  5. Denoise             - bilateral filter, preserves thin digit strokes
  6. Unsharp mask (NEW, conditional) - mild deblurring on the GRAYSCALE
                           image only, applied only to pages flagged weak
                           on sharpness. This is different from the old
                           bug: sharpening a BINARY image causes ringing;
                           sharpening grayscale before thresholding does not.
  7. CLAHE contrast      - local contrast boost; stronger clip limit for
                           flagged pages, standard for clean pages
  8. Adaptive threshold  - clean black/white for vision-model parsing;
                           lower C (more permissive) for flagged pages so
                           faint strokes aren't thresholded away entirely
  9. Otsu fallback (NEW) - safety check: if adaptive threshold produces a
                           near-blank result (a real failure mode on very
                           faint pages), retry with Otsu global threshold
                           instead of silently shipping a broken image
  10. Morphological clean - removes stray speckle without distorting shape

NOTE: still deliberately NO sharpening AFTER thresholding (confirmed
ringing/ghost-edge artifacts in earlier testing).
"""

import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # our large table images are legitimate, not malicious

import cv2
import numpy as np

from config import TABLES_DIR, TABLES_ENHANCED_DIR, PDF_NAMES, LOG_DIR, BASE_DIR

# ----------------------------------------------------------------------
# TUNABLE PARAMETERS - STANDARD PROFILE (clean pages: GOOD or no report)
# ----------------------------------------------------------------------
UPSCALE_FACTOR = 2.0

BILATERAL_DIAMETER = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50

CLAHE_CLIP_LIMIT_STANDARD = 2.0
CLAHE_CLIP_LIMIT_FLAGGED = 3.5          # stronger for MODERATE/POOR pages
CLAHE_TILE_GRID_SIZE = (8, 8)

ADAPTIVE_THRESH_BLOCK_SIZE = 31          # must be odd
ADAPTIVE_THRESH_C_STANDARD = 12
ADAPTIVE_THRESH_C_FLAGGED = 7            # more permissive - keeps faint strokes

MORPH_CLOSE_KERNEL_SIZE = (2, 2)

CONTRAST_STRETCH_LOW_PCT = 2             # percentile clipping for global stretch
CONTRAST_STRETCH_HIGH_PCT = 98

UNSHARP_AMOUNT = 0.6                     # mild - avoid overshoot/ringing
UNSHARP_RADIUS = 2

# Below this, adaptive threshold is considered to have failed (near-blank)
MIN_INK_FRACTION = 0.01                  # at least 1% of pixels should be "ink"

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "03_enhance_table.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# QUALITY REPORT LOOKUP
# ----------------------------------------------------------------------
def load_flagged_pages() -> dict:
    """
    Returns {pdf_name: {page_num_str: verdict}} for pages that were
    scored MODERATE or POOR by assess_scan_quality_combined.py.
    Returns {} if no report exists yet - script still runs fine with
    the standard profile on everything in that case.
    """
    report_path = BASE_DIR / "scan_quality_report.json"
    if not report_path.exists():
        logger.info("No scan_quality_report.json found - using standard "
                     "enhancement profile for all pages. Run the quality "
                     "check first if you want adaptive per-page enhancement.")
        return {}

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    flagged = {}
    for pdf_name, data in report.items():
        pages = {}
        for entry in data.get("image_quality", []):
            if entry["verdict"] in ("MODERATE", "POOR"):
                pages[entry["page"]] = entry["verdict"]
        if pages:
            flagged[pdf_name] = pages
    return flagged


# ----------------------------------------------------------------------
# ENHANCEMENT STEPS
# ----------------------------------------------------------------------
def fine_deskew(gray: np.ndarray) -> np.ndarray:
    """Corrects small residual tilt using the table's own grid lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=150,
        minLineLength=gray.shape[1] // 3, maxLineGap=15
    )
    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 5:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.1:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def global_contrast_stretch(gray: np.ndarray) -> np.ndarray:
    """
    Percentile-based global contrast stretch. Fixes uniformly faint pages
    that CLAHE alone can't help much, since CLAHE only boosts LOCAL
    contrast within small tiles - it has little to work with if the
    whole page sits in a narrow, light gray band to begin with.
    """
    low = np.percentile(gray, CONTRAST_STRETCH_LOW_PCT)
    high = np.percentile(gray, CONTRAST_STRETCH_HIGH_PCT)
    if high <= low:
        return gray  # degenerate case, skip rather than divide by zero
    stretched = np.clip((gray.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255)
    return stretched.astype(np.uint8)


def unsharp_mask(gray: np.ndarray, amount: float, radius: int) -> np.ndarray:
    """
    Mild deblurring on GRAYSCALE, before thresholding. This is NOT the
    same operation that caused ringing before - that was sharpening the
    final BINARY image. Sharpening grayscale text before it's thresholded
    is a standard, safe deblurring technique.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), radius)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def threshold_with_fallback(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    """
    Adaptive threshold, with an Otsu fallback if the result is
    suspiciously near-blank (a real failure mode on very faint source
    pages, where adaptive thresholding can wipe out almost everything).
    """
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, c
    )
    ink_fraction = 1.0 - (np.sum(thresh == 255) / thresh.size)  # fraction of black (ink) pixels

    if ink_fraction < MIN_INK_FRACTION:
        logger.warning("    Adaptive threshold produced near-blank result "
                        f"(ink fraction {ink_fraction:.4f}) - falling back to Otsu")
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    return thresh


def enhance(img_bgr: np.ndarray, is_flagged: bool) -> np.ndarray:
    """Full enhancement pipeline. is_flagged=True applies the stronger profile."""

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    upscaled = cv2.resize(
        gray, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC
    )

    deskewed = fine_deskew(upscaled)

    stretched = global_contrast_stretch(deskewed)

    denoised = cv2.bilateralFilter(
        stretched, d=BILATERAL_DIAMETER,
        sigmaColor=BILATERAL_SIGMA_COLOR, sigmaSpace=BILATERAL_SIGMA_SPACE
    )

    if is_flagged:
        denoised = unsharp_mask(denoised, UNSHARP_AMOUNT, UNSHARP_RADIUS)

    clip_limit = CLAHE_CLIP_LIMIT_FLAGGED if is_flagged else CLAHE_CLIP_LIMIT_STANDARD
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=CLAHE_TILE_GRID_SIZE)
    contrasted = clahe.apply(denoised)

    thresh_c = ADAPTIVE_THRESH_C_FLAGGED if is_flagged else ADAPTIVE_THRESH_C_STANDARD
    thresh = threshold_with_fallback(contrasted, ADAPTIVE_THRESH_BLOCK_SIZE, thresh_c)

    kernel = np.ones(MORPH_CLOSE_KERNEL_SIZE, np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned


def process_page(src_path: Path, dst_path: Path, is_flagged: bool) -> bool:
    img_bgr = cv2.imread(str(src_path))

    if img_bgr is None:
        logger.error(f"  Could not read image: {src_path}")
        return False

    try:
        enhanced = enhance(img_bgr, is_flagged)
    except Exception as e:
        logger.error(f"  Enhancement failed on {src_path.name}: {e}")
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), enhanced)
    return True


def process_pdf(pdf_name: str, flagged_pages: dict):
    src_dir = TABLES_DIR / pdf_name
    dst_dir = TABLES_ENHANCED_DIR / pdf_name

    if not src_dir.exists():
        logger.warning(f"[{pdf_name}] SKIP - run 02_detect_table.py first")
        return

    tables = sorted(src_dir.glob("page_*_table.png"),
                     key=lambda p: int(p.stem.split("_")[1]))

    if not tables:
        logger.warning(f"[{pdf_name}] No table images found in {src_dir}")
        return

    pdf_flagged = flagged_pages.get(pdf_name, {})

    logger.info(f"{'=' * 70}")
    logger.info(f"ENHANCING TABLES: {pdf_name} ({len(tables)} pages)"
                + (f" - {len(pdf_flagged)} page(s) flagged for stronger profile"
                   if pdf_flagged else ""))
    logger.info(f"{'=' * 70}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for table_path in tables:
        page_num = table_path.stem.split("_")[1]
        is_flagged = page_num in pdf_flagged
        dst_path = dst_dir / table_path.name
        if process_page(table_path, dst_path, is_flagged):
            success_count += 1
            tag = f" [STRONG PROFILE - {pdf_flagged[page_num]}]" if is_flagged else ""
            logger.info(f"Enhanced: {dst_path.name}{tag}")

    logger.info(f"[{pdf_name}] Done - {success_count}/{len(tables)} pages enhanced")


if __name__ == "__main__":
    flagged_pages = load_flagged_pages()
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name, flagged_pages)