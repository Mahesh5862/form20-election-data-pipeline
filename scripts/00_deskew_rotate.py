"""
STEP 0 - ROTATE PAGES (config-driven, no OCR)

Input   : data/pages_raw/{pdf_name}/page_N.png
Output  : data/pages_rotated/{pdf_name}/page_N.png

Page 1 of each Form-20 PDF comes in correctly oriented; every other page
needs the SAME fixed rotation applied (set once per PDF in
constituency_config/{pdf_name}.json as "other_pages_rotation_degrees").
For PDFs where every page is already correct (e.g. Chennur, Bellampalli),
that value is just 0 and pages pass through unchanged (aside from a tiny
automatic straighten, see below).

After the fixed rotation, a small automatic deskew (a degree or two at most)
straightens any minor scan tilt using the table's own grid lines - this
part uses OpenCV only, no OCR involved.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from config import PAGES_RAW_DIR, PAGES_ROTATED_DIR, PDF_NAMES, load_constituency_config


def fine_deskew(pil_img: Image.Image) -> Image.Image:
    """Corrects small residual tilt (a few degrees) using table grid line angle."""
    img = np.array(pil_img.convert("L"))
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=200,
        minLineLength=img.shape[1] // 4, maxLineGap=20
    )

    if lines is None:
        return pil_img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 5:  # near-horizontal lines only
            angles.append(angle)

    if not angles:
        return pil_img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return pil_img  # already straight enough

    return pil_img.rotate(median_angle, expand=True, fillcolor="white")


def process_page(src_path: Path, dst_path: Path, fixed_rotation: int):
    img = Image.open(src_path)

    if fixed_rotation != 0:
        # PIL rotate() is counter-clockwise, so negate for a clockwise-positive convention
        img = img.rotate(-fixed_rotation, expand=True, fillcolor="white")

    img = fine_deskew(img)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)


def process_pdf(pdf_name: str):
    src_dir = PAGES_RAW_DIR / pdf_name
    dst_dir = PAGES_ROTATED_DIR / pdf_name

    if not src_dir.exists():
        print(f"[{pdf_name}] SKIP - run 01_pdf_to_images.py first")
        return

    config = load_constituency_config(pdf_name)
    other_pages_rotation = config.get("other_pages_rotation_degrees", 0)

    pages = sorted(src_dir.glob("page_*.png"),
                   key=lambda p: int(p.stem.split("_")[1]))

    print(f"\n{'=' * 70}\nROTATING: {pdf_name} ({len(pages)} pages, "
          f"rotation={other_pages_rotation} deg for pages 2+)\n{'=' * 70}")

    for page_path in pages:
        page_num = int(page_path.stem.split("_")[1])
        rotation = 0 if page_num == 1 else other_pages_rotation

        dst_path = dst_dir / page_path.name
        process_page(page_path, dst_path, rotation)
        print(f"Page {page_num}: rotated {rotation} deg -> {dst_path.name}")

    print(f"[{pdf_name}] Done")


if __name__ == "__main__":
    for pdf_name in PDF_NAMES:
        process_pdf(pdf_name)