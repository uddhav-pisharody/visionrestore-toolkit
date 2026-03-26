"""
Stage 3: Split detected defects into thin vs thick regions.

Method:
  - Distance transform on the binary mask provides a local radius estimate.
  - Local width is approximated as 2 * distance.
  - Pixels are classified as thin/thick based on a width threshold.

Outputs:
  - 01_mask_thin.png
  - 02_mask_thick.png
  - 03_overlay_thin_thick.png
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clahe", required=True, help="Stage1 CLAHE image path")
    ap.add_argument("--mask", required=True, help="Stage2 final mask path (binary 0/255)")
    ap.add_argument("-o", "--outdir", default="out_stage3")
    ap.add_argument("--width_thr", type=float, default=4.0,
                    help="Width threshold in pixels (<= thin, > thick)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    gray = cv2.imread(args.clahe, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if gray is None or mask is None:
        raise FileNotFoundError("Could not load clahe or mask")

    m = (mask > 0).astype(np.uint8)

    # Distance transform yields the distance to the nearest background pixel (approx radius).
    dist = cv2.distanceTransform((m * 255).astype(np.uint8), cv2.DIST_L2, 3)
    width = 2.0 * dist

    thin = (m > 0) & (width <= args.width_thr)
    thick = (m > 0) & (width > args.width_thr)

    thin_u8 = thin.astype(np.uint8) * 255
    thick_u8 = thick.astype(np.uint8) * 255

    save(outdir / "01_mask_thin.png", thin_u8)
    save(outdir / "02_mask_thick.png", thick_u8)

    # Overlay is for qualitative inspection of the split.
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[thin] = [0, 255, 255]   # thin: yellow
    overlay[thick] = [0, 0, 255]    # thick: red
    save(outdir / "03_overlay_thin_thick.png", overlay)

    print("Stage 3 done.")
    print("Inspect: 01_mask_thin.png, 02_mask_thick.png, 03_overlay_thin_thick.png")


if __name__ == "__main__":
    main()