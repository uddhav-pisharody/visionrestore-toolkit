"""
Stage 4: Restore thin cracks using Telea inpainting.

Telea (Fast Marching Method) tends to work well for narrow defects where nearby
structure provides reliable context.

Outputs:
  - 03_inpaint_thin.png : image after thin-region inpainting
  - 04_overlay_mask.png : visualization of inpainted region
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
    ap.add_argument("-i", "--input", required=True, help="Original image (BGR)")
    ap.add_argument("--mask_thin", required=True, help="Thin mask (0/255)")
    ap.add_argument("-o", "--outdir", default="out_stage4")
    ap.add_argument("--radius", type=float, default=3.0,
                    help="Telea inpaint radius (typical 2–4)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    mask = cv2.imread(args.mask_thin, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise FileNotFoundError("Could not load input or mask")

    save(outdir / "01_input.png", img)
    save(outdir / "02_mask_thin.png", mask)

    # Slight dilation expands the support so the inpaint solver has enough context.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_d = cv2.dilate(mask, kernel, iterations=2)
    save(outdir / "02_mask_thin_dilated.png", mask_d)

    inpainted = cv2.inpaint(img, mask_d, args.radius, cv2.INPAINT_TELEA)
    save(outdir / "03_inpaint_thin.png", inpainted)

    overlay = img.copy()
    overlay[mask_d > 0] = (0, 0, 255)
    save(outdir / "04_overlay_mask.png", overlay)

    print("Stage 4 done.")
    print("Inspect: 03_inpaint_thin.png")


if __name__ == "__main__":
    main()