"""
Stage 5: Restore thicker damaged regions using Navier–Stokes inpainting.

Navier–Stokes inpainting propagates isophotes into the missing region and is often
more stable for wider defects than fast-marching methods.

Outputs:
  - 04_inpaint_thick.png : final restoration output
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
    ap.add_argument("-i", "--input", required=True, help="Image after Stage 4 (thin inpaint)")
    ap.add_argument("--mask_thick", required=True, help="Thick mask (0/255)")
    ap.add_argument("-o", "--outdir", default="out_stage5")
    ap.add_argument("--radius", type=float, default=7.0, help="Navier–Stokes radius (typical 6–10)")
    ap.add_argument("--dilate", type=int, default=2, help="Mask dilation iterations")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    mask = cv2.imread(args.mask_thick, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load input image: {args.input}")
    if mask is None:
        raise FileNotFoundError(f"Could not load thick mask: {args.mask_thick}")

    save(outdir / "01_input.png", img)
    save(outdir / "02_mask_thick.png", mask)

    # Dilation expands the target region to include boundary pixels around the defect.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_d = cv2.dilate(mask, kernel, iterations=args.dilate)
    save(outdir / "03_mask_thick_dilated.png", mask_d)

    inpainted = cv2.inpaint(img, mask_d, args.radius, cv2.INPAINT_NS)
    save(outdir / "04_inpaint_thick.png", inpainted)

    print("Stage 5 done.")
    print("Inspect: 04_inpaint_thick.png")


if __name__ == "__main__":
    main()