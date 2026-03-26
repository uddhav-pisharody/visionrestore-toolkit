"""
Stage 1: Pre-processing for crack detection.

Outputs:
  - 01_gray.png      : grayscale conversion
  - 02_denoised.png  : optional denoising (median or bilateral)
  - 03_clahe.png     : contrast-limited adaptive histogram equalization (CLAHE)

Rationale:
  - Detection is performed on grayscale to simplify subsequent filtering.
  - Denoising reduces salt-and-pepper / sensor noise that can trigger false ridges.
  - CLAHE enhances local contrast, making fine cracks more visible.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    """Create output directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def save_u8(path: Path, img: np.ndarray) -> None:
    """
    Save an image as uint8.
    Accepts either uint8 or float in [0, 1] and converts if needed.
    """
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input image path")
    ap.add_argument("-o", "--outdir", default="out_stage1", help="Output directory")
    ap.add_argument("--median_ksize", type=int, default=3, help="Median blur ksize (odd). 0 disables.")
    ap.add_argument("--bilateral", action="store_true", help="Use bilateral filter instead of median")
    ap.add_argument("--bil_d", type=int, default=7, help="Bilateral diameter")
    ap.add_argument("--bil_sigma_color", type=float, default=35.0)
    ap.add_argument("--bil_sigma_space", type=float, default=7.0)
    ap.add_argument("--clahe_clip", type=float, default=2.0)
    ap.add_argument("--clahe_grid", type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    # Convert to grayscale; later stages operate in single-channel intensity.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    save_u8(outdir / "01_gray.png", gray)

    # Denoising is optional. Bilateral preserves edges; median is robust to impulse noise.
    if args.bilateral:
        den = cv2.bilateralFilter(
            gray,
            d=args.bil_d,
            sigmaColor=args.bil_sigma_color,
            sigmaSpace=args.bil_sigma_space,
        )
    else:
        if args.median_ksize and args.median_ksize > 0:
            k = int(args.median_ksize)
            if k % 2 == 0:
                k += 1  # OpenCV median requires odd kernel size.
            den = cv2.medianBlur(gray, k)
        else:
            den = gray.copy()
    save_u8(outdir / "02_denoised.png", den)

    # CLAHE boosts local contrast while limiting over-amplification (clipLimit).
    clahe = cv2.createCLAHE(
        clipLimit=args.clahe_clip,
        tileGridSize=(args.clahe_grid, args.clahe_grid),
    )
    gray_clahe = clahe.apply(den)
    save_u8(outdir / "03_clahe.png", gray_clahe)

    print("Stage 1 done.")
    print(f"Outputs in: {outdir.resolve()}")
    print("Files: 01_gray.png, 02_denoised.png, 03_clahe.png")


if __name__ == "__main__":
    main()