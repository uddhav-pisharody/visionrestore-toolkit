"""
Stage 2: Crack detection via multi-scale Hessian ridge response.

Outputs:
  - 02_ridge_response.png : normalized ridge response (diagnostic)
  - 02_score_RminusG.png  : ridge score with gradient penalty (diagnostic)
  - 03_mask_raw.png       : initial thresholded mask
  - 04_mask_clean.png     : cleaned final mask (binary 0/255)
  - 05_mask_overlay.png   : visualization overlay on the input

Rationale:
  - Cracks appear as elongated, high-contrast ridges.
  - A Hessian-based ridge score is computed across multiple scales to detect thin and thicker defects.
  - A gradient penalty helps suppress strong natural edges (e.g., face contours).
  - Morphology and component filtering remove noise and non-crack structures.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    """Create output directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def save(path: Path, img: np.ndarray) -> None:
    """Save image to disk (expects uint8)."""
    cv2.imwrite(str(path), img)


def ridge_response(gray_u8: np.ndarray, sigma: float) -> np.ndarray:
    """
    Bright ridge detector using Hessian eigenvalues at a given scale.
    Returns float32 response; higher values indicate ridge-like structures.
    """
    # Smooth at the target scale to build a scale-space representation.
    k = int(max(3, round(sigma * 6 + 1)))
    if k % 2 == 0:
        k += 1
    g = cv2.GaussianBlur(gray_u8, (k, k), sigmaX=sigma, sigmaY=sigma)

    # Second derivatives form the 2x2 Hessian matrix at each pixel.
    Ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)

    # Eigenvalues of Hessian (closed form for 2x2 matrices).
    tmp = np.sqrt((Ixx - Iyy) ** 2 + 4.0 * (Ixy ** 2))
    l1 = 0.5 * (Ixx + Iyy - tmp)
    l2 = 0.5 * (Ixx + Iyy + tmp)

    # Order eigenvalues such that |l1| <= |l2|.
    swap = np.abs(l1) > np.abs(l2)
    l1s = l1.copy()
    l2s = l2.copy()
    l1s[swap], l2s[swap] = l2[swap], l1[swap]
    l1, l2 = l1s, l2s

    # For bright ridges, the dominant eigenvalue tends to be negative with large magnitude.
    strength = np.maximum(0.0, -l2)

    # Penalize blob-like response using eigenvalue ratio.
    eps = 1e-6
    ratio = (np.abs(l1) / (np.abs(l2) + eps))
    score = strength * (1.0 - np.clip(ratio, 0.0, 1.0))
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Stage1 CLAHE image path")
    ap.add_argument("-o", "--outdir", default="out_stage2_ridge")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[1.0, 2.0, 3.5],
                    help="Scales for multi-scale ridge detection")
    ap.add_argument("--percentile", type=float, default=96.5,
                    help="Percentile threshold on the ridge score (higher => fewer detections)")
    ap.add_argument("--close", type=int, default=5, help="Closing kernel size (odd), e.g. 3/5/7")
    ap.add_argument("--min_area", type=int, default=20, help="Remove components smaller than this")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read: {args.input}")
    save(outdir / "01_input.png", gray)

    # Multi-scale ridge response combines sensitivity to both thin and thicker structures.
    responses = [ridge_response(gray, s) for s in args.sigmas]
    R = np.maximum.reduce(responses)

    R_norm = R / (R.max() + 1e-6)
    save(outdir / "02_ridge_response.png", (R_norm * 255).astype(np.uint8))

    # Gradient penalty suppresses strong natural edges that are not cracks.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    G = cv2.magnitude(gx, gy)
    G_norm = G / (G.max() + 1e-6)

    alpha = 0.70  # weighting between ridge response and gradient penalty
    S = np.clip(R_norm - alpha * G_norm, 0.0, 1.0)
    save(outdir / "02_score_RminusG.png", (S * 255).astype(np.uint8))

    # Percentile threshold is robust across images with different global contrast.
    thr = np.percentile(S, args.percentile)
    mask_raw = (S >= thr).astype(np.uint8) * 255
    save(outdir / "03_mask_raw.png", mask_raw)

    # Closing connects broken crack segments and fills small gaps.
    k = args.close if args.close % 2 == 1 else args.close + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)

    # Component filtering removes small noise and keeps crack-like (elongated / low-extent) shapes.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
    final_mask = np.zeros_like(mask_clean)

    min_area = args.min_area
    min_elong = 3.5   # elongation threshold
    min_extent = 0.08 # area / bbox-area threshold

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        extent = area / float(w * h + 1e-6)
        elong = max(w, h) / float(min(w, h) + 1e-6)

        # Refine elongation using an ellipse fit when possible.
        component = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
                (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
                elong = max(MA, ma) / float(min(MA, ma) + 1e-6)

        if elong >= min_elong or extent <= min_extent:
            final_mask[labels == i] = 255

    # Light dilation ensures the mask covers crack pixels around the ridge centerline.
    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.dilate(final_mask, dil, iterations=1)

    save(outdir / "04_mask_clean.png", final_mask)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[final_mask > 0] = [0, 0, 255]
    save(outdir / "05_mask_overlay.png", overlay)

    print("Stage 2 done.")
    print("Inspect: 02_ridge_response.png, 04_mask_clean.png, 05_mask_overlay.png")


if __name__ == "__main__":
    main()