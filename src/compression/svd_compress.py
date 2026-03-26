"""
SVD-based image compression (performed independently per RGB channel).

Method:
  - Convert image to float in [0, 1]
  - For each channel, compute SVD of the mean-centered matrix Xc
  - Keep rank-k approximation and reconstruct
  - k can be selected either by a target compression ratio or by retained energy

Notes:
  - This is a classical linear low-rank approximation (PCA-like when mean-centered).
  - Per-channel processing avoids mixing color channels and keeps the implementation simple.

Examples:
  # Target ~10x compression
  python svd_compress.py -i bird.jpg -o bird_svd_ratio10.png --mode ratio --value 10

  # Retain 95% energy
  python svd_compress.py -i bird.jpg -o bird_svd_e95.png --mode energy --value 0.95

  # Save a comparison image too
  python svd_compress.py -i bird.jpg -o out.png --mode ratio --value 10 --save_compare compare.png
"""

import argparse
import os
import cv2
import numpy as np


def k_from_compression_ratio(H: int, W: int, CR: float) -> int:
    """
    Select rank k from an approximate storage-based compression ratio.

    We approximate storage as:
      - Original: H*W values
      - Rank-k SVD: U(H,k) + S(k) + V(W,k)  -> k(H + W + 1)
      - Plus one scalar for the per-channel mean (used for centering)

    CR is defined as: original_size / compressed_size (e.g., CR=10 means ~10x smaller).
    """
    CR = float(CR)
    k = (H * W / CR) / (H + W + 1)
    k = int(np.floor(k))
    # Clamp to a valid rank range
    k = max(1, min(k, min(H, W)))
    return k


def k_from_energy(singular_values: np.ndarray, energy: float = 0.95) -> int:
    """
    Select the smallest k such that retained spectral energy >= `energy`.

    For SVD, energy is commonly measured as sum(s_i^2).
    """
    s2 = singular_values**2
    cum = np.cumsum(s2) / np.sum(s2)
    # +1 because searchsorted returns the first index where cum >= energy (0-based)
    k = int(np.searchsorted(cum, energy) + 1)
    return k


def svd_compress_channel(channel: np.ndarray, mode: str = "ratio", value: float = 10.0):
    """
    Compress a single 2D channel via rank-k SVD approximation.

    Args:
      channel: float array in [0, 1], shape (H, W)
      mode:
        - "ratio": `value` is the target compression ratio (e.g., 10)
        - "energy": `value` is the retained energy fraction (e.g., 0.95)

    Returns:
      recon: reconstructed channel in [0, 1]
      k_used: selected rank
      achieved_CR: approximate compression ratio achieved under the storage model
    """
    X = channel.astype(np.float32)
    H, W = X.shape

    # Mean-centering makes the low-rank approximation closer to PCA on the channel.
    mu = float(X.mean())
    Xc = X - mu

    # Full (economy) SVD: Xc = U diag(s) Vt
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    if mode == "ratio":
        k = k_from_compression_ratio(H, W, value)
    elif mode == "energy":
        k = k_from_energy(s, energy=float(value))
    else:
        raise ValueError("mode must be 'ratio' or 'energy'")

    # Rank-k reconstruction: U_k diag(s_k) V_k^T
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    Xk = (Uk * sk) @ Vtk
    recon = Xk + mu

    # Approximate achieved compression ratio under the storage model.
    original = H * W
    compressed = k * (H + W + 1) + 1  # +1 for mean
    achieved_CR = original / compressed

    recon = np.clip(recon, 0.0, 1.0)
    return recon, k, achieved_CR


def compress_rgb(img_bgr: np.ndarray, mode: str = "ratio", value: float = 10.0):
    """
    Compress an 8-bit color image by applying SVD per channel in RGB space.

    Args:
      img_bgr: uint8 OpenCV image (BGR)
      mode/value: forwarded to svd_compress_channel

    Returns:
      recon_bgr_u8: reconstructed uint8 BGR image
      info: dict with per-channel ranks and compression ratios
    """
    img = img_bgr.astype(np.float32) / 255.0

    # Convert to RGB so channels map to (R, G, B) consistently in code.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)

    rr, kr, crr = svd_compress_channel(r, mode=mode, value=value)
    gg, kg, crg = svd_compress_channel(g, mode=mode, value=value)
    bb, kb, crb = svd_compress_channel(b, mode=mode, value=value)

    recon_rgb = cv2.merge((rr, gg, bb))
    recon_rgb_u8 = (recon_rgb * 255.0).round().astype(np.uint8)
    recon_bgr_u8 = cv2.cvtColor(recon_rgb_u8, cv2.COLOR_RGB2BGR)

    info = {
        "k": (kr, kg, kb),
        "achieved_CR": (crr, crg, crb),
        "avg_achieved_CR": float(np.mean([crr, crg, crb])),
    }
    return recon_bgr_u8, info


def save_comparison(original_bgr: np.ndarray, recon_bgr: np.ndarray, out_path: str, title_suffix: str = ""):
    """
    Save a side-by-side comparison figure for reporting/debugging.
    Matplotlib is imported locally to keep CLI dependency minimal when not used.
    """
    import matplotlib.pyplot as plt

    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(orig_rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"SVD Compressed{title_suffix}")
    plt.imshow(recon_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="SVD image compression (per RGB channel).")
    ap.add_argument("-i", "--input", required=True, help="Input image path")
    ap.add_argument("-o", "--output", required=True, help="Output image path (e.g., .png/.jpg)")
    ap.add_argument(
        "--mode",
        choices=["ratio", "energy"],
        default="ratio",
        help="Compression mode: ratio (target CR) or energy (retain fraction)",
    )
    ap.add_argument(
        "--value",
        type=float,
        default=10.0,
        help="If mode=ratio: target CR (e.g., 10). If mode=energy: retained energy (e.g., 0.95).",
    )
    ap.add_argument(
        "--save_compare",
        default=None,
        help="Optional path to save a side-by-side comparison PNG",
    )

    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    recon, info = compress_rgb(img, mode=args.mode, value=args.value)

    # Create output directory if needed.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwrite(args.output, recon)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.output}")

    # Console output is kept short but includes the key parameters for reproducibility.
    print("Saved:", args.output)
    print("Mode:", args.mode, "Value:", args.value)
    print("k(R,G,B):", info["k"])
    print("achieved_CR(R,G,B):", tuple(round(x, 3) for x in info["achieved_CR"]))
    print("avg_achieved_CR:", round(info["avg_achieved_CR"], 3))

    if args.save_compare:
        title_suffix = f" ({args.mode}={args.value})"
        save_comparison(img, recon, args.save_compare, title_suffix=title_suffix)
        print("Saved comparison:", args.save_compare)


if __name__ == "__main__":
    main()