import argparse
import os
import cv2
import numpy as np
from typing import Optional


# =============================================================================
# I/O + numeric helpers
# =============================================================================

def read_bgr(path: str) -> np.ndarray:
    """Read an image from disk using OpenCV (BGR uint8)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_float01_u8(bgr_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 image to float32 in [0, 1]."""
    return bgr_u8.astype(np.float32) / 255.0


def to_u8_float01(img01: np.ndarray) -> np.ndarray:
    """Convert float image in [0, 1] to uint8."""
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def rgb_minmax(img01: np.ndarray) -> tuple[float, float]:
    """Return global min/max (as floats) for an RGB or grayscale float image."""
    mn = float(np.min(img01))
    mx = float(np.max(img01))
    return mn, mx


def normalize01(img: np.ndarray) -> np.ndarray:
    """
    Normalize an array to [0, 1] using min-max scaling.
    Used only for saving intermediate visualizations when values are outside [0, 1].
    """
    mn, mx = rgb_minmax(img)
    if mx - mn < 1e-12:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def save_step_gray01(out_dir: Optional[str], idx: int, name: str, gray01: np.ndarray) -> None:
    """
    Save a grayscale image for debugging/reporting.
    If the input is not in [0, 1] (e.g., log-domain signals), it is normalized for display.
    """
    assert out_dir is not None
    g = gray01.astype(np.float32)

    # Many intermediate tensors live in log space; normalize only for visualization.
    if g.min() < -1e-3 or g.max() > 1.0 + 1e-3:
        g = normalize01(g)

    g = np.clip(g, 0.0, 1.0)
    g_u8 = to_u8_float01(g)
    bgr = cv2.cvtColor(g_u8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(out_dir, f"{idx:02d}_{name}.png"), bgr)


def save_step_rgb01(out_dir: Optional[str], idx: int, name: str, rgb01: np.ndarray) -> None:
    """Save an RGB float image in [0, 1] as PNG (converted to BGR for OpenCV)."""
    assert out_dir is not None
    bgr = cv2.cvtColor(to_u8_float01(rgb01), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, f"{idx:02d}_{name}.png"), bgr)


# =============================================================================
# Durand-style tone mapping in log-luminance
# =============================================================================

def rgb_to_luminance(rgb01: np.ndarray) -> np.ndarray:
    """
    Convert RGB (float, [0,1]) to luminance using BT.601 weights:
      Y = 0.299 R + 0.587 G + 0.114 B
    """
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb01 * w).sum(axis=2).astype(np.float32)


def bilateral_base_logL(
    logL: np.ndarray,
    d: int,
    sigma_space: float,
    sigma_range: float,
    passes: int = 1
) -> np.ndarray:
    """
    Estimate the 'base' layer by bilateral filtering in log-luminance.

    Parameters follow OpenCV bilateralFilter:
      - d: kernel diameter
      - sigmaColor: range sigma (here: in log-domain units)
      - sigmaSpace: spatial sigma in pixels

    Repeating the filter can strengthen the base smoothing while preserving edges.
    """
    base = logL.astype(np.float32)
    for _ in range(max(1, passes)):
        base = cv2.bilateralFilter(
            base,
            d=d,
            sigmaColor=float(sigma_range),
            sigmaSpace=float(sigma_space),
        )
    return base


def compress_base_contrast(base: np.ndarray, target_log_range: float = 2.0) -> np.ndarray:
    """
    Compress the dynamic range of the base layer in log space.

    This maps the base range approximately to `target_log_range` while anchoring
    around the maximum to reduce highlight drift.
    """
    bmin = float(np.min(base))
    bmax = float(np.max(base))
    rng = max(bmax - bmin, 1e-6)
    scale = float(target_log_range) / rng
    out = (base - bmax) * scale + bmax
    return out.astype(np.float32)


def highlight_rolloff(L01: np.ndarray, knee: float = 0.85, strength: float = 0.6) -> np.ndarray:
    """
    Apply a smooth highlight compression in linear luminance space.

    - knee: threshold (0..1) above which compression starts
    - strength: 0..1, controls how aggressively highlights are rolled off
    """
    L = np.clip(L01, 0.0, 1.0).astype(np.float32)
    if strength <= 0:
        return L

    # Remap (knee..1) to (0..1) and apply a smooth saturating curve.
    x = np.maximum(L - knee, 0.0) / max(1e-6, (1.0 - knee))
    a = 1.0 + 9.0 * float(strength)
    y = (1.0 - np.exp(-a * x)) / max(1e-6, (1.0 - np.exp(-a)))
    out = np.where(L > knee, knee + y * (1.0 - knee), L)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def boost_detail_with_gating(
    detail_log: np.ndarray,
    base_log: np.ndarray,
    boost: float = 1.0,
    shadow_thresh: float = 0.20,
    shadow_min_gain: float = 0.25,
    detail_clip: float = 0.35,
) -> np.ndarray:
    """
    Boost the detail layer in log space with shadow gating.

    Rationale:
      - Detail amplification in dark regions tends to boost noise.
      - Gate the amplification using the base brightness estimate.
      - Clip extreme detail values to reduce halo artifacts.
    """
    if boost <= 1.0:
        return detail_log.astype(np.float32)

    # Base luminance estimate in linear space; normalized to [0, 1] for gating.
    base_lin = np.exp(base_log).astype(np.float32)
    b = np.clip(base_lin, 0.0, 1.0)

    # Smoothstep ramp from shadows to midtones.
    ramp = np.clip(b / max(1e-6, shadow_thresh), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)

    gate = float(shadow_min_gain) + (1.0 - float(shadow_min_gain)) * ramp
    eff_boost = 1.0 + (float(boost) - 1.0) * gate

    d = detail_log.astype(np.float32)
    if detail_clip is not None and detail_clip > 0:
        d = np.clip(d, -float(detail_clip), float(detail_clip))

    return (d * eff_boost).astype(np.float32)


def rescale_rgb_by_luminance(
    rgb01: np.ndarray,
    L_orig: np.ndarray,
    L_new: np.ndarray,
    saturation: float = 0.6,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Reconstruct color by scaling RGB using the luminance ratio:
      RGB' = RGB * (L_new / L_orig)^saturation

    `saturation` < 1 reduces chroma shifts for aggressive tone mapping.
    """
    ratio = (L_new + eps) / (L_orig + eps)
    ratio_s = np.power(ratio, float(saturation)).astype(np.float32)
    out = rgb01 * ratio_s[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def unsharp_mask_luma(rgb01: np.ndarray, amount=0.6, radius=1.2, threshold=0.015) -> np.ndarray:
    """
    Unsharp masking on luminance only (preserves chroma).

    Thresholding avoids sharpening low-amplitude variations that are often noise.
    """
    L = (rgb01[..., 0] * 0.299 + rgb01[..., 1] * 0.587 + rgb01[..., 2] * 0.114).astype(np.float32)
    L_blur = cv2.GaussianBlur(L, (0, 0), sigmaX=radius, sigmaY=radius)
    hi = L - L_blur

    # Suppress micro-contrast to reduce noise sharpening.
    m = np.abs(hi) > float(threshold)
    hi = hi * m.astype(np.float32)

    L_sharp = np.clip(L + float(amount) * hi, 0.0, 1.0)

    # Inject sharpened luminance back by scaling RGB with the luminance ratio.
    eps = 1e-6
    ratio = (L_sharp + eps) / (L + eps)
    out = np.clip(rgb01 * ratio[..., None], 0.0, 1.0).astype(np.float32)
    return out


def durand_tonemap_plus(
    bgr_u8: np.ndarray,
    # Bilateral parameters in log-luminance
    d: int = 9,
    sigma_space: float = 6.0,
    sigma_range: float = 0.35,
    passes: int = 2,
    # Base contrast compression target (log domain)
    target_log_range: float = 2.2,
    # Detail control
    detail_boost: float = 1.15,
    detail_clip: float = 0.30,
    shadow_thresh: float = 0.18,
    shadow_min_gain: float = 0.25,
    # Highlight control (linear domain)
    highlight_knee: float = 0.85,
    highlight_strength: float = 0.65,
    # Color and display
    saturation: float = 0.6,
    gamma: float = 1.0,
    # Optional post-process sharpening
    sharpen_amount: float = 0.55,
    sharpen_radius: float = 1.1,
    sharpen_threshold: float = 0.012,
    enable_sharpen: bool = True,
    # Debug outputs
    save_steps: bool = False,
    out_dir: str | None = None,
) -> np.ndarray:
    """
    Durand-style tone mapping with additional controls for detail, shadows and highlights.

    Pipeline overview:
      1) Convert to RGB float and compute luminance L
      2) Work in log-luminance: logL = log(L + eps)
      3) Bilateral filter logL -> base; detail = logL - base
      4) Compress base range; optionally boost gated detail
      5) Recombine, exponentiate, percentile-normalize, apply highlight rolloff
      6) Reconstruct RGB by luminance ratio scaling
      7) Optional luminance-only unsharp mask + optional gamma
    """
    if save_steps:
        if out_dir is None:
            raise ValueError("save_steps=True requires out_dir.")
        os.makedirs(out_dir, exist_ok=True)

    # OpenCV loads BGR; convert to RGB float in [0, 1].
    rgb01 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    if save_steps:
        save_step_rgb01(out_dir, 0, "input_rgb", rgb01)

    L = rgb_to_luminance(rgb01)
    if save_steps:
        save_step_gray01(out_dir, 1, "luminance_L", L)

    # Avoid log(0); eps is small enough to not affect normal pixels.
    eps = 1e-6
    logL = np.log(L + eps).astype(np.float32)
    if save_steps:
        save_step_gray01(out_dir, 2, "logL_vis", logL)

    base = bilateral_base_logL(logL, d=d, sigma_space=sigma_space, sigma_range=sigma_range, passes=passes)
    detail = (logL - base).astype(np.float32)

    if save_steps:
        save_step_gray01(out_dir, 3, "base_log_vis", base)
        save_step_gray01(out_dir, 4, "detail_log_vis", detail)

    base_c = compress_base_contrast(base, target_log_range=target_log_range)
    if save_steps:
        save_step_gray01(out_dir, 5, "base_compressed_log_vis", base_c)

    detail_b = boost_detail_with_gating(
        detail,
        base_log=base,
        boost=detail_boost,
        shadow_thresh=shadow_thresh,
        shadow_min_gain=shadow_min_gain,
        detail_clip=detail_clip,
    )
    if save_steps:
        save_step_gray01(out_dir, 6, "detail_boosted_log_vis", detail_b)

    # Recombine in log domain, then return to linear luminance.
    logL_new = (base_c + detail_b).astype(np.float32)
    L_new = np.exp(logL_new).astype(np.float32)

    if save_steps:
        save_step_gray01(out_dir, 7, "L_new_linear_pre_norm", L_new)

    # Percentile normalization is more robust than min-max for outliers.
    p = float(np.percentile(L_new, 99.5))
    p = float(max(p, 1e-6))
    L_new = np.clip(L_new / p, 0.0, 1.0)

    if save_steps:
        save_step_gray01(out_dir, 8, "L_new_post_percentile_norm", L_new)

    L_new = highlight_rolloff(L_new, knee=highlight_knee, strength=highlight_strength)
    if save_steps:
        save_step_gray01(out_dir, 9, "L_new_post_highlight_rolloff", L_new)

    rgb_out = rescale_rgb_by_luminance(rgb01, L_orig=L, L_new=L_new, saturation=saturation)
    if save_steps:
        save_step_rgb01(out_dir, 10, "rgb_post_rescale", rgb_out)

    if enable_sharpen and sharpen_amount > 0:
        rgb_out = unsharp_mask_luma(
            rgb_out,
            amount=sharpen_amount,
            radius=sharpen_radius,
            threshold=sharpen_threshold,
        )
        if save_steps:
            save_step_rgb01(out_dir, 11, "rgb_post_sharpen", rgb_out)

    if gamma is not None and abs(float(gamma) - 1.0) > 1e-6:
        rgb_out = np.power(np.clip(rgb_out, 0.0, 1.0), float(gamma)).astype(np.float32)
        if save_steps:
            save_step_rgb01(out_dir, 12, "rgb_post_gamma", rgb_out)

    bgr_out = cv2.cvtColor(to_u8_float01(rgb_out), cv2.COLOR_RGB2BGR)
    return bgr_out


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Durand-style bilateral tone mapping + detail control (Python).")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)

    ap.add_argument("--d", type=int, default=9)
    ap.add_argument("--sigma_space", type=float, default=6.0)
    ap.add_argument("--sigma_range", type=float, default=0.35)
    ap.add_argument("--passes", type=int, default=2)

    ap.add_argument("--target_log_range", type=float, default=2.2)

    ap.add_argument("--detail_boost", type=float, default=1.15)
    ap.add_argument("--detail_clip", type=float, default=0.30)
    ap.add_argument("--shadow_thresh", type=float, default=0.18)
    ap.add_argument("--shadow_min_gain", type=float, default=0.25)

    ap.add_argument("--highlight_knee", type=float, default=0.85)
    ap.add_argument("--highlight_strength", type=float, default=0.65)

    ap.add_argument("--saturation", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=1.0)

    ap.add_argument("--no_sharpen", action="store_true", help="Disable post tone-map sharpening.")
    ap.add_argument("--sharpen_amount", type=float, default=0.7, help="Sharpen strength (typical 0.3–0.8).")
    ap.add_argument("--sharpen_radius", type=float, default=1.1, help="Gaussian sigma for unsharp mask (typical 0.8–1.6).")
    ap.add_argument("--sharpen_threshold", type=float, default=0.015, help="Threshold to avoid sharpening noise (typical 0.01–0.03).")

    ap.add_argument("--save_steps", action="store_true", help="Save intermediate steps to a folder.")
    ap.add_argument("--out_dir", default=None, help="Folder for intermediate outputs (used with --save_steps).")

    args = ap.parse_args()

    img = read_bgr(args.input)

    # Default intermediate-output directory derived from the output filename.
    out_dir = args.out_dir
    if args.save_steps and (out_dir is None or out_dir.strip() == ""):
        base = os.path.splitext(os.path.basename(args.output))[0]
        out_dir = os.path.join(os.path.dirname(args.output) or ".", base + "_steps")

    out = durand_tonemap_plus(
        img,
        d=args.d,
        sigma_space=args.sigma_space,
        sigma_range=args.sigma_range,
        passes=args.passes,
        target_log_range=args.target_log_range,
        detail_boost=args.detail_boost,
        detail_clip=args.detail_clip,
        shadow_thresh=args.shadow_thresh,
        shadow_min_gain=args.shadow_min_gain,
        highlight_knee=args.highlight_knee,
        highlight_strength=args.highlight_strength,
        saturation=args.saturation,
        gamma=args.gamma,
        enable_sharpen=(not args.no_sharpen),
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius,
        sharpen_threshold=args.sharpen_threshold,
        save_steps=args.save_steps,
        out_dir=out_dir,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ok = cv2.imwrite(args.output, out)
    if not ok:
        raise RuntimeError(f"Failed to write output: {args.output}")
    print("Saved:", args.output)
    if args.save_steps:
        print("Saved steps to:", out_dir)


if __name__ == "__main__":
    main()