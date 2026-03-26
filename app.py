from __future__ import annotations

import sys
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TEMP_DIR = ROOT / "temp"

COMPRESSION_SCRIPT = SRC / "compression" / "svd_compress.py"
TONEMAP_SCRIPT = SRC / "tonemapping" / "durand_bilateral_tonemap_plus.py"
RESTORE_SCRIPT = SRC / "restoration" / "run_restoration_pipeline.py"


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(prefix: str) -> Path:
    run_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    run_dir = TEMP_DIR / run_id
    ensure_dir(run_dir)
    return run_dir


def save_uploaded_image(img: np.ndarray, out_path: Path) -> None:
    if img is None:
        raise ValueError("No image uploaded.")

    if img.ndim == 2:
        cv2.imwrite(str(out_path), img)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)


def load_image_for_gradio(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if img.ndim == 2:
        return img

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def first_existing_image(paths: List[Path]) -> Optional[np.ndarray]:
    for p in paths:
        if p.exists():
            return load_image_for_gradio(p)
    return None


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return result.returncode == 0, output.strip()
    except Exception as e:
        return False, str(e)


def safe_cleanup_old_temp(max_runs: int = 30) -> None:
    if not TEMP_DIR.exists():
        return

    dirs = [p for p in TEMP_DIR.iterdir() if p.is_dir()]
    if len(dirs) <= max_runs:
        return

    dirs.sort(key=lambda p: p.stat().st_mtime)
    for old_dir in dirs[:-max_runs]:
        shutil.rmtree(old_dir, ignore_errors=True)


def check_script_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")


# ============================================================
# Compression
# ============================================================

def compression_ui(
    image: np.ndarray,
    mode: str,
    value: float,
):
    safe_cleanup_old_temp()
    run_dir = make_run_dir("compress")

    input_path = run_dir / "input.png"
    output_path = run_dir / "compressed.png"
    compare_path = run_dir / "comparison.png"

    save_uploaded_image(image, input_path)
    check_script_exists(COMPRESSION_SCRIPT)

    cmd = [
        sys.executable,
        str(COMPRESSION_SCRIPT),
        "-i", str(input_path),
        "-o", str(output_path),
        "--mode", mode,
        "--value", str(value),
        "--save_compare", str(compare_path),
    ]

    ok, logs = run_command(cmd, cwd=ROOT)
    if not ok:
        return None, None, f"Compression failed.\n\n{logs}"

    compressed_img = load_image_for_gradio(output_path) if output_path.exists() else None
    comparison_img = load_image_for_gradio(compare_path) if compare_path.exists() else None

    summary = (
        "Compression completed.\n\n"
        f"Mode: {mode}\n"
        f"Value: {value}\n\n"
        f"Logs:\n{logs}"
    )

    return compressed_img, comparison_img, summary


# ============================================================
# Tone Mapping
# ============================================================

def tonemap_ui(
    image: np.ndarray,
    detail_boost: float,
    saturation: float,
    gamma: float,
    use_sharpen: bool,
    save_steps: bool,
):
    safe_cleanup_old_temp()
    run_dir = make_run_dir("tonemap")

    input_path = run_dir / "input.png"
    output_path = run_dir / "tonemapped.png"
    steps_dir = ensure_dir(run_dir / "steps")

    save_uploaded_image(image, input_path)
    check_script_exists(TONEMAP_SCRIPT)

    cmd = [
        sys.executable,
        str(TONEMAP_SCRIPT),
        "-i", str(input_path),
        "-o", str(output_path),
        "--detail_boost", str(detail_boost),
        "--saturation", str(saturation),
        "--gamma", str(gamma),
        "--out_dir", str(steps_dir),
    ]

    if save_steps:
        cmd.append("--save_steps")

    if use_sharpen:
        cmd += ["--sharpen_amount", "1.0"]
    else:
        cmd += ["--sharpen_amount", "0.0"]

    ok, logs = run_command(cmd, cwd=ROOT)
    if not ok:
        return None, [], f"Tone mapping failed.\n\n{logs}"

    final_img = load_image_for_gradio(output_path) if output_path.exists() else None

    gallery = []
    if save_steps and steps_dir.exists():
        for p in sorted(steps_dir.glob("*.png")):
            try:
                gallery.append(load_image_for_gradio(p))
            except Exception:
                pass

    summary = (
        "Tone mapping completed.\n\n"
        f"detail_boost: {detail_boost}\n"
        f"saturation: {saturation}\n"
        f"gamma: {gamma}\n"
        f"sharpen: {use_sharpen}\n"
        f"save_steps: {save_steps}\n\n"
        f"Logs:\n{logs}"
    )

    return final_img, gallery, summary


# ============================================================
# Restoration
# ============================================================

def restoration_ui(
    image: np.ndarray,
    percentile: float,
    min_area: int,
    width_thr: float,
    telea_radius: int,
    ns_radius: int,
):
    safe_cleanup_old_temp()
    run_dir = make_run_dir("restore")

    input_path = run_dir / "input.png"
    out_dir = ensure_dir(run_dir / "outputs")
    work_dir = ensure_dir(run_dir / "work")

    save_uploaded_image(image, input_path)
    check_script_exists(RESTORE_SCRIPT)

    cmd = [
        sys.executable,
        str(RESTORE_SCRIPT),
        "-i", str(input_path),
        "-o", str(out_dir),
        "--workdir", str(work_dir),
        "--percentile", str(percentile),
        "--min_area", str(min_area),
        "--width_thr", str(width_thr),
        "--telea_radius", str(telea_radius),
        "--ns_radius", str(ns_radius),
    ]

    ok, logs = run_command(cmd, cwd=ROOT)
    if not ok:
        return None, None, None, f"Restoration failed.\n\n{logs}"

    # Adjust these names if your runner exports different filenames
    final_img = first_existing_image([
        out_dir / "05_final.png",
        out_dir / "final.png",
    ])

    mask_img = first_existing_image([
        out_dir / "02_mask_overlay.png",
        out_dir / "mask_overlay.png",
    ])

    split_img = first_existing_image([
        out_dir / "03_thin_thick_overlay.png",
        out_dir / "thin_thick_overlay.png",
    ])

    summary = (
        "Restoration completed.\n\n"
        f"percentile: {percentile}\n"
        f"min_area: {min_area}\n"
        f"width_thr: {width_thr}\n"
        f"telea_radius: {telea_radius}\n"
        f"ns_radius: {ns_radius}\n\n"
        f"Logs:\n{logs}"
    )

    return final_img, mask_img, split_img, summary


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="VisionRestore Toolkit") as demo:
    gr.Markdown(
        """
        # VisionRestore Toolkit
        A basic UI for image compression, tone mapping, and defect restoration.
        """
    )

    with gr.Tabs():
        with gr.Tab("Compression"):
            with gr.Row():
                comp_input = gr.Image(type="numpy", label="Input Image")
                comp_output = gr.Image(type="numpy", label="Compressed Output")

            with gr.Row():
                comp_mode = gr.Dropdown(
                    choices=["ratio", "energy"],
                    value="ratio",
                    label="Compression Mode",
                )
                comp_value = gr.Number(
                    value=10,
                    label="Value (e.g. ratio=10 or energy=0.95)",
                )

            comp_compare = gr.Image(type="numpy", label="Comparison (Optional)")
            comp_logs = gr.Textbox(label="Summary / Logs", lines=12)
            comp_btn = gr.Button("Run Compression")

            comp_btn.click(
                fn=compression_ui,
                inputs=[comp_input, comp_mode, comp_value],
                outputs=[comp_output, comp_compare, comp_logs],
            )

        with gr.Tab("Tone Mapping"):
            with gr.Row():
                tone_input = gr.Image(type="numpy", label="Input Image")
                tone_output = gr.Image(type="numpy", label="Tone Mapped Output")

            with gr.Row():
                detail_boost = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Detail Boost")
                saturation = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Saturation")
                gamma = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Gamma")

            with gr.Row():
                use_sharpen = gr.Checkbox(value=True, label="Enable Sharpening")
                save_steps = gr.Checkbox(value=True, label="Show Intermediate Steps")

            tone_gallery = gr.Gallery(label="Intermediate Steps", columns=3, height="auto")
            tone_logs = gr.Textbox(label="Summary / Logs", lines=12)
            tone_btn = gr.Button("Run Tone Mapping")

            tone_btn.click(
                fn=tonemap_ui,
                inputs=[
                    tone_input,
                    detail_boost,
                    saturation,
                    gamma,
                    use_sharpen,
                    save_steps,
                ],
                outputs=[tone_output, tone_gallery, tone_logs],
            )

        with gr.Tab("Restoration"):
            restore_input = gr.Image(type="numpy", label="Input Image")

            with gr.Row():
                percentile = gr.Slider(90.0, 99.9, value=96.5, step=0.1, label="Detection Percentile")
                min_area = gr.Slider(1, 200, value=20, step=1, label="Minimum Area")
                width_thr = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="Width Threshold")

            with gr.Row():
                telea_radius = gr.Slider(1, 10, value=3, step=1, label="Telea Radius")
                ns_radius = gr.Slider(1, 15, value=7, step=1, label="Navier-Stokes Radius")

            restore_btn = gr.Button("Run Restoration")

            with gr.Row():
                restore_final = gr.Image(type="numpy", label="Final Restored Image")
                restore_mask = gr.Image(type="numpy", label="Mask Overlay")
                restore_split = gr.Image(type="numpy", label="Thin/Thick Overlay")

            restore_logs = gr.Textbox(label="Summary / Logs", lines=14)

            restore_btn.click(
                fn=restoration_ui,
                inputs=[
                    restore_input,
                    percentile,
                    min_area,
                    width_thr,
                    telea_radius,
                    ns_radius,
                ],
                outputs=[
                    restore_final,
                    restore_mask,
                    restore_split,
                    restore_logs,
                ],
            )


if __name__ == "__main__":
    ensure_dir(TEMP_DIR)
    demo.launch(server_name="0.0.0.0", server_port=7860)