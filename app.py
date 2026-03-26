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


# Paths

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TEMP_DIR = ROOT / "temp"

COMPRESSION_SCRIPT = SRC / "compression" / "svd_compress.py"
TONEMAP_SCRIPT = SRC / "tonemapping" / "durand_bilateral_tonemap_plus.py"
RESTORE_SCRIPT = SRC / "restoration" / "run_restoration_pipeline.py"


# Helpers

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


# Compression

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
        f"Value: {value}"
    )

    return compressed_img, comparison_img, summary


# Tone Mapping

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
        f"save_steps: {save_steps}"
    )

    return final_img, gallery, summary


# Restoration

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
        "--stage2_percentile", str(percentile),
        "--stage2_min_area", str(min_area),
        "--width_thr", str(width_thr),
        "--telea_radius", str(telea_radius),
        "--ns_radius", str(ns_radius),
    ]

    ok, logs = run_command(cmd, cwd=ROOT)
    if not ok:
        return None, None, None, f"Restoration failed.\n\n{logs}"

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
        f"ns_radius: {ns_radius}"
    )

    return final_img, mask_img, split_img, summary


# Custom Theme / CSS

custom_css = """
:root {
    --radius-xl: 18px;
    --radius-lg: 14px;
}

.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
    padding-top: 18px !important;
    padding-bottom: 24px !important;
}

.main-title {
    text-align: center;
    margin-bottom: 0.35rem;
}

.sub-title {
    text-align: center;
    color: #6b7280;
    margin-bottom: 1.2rem;
    font-size: 1rem;
}

.section-card {
    border: 1px solid #e5e7eb !important;
    border-radius: var(--radius-xl) !important;
    padding: 18px !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

.dark .section-card {
    background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(148,163,184,0.18) !important;
}

.tool-header {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.tool-desc {
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
}

.dark .tool-desc,
.dark .sub-title {
    color: #cbd5e1;
}

.gr-button-primary {
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 44px !important;
}

.gr-button-secondary {
    border-radius: 12px !important;
}

.gr-box, .gr-form, .gr-group {
    border-radius: var(--radius-lg) !important;
}

textarea, input, .wrap, .gr-textbox, .gr-dropdown, .gr-slider {
    border-radius: 12px !important;
}

.image-frame img {
    border-radius: 14px !important;
}

footer {
    visibility: hidden;
}

.tab-nav {
    border-radius: 14px !important;
    padding: 6px !important;
}

.tabitem {
    padding-top: 12px !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size="lg",
    text_size="md",
).set(
    block_title_text_weight="700",
    block_label_text_weight="600",
)


# UI

with gr.Blocks(
    title="VisionRestore Toolkit",
    fill_width=True,
) as demo:
    gr.HTML(
        """
        <div class="main-title">
            <h1>VisionRestore Toolkit</h1>
        </div>
        <div class="sub-title">
            An interface for image compression, tone mapping, and defect restoration.
        </div>
        """
    )

    with gr.Tabs():
        # Compression Tab
        with gr.Tab("Compression"):
            with gr.Column(elem_classes="section-card"):
                gr.HTML(
                    """
                    <div class="tool-header">SVD Image Compression</div>
                    <div class="tool-desc">
                        Compress an image using SVD with either target compression ratio or retained energy.
                    </div>
                    """
                )

                with gr.Row(equal_height=True):
                    comp_input = gr.Image(
                        type="numpy",
                        label="Input Image",
                        height=360,
                        elem_classes="image-frame",
                    )
                    comp_output = gr.Image(
                        type="numpy",
                        label="Compressed Output",
                        height=360,
                        elem_classes="image-frame",
                    )

                with gr.Row():
                    comp_mode = gr.Dropdown(
                        choices=["ratio", "energy"],
                        value="ratio",
                        label="Compression Mode",
                        info="Choose ratio or retained energy mode",
                    )
                    comp_value = gr.Number(
                        value=10,
                        label="Value",
                        info="Examples: ratio=10, energy=0.95",
                    )

                comp_btn = gr.Button("Run Compression", variant="primary")

                with gr.Row(equal_height=True):
                    comp_compare = gr.Image(
                        type="numpy",
                        label="Comparison View",
                        height=320,
                        elem_classes="image-frame",
                    )
                    comp_logs = gr.Textbox(
                        label="Summary / Logs",
                        lines=10,
                    )

                comp_btn.click(
                    fn=compression_ui,
                    inputs=[comp_input, comp_mode, comp_value],
                    outputs=[comp_output, comp_compare, comp_logs],
                )

        # Tone Mapping Tab
        with gr.Tab("Tone Mapping"):
            with gr.Column(elem_classes="section-card"):
                gr.HTML(
                    """
                    <div class="tool-header">Durand Bilateral Tone Mapping</div>
                    <div class="tool-desc">
                        Enhance luminance, control detail, and optionally inspect intermediate processing steps.
                    </div>
                    """
                )

                with gr.Row(equal_height=True):
                    tone_input = gr.Image(
                        type="numpy",
                        label="Input Image",
                        height=360,
                        elem_classes="image-frame",
                    )
                    tone_output = gr.Image(
                        type="numpy",
                        label="Tone Mapped Output",
                        height=360,
                        elem_classes="image-frame",
                    )

                with gr.Accordion("Tone Mapping Controls", open=True):
                    with gr.Row():
                        detail_boost = gr.Slider(
                            0.5, 3.0,
                            value=1.5,
                            step=0.1,
                            label="Detail Boost",
                        )
                        saturation = gr.Slider(
                            0.5, 2.0,
                            value=1.0,
                            step=0.1,
                            label="Saturation",
                        )
                        gamma = gr.Slider(
                            0.5, 2.0,
                            value=1.0,
                            step=0.1,
                            label="Gamma",
                        )

                    with gr.Row():
                        use_sharpen = gr.Checkbox(
                            value=True,
                            label="Enable Sharpening",
                        )
                        save_steps = gr.Checkbox(
                            value=True,
                            label="Show Intermediate Steps",
                        )

                tone_btn = gr.Button("Run Tone Mapping", variant="primary")

                tone_gallery = gr.Gallery(
                    label="Intermediate Steps",
                    columns=3,
                    height="auto",
                    preview=True,
                )

                tone_logs = gr.Textbox(
                    label="Summary / Logs",
                    lines=10,
                )

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

        # Restoration Tab
        with gr.Tab("Restoration"):
            with gr.Column(elem_classes="section-card"):
                gr.HTML(
                    """
                    <div class="tool-header">Defect Restoration Pipeline</div>
                    <div class="tool-desc">
                        Detect crack-like structures, split thin and thick regions, and restore them using inpainting.
                    </div>
                    """
                )

                restore_input = gr.Image(
                    type="numpy",
                    label="Input Image",
                    height=380,
                    elem_classes="image-frame",
                )

                with gr.Accordion("Restoration Controls", open=True):
                    with gr.Row():
                        percentile = gr.Slider(
                            90.0, 99.9,
                            value=96.5,
                            step=0.1,
                            label="Detection Percentile",
                        )
                        min_area = gr.Slider(
                            1, 200,
                            value=20,
                            step=1,
                            label="Minimum Area",
                        )
                        width_thr = gr.Slider(
                            1.0, 15.0,
                            value=5.0,
                            step=0.5,
                            label="Width Threshold",
                        )

                    with gr.Row():
                        telea_radius = gr.Slider(
                            1, 10,
                            value=3,
                            step=1,
                            label="Telea Radius",
                        )
                        ns_radius = gr.Slider(
                            1, 15,
                            value=7,
                            step=1,
                            label="Navier–Stokes Radius",
                        )

                restore_btn = gr.Button("Run Restoration", variant="primary")

                with gr.Row(equal_height=True):
                    restore_final = gr.Image(
                        type="numpy",
                        label="Final Restored Image",
                        height=320,
                        elem_classes="image-frame",
                    )
                    restore_mask = gr.Image(
                        type="numpy",
                        label="Mask Overlay",
                        height=320,
                        elem_classes="image-frame",
                    )
                    restore_split = gr.Image(
                        type="numpy",
                        label="Thin / Thick Overlay",
                        height=320,
                        elem_classes="image-frame",
                    )

                restore_logs = gr.Textbox(
                    label="Summary / Logs",
                    lines=10,
                )

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
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme, css=custom_css)