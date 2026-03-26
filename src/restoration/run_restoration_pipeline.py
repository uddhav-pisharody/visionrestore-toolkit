"""
Run the full defect detection + restoration pipeline (Stages 1–5).

Design choice:
  - Stages are executed as separate scripts for modularity and easier debugging.
  - A dedicated work directory stores full intermediate outputs per stage.
  - A clean 'outdir' is produced with only the most relevant artifacts for grading/reporting.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def run_cmd(cmd, cwd=None) -> None:
    """Execute a command and fail loudly with captured stdout/stderr on error."""
    print("\n>>>", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(map(str, cmd))}")
    if res.stdout.strip():
        print(res.stdout.strip())


def copy_if_exists(src: Path, dst: Path) -> None:
    """Copy a file if present (keeps pipeline robust when optional outputs are missing)."""
    if not src.exists():
        print(f"[WARN] Missing (not copied): {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full defect detection + restoration pipeline (Stages 1–5).")
    ap.add_argument("-i", "--input", required=True, help="Input image path")
    ap.add_argument("-o", "--outdir", default="pipeline_output", help="Final output directory")
    ap.add_argument("--workdir", default="_work_pipeline", help="Working directory for stage outputs")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")

    # Expose the key tunables used in detection/restoration.
    ap.add_argument("--stage2_percentile", type=float, default=96.0)
    ap.add_argument("--stage2_close", type=int, default=7)
    ap.add_argument("--stage2_min_area", type=int, default=40)
    ap.add_argument("--width_thr", type=float, default=7.0)
    ap.add_argument("--telea_radius", type=float, default=4.0)
    ap.add_argument("--ns_radius", type=float, default=8.0)
    ap.add_argument("--ns_dilate", type=int, default=2)
    args = ap.parse_args()

    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR.parent.parent

    # Resolve input path relative to the project root for consistent CLI behavior.
    inp = Path(args.input)
    inp = (PROJECT_ROOT / inp).resolve() if not inp.is_absolute() else inp.resolve()
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    outdir = Path(args.outdir)
    workdir = Path(args.workdir)

    outdir = (PROJECT_ROOT / outdir).resolve() if not outdir.is_absolute() else outdir.resolve()
    workdir = (PROJECT_ROOT / workdir).resolve() if not workdir.is_absolute() else workdir.resolve()

    # Working stage directories (full outputs per stage).
    s1 = workdir / "out_stage1"
    s2 = workdir / "out_stage2"
    s3 = workdir / "out_stage3"
    s4 = workdir / "out_stage4"
    s5 = workdir / "out_stage5"

    stage1 = THIS_DIR / "stage1_preprocess.py"
    stage2 = THIS_DIR / "stage2_detect_cracks_ridge.py"
    stage3 = THIS_DIR / "stage3_split_thin_thick.py"
    stage4 = THIS_DIR / "stage4_inpaint_thin.py"
    stage5 = THIS_DIR / "stage5_inpaint_thick.py"

    # Execute stages in order. Each stage reads the previous stage’s output.
    run_cmd([args.python, str(stage1), "-i", str(inp), "-o", str(s1)])
    run_cmd([
        args.python, str(stage2),
        "-i", str(s1 / "03_clahe.png"),
        "-o", str(s2),
        "--percentile", str(args.stage2_percentile),
        "--close", str(args.stage2_close),
        "--min_area", str(args.stage2_min_area),
    ])
    run_cmd([
        args.python, str(stage3),
        "--clahe", str(s1 / "03_clahe.png"),
        "--mask", str(s2 / "04_mask_clean.png"),
        "-o", str(s3),
        "--width_thr", str(args.width_thr),
    ])
    run_cmd([
        args.python, str(stage4),
        "-i", str(inp),
        "--mask_thin", str(s3 / "01_mask_thin.png"),
        "-o", str(s4),
        "--radius", str(args.telea_radius),
    ])
    run_cmd([
        args.python, str(stage5),
        "-i", str(s4 / "03_inpaint_thin.png"),
        "--mask_thick", str(s3 / "02_mask_thick.png"),
        "-o", str(s5),
        "--radius", str(args.ns_radius),
        "--dilate", str(args.ns_dilate),
    ])

    # Collect a small, grade-friendly set of artifacts.
    outdir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(inp, outdir / f"00_input{inp.suffix}")
    copy_if_exists(s1 / "03_clahe.png", outdir / "01_clahe.png")

    copy_if_exists(s2 / "04_mask_clean.png", outdir / "02_mask.png")
    copy_if_exists(s2 / "05_mask_overlay.png", outdir / "02_mask_overlay.png")
    copy_if_exists(s2 / "02_score_RminusG.png", outdir / "02_score_RminusG.png")

    copy_if_exists(s3 / "03_overlay_thin_thick.png", outdir / "03_thin_thick_overlay.png")
    copy_if_exists(s3 / "01_mask_thin.png", outdir / "03_mask_thin.png")
    copy_if_exists(s3 / "02_mask_thick.png", outdir / "03_mask_thick.png")

    copy_if_exists(s4 / "03_inpaint_thin.png", outdir / "04_inpaint_thin.png")
    copy_if_exists(s5 / "04_inpaint_thick.png", outdir / "05_final.png")

    print("\nPipeline complete.")
    print("Final output:", (outdir / "05_final.png").resolve())
    print("Important intermediates saved in:", outdir.resolve())
    print("Working directory:", workdir.resolve())


if __name__ == "__main__":
    main()