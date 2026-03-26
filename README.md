# VisionRestore Toolkit

A modular computer vision toolkit for:

- Image compression (SVD-based)
- Tone mapping (Durand bilateral method)
- Defect restoration (multi-stage pipeline)

## Features

- RGB SVD compression with controllable rank
- Log-luminance tone mapping with detail enhancement
- Crack detection using multi-scale Hessian filtering
- Adaptive inpainting using Telea and Navier–Stokes methods

## Run UI

```bash
python app.py