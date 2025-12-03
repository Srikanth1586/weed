# AI Weed Detection — Repository Overview, Setup & Usage

This repository contains an Ultralytics YOLO-based project for real-time weed detection using a webcam and a convenience script to test on saved images. The main features include:
- Real-time detection showing only "weed" bounding boxes.
- Saving single frames and recording video from the webcam.
- Console output of bounding boxes including center coordinates for each detected weed.
- A small test runner to try detection on sample images.

This README documents repository layout, purpose of major files, setup, how to run everything and troubleshooting tips.

---

## Table of contents
- Project structure
- Key files & purpose
- Requirements
- Virtual environment (recommended)
- Setup (Windows)
- Running the code
  - Test on sample images
  - Real-time webcam detection
  - Keyboard controls & behavior
- Tips for video saving & codec issues
- GPU vs CPU usage
- Customization
- Troubleshooting & FAQs
- Contributors / License / Next steps

---

## Project structure (high level)
Assuming root folder: `c:\Users\Chivukula\Projects\weed`

- scripts/
  - webcam_detection.py
    - Main real-time detection script (webcam capture, detection, drawing boxes, save frame/video).
- weed-detector/
  - Typical Ultralytics project layout (not fully listed here):
    - results/ — model training output and saved artifacts
      - train/weights/best.pt — (expected) trained model weights used by the script
    - data/processed/images/test/ — sample test images used by the test-utility in the script
- README.md — (this file)
- (Other files/folders you may have)
  - requirements.txt (recommended) — list of python deps (create if not present)
  - .vscode/, docs/, scripts/ etc.

If your tree differs, adjust paths accordingly. The default model path in the script expects:
`weed-detector/results/train/weights/best.pt`

---

## Key files & purpose
- scripts/webcam_detection.py
  - Loads a YOLO model from `model_path` (default points into `weed-detector/results/train/weights/best.pt`).
  - Offers:
    - test_weed_detection_with_images() — test detection on example images.
    - run_detection() — open camera, run real-time detection, draw boxes and center points only for class `weed` (class 0).
  - Saves output frames and recordings under `weed-detector/results`.
  - Keyboard controls:
    - `q` — quit
    - `s` — save current frame to results folder
    - `v` — toggle video recording (saved into results folder)
    - `d` — toggle printing bounding box coordinates to console
    - `c` — change confidence threshold interactively (entered into console)
  - Prints for each detection:
    - bounding box [x,y,w,h], center (cx,cy), and confidence value.

- weed-detector/results/...
  - Contains training outputs and saved recordings/frames created by scripts.

---

## Requirements
- OS: Windows (documented for Windows commands). Works on Linux/macOS as well with camera device path and codec differences.
- Python: 3.8+ recommended
- Hardware: CPU works; GPU recommended for real-time performance (with compatible PyTorch + CUDA)
- Python packages:
  - ultralytics (YOLO)
  - opencv-python (or opencv-python-headless if you don't need display windows)
  - torch (CPU or GPU build)
  - numpy
  - pathlib (stdlib)
  - time (stdlib)

Example packages:
- ultralytics
- opencv-python
- numpy
- torch

Make a `requirements.txt` with these entries (example):
```
ultralytics
opencv-python
numpy
torch
```

---

## Virtual environment (recommended)
Use a virtual environment to isolate dependencies per project and avoid polluting system Python.

Windows (CMD):
- python -m venv .venv
- .venv\Scripts\activate
- pip install --upgrade pip
- pip install -r requirements.txt

Windows (PowerShell):
- python -m venv .venv
- . .venv\Scripts\Activate.ps1
  - If activation fails, run PowerShell as administrator or allow script execution:
    - Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
- pip install --upgrade pip
- pip install -r requirements.txt

macOS / Linux:
- python3 -m venv .venv
- source .venv/bin/activate
- pip install --upgrade pip
- pip install -r requirements.txt

Conda (alternative):
- conda create -n weed python=3.10
- conda activate weed
- pip install -r requirements.txt (or use conda packages where appropriate)

VS Code integration:
- Open the project in VS Code.
- Select the interpreter: Ctrl+Shift+P → "Python: Select Interpreter" → choose the `.venv` interpreter.
- Optional: add `.vscode/settings.json` (workspace) with:
  ```
  {
    "python.pythonPath": ".venv\\Scripts\\python.exe"
  }
  ```

Reproducible environment:
- After installing dependencies, export the lock:
  - pip freeze > requirements.txt
- Consider using poetry or pip-tools for better reproducibility.

---

## Setup (Windows - recommended steps)
1. Clone repo and open project folder:
   - cd c:\Users\Chivukula\Projects\weed

2. Create and activate a virtual environment (see Virtual environment section above).

3. Install dependencies:
   - pip install --upgrade pip
   - pip install -r requirements.txt
   - If requirements.txt is missing:
     - pip install ultralytics opencv-python numpy
     - Install PyTorch separately (choose correct build for CPU/GPU):
       - CPU example:
         - pip install torch --index-url https://download.pytorch.org/whl/cpu
       - GPU (CUDA) — visit https://pytorch.org/get-started/locally/ and run recommended command for your CUDA version.

4. Confirm packages:
   - python -c "import torch, ultralytics, cv2, numpy; print('OK')"

5. Ensure model weights are present:
   - Default script expects: `weed-detector/results/train/weights/best.pt`
   - If you trained a model elsewhere, either:
     - copy your weights to that path, or
     - edit scripts/webcam_detection.py and set `model_path` or modify to accept CLI args.

---

## Running the code

Open an activated terminal in project root.

1) Run detection tests on sample images (quick validation)
   - python scripts/webcam_detection.py
   - The script will run test_weed_detection_with_images() first (it will open and display up to 3 images if present), then start the real-time webcam loop.

2) Real-time webcam detection
   - By default, it uses camera index `0` (internal webcam). For external camera, change camera_id to `1` or the correct ID in the script `setup_webcam` call (or modify code to accept CLI arg).
   - Run:
     - python scripts/webcam_detection.py
   - The GUI window shows detection results and overlays.

3) Keyboard commands while running:
   - q — Exit cleanly
   - s — Save the current frame as: weed-detector/results/weed_capture_<timestamp>.jpg
   - v — Toggle start/stop video recording. Video saved in `weed-detector/results`:
     - Uses MP4 (mp4v) -> fallback AVI (XVID) if mp4 fails.
     - Filename: weed_recording_<timestamp>.mp4 or .avi
   - d — Toggle coordinate printing on/off in console
   - c — Change confidence threshold (prompt in terminal); valid range 0.1–0.9

4) Where saved outputs live:
   - All saved frames and recordings are placed into: `weed-detector/results`

---

## Video saving & codec tips
- If saved video is not generated or video file size is zero:
  - Make sure `weed-detector/results` has write permissions.
  - Ensure the OpenCV VideoWriter can open a writer for the codec:
    - Windows commonly supports `mp4v` (for .mp4) and `XVID` (for .avi).
    - If `mp4v` fails, the script tries `XVID`. If both fail, check installed codecs or install FFmpeg (and ensure OpenCV built with FFmpeg support).
  - If the display size differs from writer size, the script resizes frames before writing; this reduces mismatches.
  - Confirm frames are being written and try saving a single frame with cv2.imwrite from a Python REPL to confirm writing works.

---

## GPU vs CPU usage
- For best real-time performance, use a PyTorch build with CUDA that matches your GPU and CUDA driver.
- Check GPU availability:
  - python -c "import torch; print(torch.cuda.is_available())"
- If GPU is available, Ultralytics/YOLO will typically use the GPU automatically. If not, it runs on CPU (slower).
- Installing GPU-enabled torch:
  - Visit https://pytorch.org/get-started/locally/ and use the recommended command with your CUDA version.

