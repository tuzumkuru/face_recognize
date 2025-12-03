# Face Recognize (prototype)

Simple webcam face-recognition prototype.

## Overview

**What it does**
- **Face capture**: reads frames from your webcam using OpenCV (`cv2`).
- **Detection**: finds faces in each frame using `face_recognition` (dlib-backed detector).
- **Embedding**: computes 128-d face descriptors using dlib models (chip-based alignment for robustness).
- **Matching**: compares descriptors against known encodings stored on disk and prints match/unknown events to the console.

**Files of interest**
- `app.py`: main entrypoint — CLI to add faces and the webcam watcher.
- `recognizer.py`: `FaceRecognizer` class — loads models, computes encodings and performs matching.
- `face_db.py`: manages on-disk encodings. Encodings are stored as individual `.npy` files in the `faces/` folder.
- `notifier.py`: stub notifier that prints matches to console (easy to extend).
- `config.yaml`: runtime configuration and detection tuning.

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Add known faces (creates a `.npy` encoding in `faces/`):

```powershell
python app.py --add "Alice" path\to\alice.jpg
python app.py --add "Bob" path\to\bob.jpg
```

3. Run the webcam watcher:

```powershell
python app.py
```

### Windows build tools (important)

On Windows, building the `dlib` dependency (used by `face_recognition`) often requires native build tools. Before running `pip install -r requirements.txt`, please install:

- **CMake**: available from https://cmake.org/download/
- **Microsoft Visual Studio Build Tools**: install the "Desktop development with C++" workload or the standalone Build Tools package.

Having these installed ensures `dlib` can be built if a compatible prebuilt wheel isn't available for your Python version.

If you prefer to avoid a local build, look for a prebuilt `dlib` wheel matching your Python version and architecture.

## Behavior and on-disk format
- Encodings: each face encoding is saved as `<stem>.npy` in the `faces/` directory (e.g. `alice.npy`). The app will scan `faces/` on startup and create missing `.npy` files for any images found there.
- No image files are written or copied by the app — image files you add remain unchanged.
- The app draws bounding boxes and labels on the live camera feed and prints timestamped match/unknown events to the console.

Configuration and tuning
- `config.yaml` contains runtime options. Important tuning knobs:
	- `detection_model`: `hog` (fast) or `cnn` (more accurate, slower; requires a dlib build supporting CNN)
	- `upsample_times`: integer >= 0. Upsampling helps detect smaller/far faces (try `1` or `2`), CPU cost increases with higher values.
	- `scale`: frame downscale factor used before detection (0.25 = faster, 0.5 = more detail). Increase toward `1.0` to help detect far faces.
	- `tolerance`: matching distance threshold (default `0.6`).

Examples (in `config.yaml`):
- Far faces (slower, more sensitive):
	- `detection_model: hog`
	- `upsample_times: 2`
	- `scale: 0.5`
- Balanced (default):
	- `detection_model: hog`
	- `upsample_times: 1`
	- `scale: 0.5`
- High-accuracy (very slow, dlib CNN):
	- `detection_model: cnn`
	- `upsample_times: 0`
	- `scale: 0.5`

## Troubleshooting
- If faces are only detected when very close, increase `upsample_times` and `scale` to give the detector more image detail (at the cost of CPU).
- If detection is very slow, reduce `upsample_times` and lower `scale` (e.g. `0.25`).
- If you encounter errors installing `dlib`, check for a prebuilt wheel for your Python/arch; otherwise ensure CMake and Visual Studio Build Tools are available (see Installation).

- Extending the project
- Replace `notifier.py` with a webhook/MQTT/GUI to take actions on matches.
- Swap the recognition backend by replacing `recognizer.py` (e.g., OpenCV DNN or a small neural network for embedded devices).

## Thanks

This project uses the open-source `face_recognition` library (by Adam Geitgey and contributors). Thank you to the authors and community for making these tools available.

## License / Notes

This is a prototype for experimentation and learning — not production-ready. Use carefully with privacy considerations in mind.
