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


## Configuration and tuning
`config.yaml` holds the runtime options you can tweak to trade accuracy vs performance. Below are the most useful knobs, what they do, and recommended starting values.

- `detection_model` (string): which face detector to use. Options:
	- `hog` — CPU-friendly and fast; good for most desktop/server use.
	- `cnn` — usually more accurate on difficult poses/lighting but much slower and requires a dlib build with CNN support (may not be available on all platforms).
	- Recommended start: `hog`.

- `upsample_times` (int, >= 0): how many times to upsample the image when searching for faces. Upsampling makes faces effectively larger so the detector can find smaller/far faces.
	- Effect: +1 upsample ≈ doubles detector work for that axis; try `1` or `2` if distant faces are missed.
	- Recommended start: `1` (increase if you miss far faces).

- `scale` (float, 0.1..1.0): the factor used to downsample frames before detection (e.g. `0.5` keeps half the width/height). Lower values are faster but lose detail; higher values keep more detail and improve distant-face detection.
	- Recommended start: `0.5` (use `0.25` for constrained devices; use `0.6`–`1.0` for best accuracy if CPU allows).

- `tolerance` (float): distance threshold for accepting a match. Smaller values are stricter (fewer false positives); larger values are more permissive.
	- Typical range: `0.4`–`0.7`. Recommended default: `0.6`.

Presets (copy these into `config.yaml` to try quickly):

- Balanced (default):
	- `detection_model: hog`
	- `upsample_times: 1`
	- `scale: 0.5`
	- `tolerance: 0.6`

- Far faces (more sensitive, slower):
	- `detection_model: hog`
	- `upsample_times: 2`
	- `scale: 0.6`
	- `tolerance: 0.6`

- Fast (low CPU):
	- `detection_model: hog`
	- `upsample_times: 0`
	- `scale: 0.25`
	- `tolerance: 0.6`

- Accurate (if you have a CNN-capable dlib build):
	- `detection_model: cnn`
	- `upsample_times: 0`
	- `scale: 0.5`
	- `tolerance: 0.55`

Step-by-step tuning guide

1. Start with the Balanced preset.
2. If the app misses faces that are far from the camera:
	 - increase `scale` (e.g. `0.6`), then test; if still missed, increase `upsample_times` to `1` or `2`.
3. If the app is too slow or you need lower CPU usage:
	 - reduce `scale` (e.g. `0.25`) and set `upsample_times: 0`.
4. If you get false positives (wrong matches):
	 - reduce `tolerance` toward `0.5` (be careful — too low may reject valid matches).
5. When you change `config.yaml`, restart the app to apply the new settings.

If you want, I can add a `config.presets.yaml` file with these presets or apply a preset to your current `config.yaml` now.

## How the system works
This section explains the end-to-end flow and the role of each component.

1. Capture (OpenCV)
	- `app.py` opens the webcam via `cv2.VideoCapture` and reads frames in a loop.
	- Frames are optionally resized (`scale` in `config.yaml`) to trade accuracy for speed.

2. Detection (`face_recognition` / dlib)
	- For each processed frame, the code calls `face_recognition.face_locations(...)` to detect face bounding boxes.
	- The detector can run in two modes: `hog` (faster, CPU-only) or `cnn` (more accurate, may require a dlib build with CNN support).
	- `upsample_times` controls how many times the detector upsamples the image before searching; higher values help find smaller/far faces.

3. Alignment & Embedding (`recognizer.py` / dlib)
	- Detected bounding boxes are passed to the `FaceRecognizer` which attempts to compute aligned face "chips" using a dlib shape predictor.
	- Each face chip is passed through the dlib ResNet-based face recognition model to produce a 128-dimensional embedding (descriptor).
	- The implementation uses a chip-based approach (via `get_face_chip`) which aligns faces to a canonical orientation, improving embedding consistency.

4. Matching
	- Face embeddings from the current frame are compared to known embeddings loaded from `face_db.py` using cosine/EUCLIDEAN-style distance (the code uses `face_recognition.face_distance` and a `tolerance` threshold).
	- If a nearest known embedding is within `tolerance`, the face is considered a match and the known name is reported; otherwise it is reported as "Not recognized".

5. Persistence (`face_db.py`)
	- Known face embeddings are stored on disk as `.npy` files under the `faces/` directory (one encoding file per identity image). This makes inspecting and managing encodings straightforward.
	- At startup the app scans `faces/` and loads all `.npy` encodings into memory. If it finds image files without an accompanying `.npy`, it will compute and save the encoding.

6. Notifier (`notifier.py`)
	- When a match or unknown event occurs the app calls the notifier. The default `notifier.py` prints timestamped messages to the console, but you can replace it with a webhook, MQTT publisher, or other action handler.

7. UI / Display (OpenCV)
	- The app draws bounding boxes and labels on the video frames and shows them in a window using OpenCV.
	- The code includes logic to detect window closure and exit cleanly to avoid re-opening the window after the user closes it.

8. Performance considerations
	- The pipeline downscales frames (controlled by `scale`) and optionally skips processing on alternating frames to maintain responsiveness.
	- For farther/smaller faces: increase `upsample_times` and `scale` (slower but more sensitive).
	- For embedded or CPU-limited environments: keep `scale` small and `upsample_times` at 0, or consider replacing the detector with a lightweight NN specialized for low-power devices.

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
