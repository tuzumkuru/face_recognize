import argparse
import os
import cv2
import yaml
import face_recognition
import numpy as np

from face_db import FaceDB
from notifier import notify_match, notify_unknown
from recognizer import FaceRecognizer


def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {
        "camera_index": 0,
        "tolerance": 0.6,
        "encoding_file": "encodings.pkl",
        "faces_dir": "faces",
        "process_every_n_frames": 2,
        # detection options: 'hog' (fast) or 'cnn' (more accurate, slower)
        "detection_model": "hog",
        # how many times to upsample the image when looking for faces (0..2+). Increasing
        # this helps detect small/far faces at the cost of CPU.
        "upsample_times": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", nargs=2, metavar=("NAME", "IMAGE"), help="Add a named face from IMAGE")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    faces_dir = cfg.get("faces_dir", "faces")
    encoding_file = cfg.get("encoding_file", "encodings.pkl")
    db = FaceDB(faces_dir=faces_dir, encoding_file=encoding_file)

    # Ensure faces in folder have encodings at startup
    added = db.ensure_encodings_from_folder()
    if added:
        print(f"Added {added} encodings from `{faces_dir}`")

    if args.add:
        name, image = args.add
        ok = db.add_face_from_image(name, image)
        if ok:
            print(f"Added face for '{name}' from {image}")
        else:
            print(f"No face found in {image}")
        return

    # Start webcam watcher (faster loop: scaled frames + every-other-frame)
    camera_index = cfg.get("camera_index", 0)
    tolerance = cfg.get("tolerance", 0.6)
    process_every_n = cfg.get("process_every_n_frames", 2)
    scale = cfg.get("scale", 0.25)
    detection_model = cfg.get("detection_model", "hog")
    upsample_times = int(cfg.get("upsample_times", 0))

    # recognizer encapsulates dlib/model usage and matching
    recognizer = FaceRecognizer(tolerance=tolerance)

    known_names, known_encodings = db.get_all_encodings()
    print(f"Loaded {len(known_encodings)} known face encodings for {len(set(known_names))} people")

    video = cv2.VideoCapture(camera_index)
    if not video.isOpened():
        print(f"Unable to open camera index {camera_index}")
        return

    process_frame = True
    window_name = 'FaceRec'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        # persistent previous detections so boxes remain on skipped frames
        face_locations = []
        face_names = []
        while True:
            ret, frame = video.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            display_frame = frame.copy()

            if process_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                rgb_small = small_frame[:, :, ::-1]

                # detect faces first
                # pass model and upsample options from config; increasing `upsample_times`
                # helps detect smaller/farther faces. Using `model='cnn'` may improve
                # detection quality but is slower.
                face_locations = face_recognition.face_locations(
                    rgb_small, number_of_times_to_upsample=upsample_times, model=detection_model
                )

                # compute encodings using the recognizer (dlib chip-based path internally)
                face_encodings = recognizer.compute_encodings(rgb_small, face_locations)

                face_names = []
                for encoding in face_encodings:
                    name = "Not recognized"
                    if encoding is None:
                        face_names.append(name)
                        notify_unknown(1.0)
                        continue

                    name, distance = recognizer.match(encoding, known_names, known_encodings)
                    if name != "Not recognized":
                        notify_match(name, distance)
                    else:
                        notify_unknown(distance)
                    face_names.append(name)

            process_frame = not process_frame

            # draw boxes
            if 'face_locations' in locals() and 'face_names' in locals():
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # scale coords back up
                    top = int(top / scale)
                    right = int(right / scale)
                    bottom = int(bottom / scale)
                    left = int(left / scale)

                    color = (0, 255, 0) if name != "Not recognized" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display_frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    label = name
                    cv2.putText(display_frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

            # if window closed by user, break BEFORE calling imshow to avoid
            # recreating the window unintentionally (imshow recreates closed windows)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
