import os
from typing import Any, List, Optional, Sequence, Tuple

import dlib
import face_recognition
import face_recognition_models
import numpy as np


class FaceRecognizer:
    """Encapsulates model loading, encoding computation and matching logic.

    Methods are intentionally small and side-effect free where possible so
    they are easy to test.
    """

    def __init__(self, tolerance: float = 0.6):
        self.tolerance = tolerance

        base = os.path.dirname(face_recognition_models.__file__)
        self.model_path = os.path.join(base, "models", "dlib_face_recognition_resnet_model_v1.dat")
        self.sp_path = os.path.join(base, "models", "shape_predictor_5_face_landmarks.dat")

        # use getattr to avoid static-analyzer issues in editors
        self._face_encoder_ctor = getattr(dlib, "face_recognition_model_v1", None)
        self._shape_predictor_ctor = getattr(dlib, "shape_predictor", None)
        self._rectangle_ctor = getattr(dlib, "rectangle", None)
        self._get_face_chip = getattr(dlib, "get_face_chip", None)

        # instantiate model objects lazily
        self._face_encoder = None
        self._shape_predictor = None

    def _ensure_models(self):
        if self._face_encoder is None:
            if self._face_encoder_ctor is None:
                raise RuntimeError("dlib face recognition model constructor not available")
            self._face_encoder = self._face_encoder_ctor(self.model_path)
        if self._shape_predictor is None:
            if self._shape_predictor_ctor is None:
                raise RuntimeError("dlib shape predictor constructor not available")
            self._shape_predictor = self._shape_predictor_ctor(self.sp_path)

    def compute_encodings(self, rgb_small: np.ndarray, face_locations: Sequence[Tuple[int, int, int, int]]) -> List[Optional[np.ndarray]]:
        """Compute face encodings for the provided (small) RGB image and face_locations.

        Returns a list of numpy arrays or None for faces that couldn't be encoded.
        """
        # try dlib chip-based path first (more stable across face orientations)
        encodings: List[Optional[np.ndarray]] = []
        if face_locations:
            try:
                self._ensure_models()
                rgb_for_dlib = np.ascontiguousarray(rgb_small, dtype=np.uint8)
                for (top, right, bottom, left) in face_locations:
                    h, w = rgb_for_dlib.shape[:2]
                    top = max(0, top)
                    left = max(0, left)
                    bottom = min(h, bottom)
                    right = min(w, right)
                    if bottom <= top or right <= left:
                        encodings.append(None)
                        continue
                    rect = self._rectangle_ctor(int(left), int(top), int(right), int(bottom))
                    try:
                        shape = self._shape_predictor(rgb_for_dlib, rect)
                        face_chip = self._get_face_chip(rgb_for_dlib, shape, size=150)
                        face_chip = np.ascontiguousarray(face_chip, dtype=np.uint8)
                        e = self._face_encoder.compute_face_descriptor(face_chip)
                        encodings.append(np.array(e))
                    except Exception:
                        encodings.append(None)
                # if all None, fall back below
            except Exception:
                encodings = [None] * len(face_locations)

        # fallback to face_recognition wrapper if dlib path didn't produce encodings
        if not encodings or all(e is None for e in encodings):
            try:
                encs = face_recognition.face_encodings(rgb_small, face_locations)
                # face_recognition returns list of enc arrays exactly one per location
                encodings = [np.array(e) for e in encs]
            except Exception:
                encodings = [None] * len(face_locations)

        return encodings

    def match(self, encoding: np.ndarray, known_names: Sequence[str], known_encodings: Sequence[np.ndarray]) -> Tuple[str, float]:
        """Return (name, distance). If no match, returns ("Not recognized", 1.0)."""
        if not known_encodings:
            return "Not recognized", 1.0

        matches = face_recognition.compare_faces(known_encodings, encoding)
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        if matches and matches[best_idx] and best_distance <= self.tolerance:
            return known_names[best_idx], best_distance
        return "Not recognized", best_distance
