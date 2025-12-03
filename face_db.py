import os
import pickle
from typing import Dict, List

import numpy as np
import face_recognition


class FaceDB:
    def __init__(self, faces_dir: str = "faces", encoding_file: str = "encodings.pkl"):
        self.faces_dir = faces_dir
        self.encoding_file = encoding_file
        os.makedirs(self.faces_dir, exist_ok=True)
        # mapping: stem -> list of encodings (we store one encoding per .npy file)
        self.encodings: Dict[str, List] = {}
        self._load()

    def _load(self):
        # Load per-file encodings stored as .npy files in faces_dir
        self.encodings = {}
        for fn in os.listdir(self.faces_dir):
            path = os.path.join(self.faces_dir, fn)
            if not os.path.isfile(path):
                continue
            name, ext = os.path.splitext(fn)
            if ext.lower() == ".npy":
                try:
                    arr = np.load(path)
                    # keep it as a list to allow multiple encodings per name in future
                    self.encodings.setdefault(name, []).append(arr)
                except Exception:
                    continue

        # backward-compat: if no .npy found but an encoding_file exists, load it
        if not self.encodings and os.path.exists(self.encoding_file):
            try:
                with open(self.encoding_file, "rb") as f:
                    self.encodings = pickle.load(f)
            except Exception:
                self.encodings = {}

    def _save_pickle(self):
        with open(self.encoding_file, "wb") as f:
            pickle.dump(self.encodings, f)

    def _save_npy(self, stem: str, encoding: np.ndarray) -> str:
        """Save a single encoding array to faces_dir as <stem>.npy. If file exists,
        pick a non-colliding name by appending an index.

        Returns the path of the saved .npy file.
        """
        base_path = os.path.join(self.faces_dir, f"{stem}.npy")
        if not os.path.exists(base_path):
            np.save(base_path, encoding)
            return base_path
        # find next available
        idx = 1
        while True:
            candidate = os.path.join(self.faces_dir, f"{stem}_{idx}.npy")
            if not os.path.exists(candidate):
                np.save(candidate, encoding)
                return candidate
            idx += 1

    def add_face_from_image(self, name: str, image_path: str) -> bool:
        """Compute encoding from an image file and save the encoding as a .npy
        file inside `faces_dir` using `name` as the stem. Does NOT write any
        image files.

        Returns True if added, False if no face found.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(image)
        if not encs:
            return False

        encoding = np.array(encs[0])

        # determine stem and save .npy in faces_dir
        stem = name
        saved_path = self._save_npy(stem, encoding)

        # update in-memory mapping
        self.encodings.setdefault(stem, []).append(encoding)
        # also update legacy pickle file for compatibility
        try:
            self._save_pickle()
        except Exception:
            pass

        return True

    def add_face_file_by_filename(self, file_path: str) -> bool:
        """Add a face encoding using the file's stem as the name if encoding missing.

        This will create a .npy alongside the image file (e.g. 'alice.jpg' -> 'alice.npy').
        Returns True if an encoding was added, False otherwise.
        """
        if not os.path.exists(file_path):
            return False
        stem = os.path.splitext(os.path.basename(file_path))[0]
        npy_path = os.path.join(self.faces_dir, f"{stem}.npy")
        # if corresponding .npy already exists, skip
        if os.path.exists(npy_path):
            return False
        return self.add_face_from_image(stem, file_path)

    def ensure_encodings_from_folder(self):
        """Scan `faces_dir` for image files and create .npy encodings for files
        that don't have a corresponding .npy file. Returns number of encodings added.
        """
        added = 0
        for fn in os.listdir(self.faces_dir):
            path = os.path.join(self.faces_dir, fn)
            if not os.path.isfile(path):
                continue
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            npy_path = os.path.join(self.faces_dir, f"{stem}.npy")
            if os.path.exists(npy_path):
                continue
            try:
                ok = self.add_face_from_image(stem, path)
                if ok:
                    added += 1
            except Exception:
                continue
        return added

    def get_all_encodings(self):
        names = []
        encs = []
        for name, arr in self.encodings.items():
            for e in arr:
                names.append(name)
                encs.append(e)
        return names, encs

    def list_known(self):
        return list(self.encodings.keys())
