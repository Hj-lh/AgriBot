"""
Plant Re-Identification Component
==================================
Appearance-based re-ID for the watered-plant memory.

ByteTrack IDs are only stable while a plant stays roughly in view. Once the
robot reverses, scans, or rotates 180°, a previously-watered plant that comes
back into frame will almost certainly receive a *new* track ID — so an
ID-only memory will happily re-water it.

This module crops each candidate plant out of the camera frame, runs it
through a tiny MobileNet-V3 backbone (already installed as a torchvision
transitive dependency of ultralytics), and yields a 1024-D unit vector.
Cosine similarity against the embeddings stored at watering time gives a
viewpoint-tolerant "is this the same physical plant?" check.

The component disables itself silently if torch / torchvision aren't
available, so the rest of the bot keeps running with ID-only filtering.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.82   # cosine similarity above this ⇒ "same plant"
_MIN_CROP_SIDE     = 8      # ignore tiny crops; embeddings would be junk


class PlantReID:
    """Appearance-based watered-plant memory using a small CNN backbone."""

    def __init__(self, similarity_threshold: float = _DEFAULT_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self._watered: list[np.ndarray] = []
        self.enabled = False

        try:
            import torch
            from torchvision import models as tvm

            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT
            backbone = tvm.mobilenet_v3_small(weights=weights)
            # Drop the classifier head — we only want the global-pooled features
            backbone.classifier = torch.nn.Identity()
            backbone.eval()

            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = backbone.to(self.device)
            self.preprocess = weights.transforms()
            self.enabled = True
            logger.info("PlantReID initialised on %s (threshold=%.2f)",
                        self.device, similarity_threshold)
        except Exception as e:  # noqa: BLE001
            logger.warning("PlantReID disabled (%s) — falling back to ID-only memory", e)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, frame, box) -> np.ndarray | None:
        """Return a unit-norm embedding for ``frame[box]``, or None."""
        if not self.enabled or frame is None or box is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if (x2 - x1) < _MIN_CROP_SIDE or (y2 - y1) < _MIN_CROP_SIDE:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        try:
            from PIL import Image
            pil = Image.fromarray(crop_rgb)
            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                feat = self.model(tensor).flatten().cpu().numpy()
        except Exception as e:  # noqa: BLE001
            logger.debug("ReID embed failed: %s", e)
            return None

        norm = float(np.linalg.norm(feat))
        if norm < 1e-6:
            return None
        return feat / norm

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def add(self, embedding: np.ndarray | None):
        """Remember ``embedding`` as belonging to a watered plant."""
        if embedding is None:
            return
        self._watered.append(embedding)
        logger.info("PlantReID: watered embedding stored (total=%d)", len(self._watered))

    def is_watered(self, embedding: np.ndarray | None) -> bool:
        """True if ``embedding`` matches any stored watered embedding."""
        if embedding is None or not self._watered:
            return False
        sims = [float(np.dot(embedding, w)) for w in self._watered]
        best = max(sims)
        if best >= self.similarity_threshold:
            logger.debug("PlantReID: match (sim=%.2f)", best)
            return True
        return False

    def best_similarity(self, embedding: np.ndarray | None) -> float:
        """Return the highest cosine similarity against stored embeddings."""
        if embedding is None or not self._watered:
            return 0.0
        return max(float(np.dot(embedding, w)) for w in self._watered)

    def clear(self):
        self._watered.clear()

    @property
    def count(self) -> int:
        return len(self._watered)
