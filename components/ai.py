"""
AI / Plant Detection Component
===============================
YOLO-based object detection for identifying plants (or any
trained class) from camera frames.

The model file is loaded from the ``ai_models/`` directory by default.
If the model file is missing or fails to load, the detector
gracefully disables itself so the rest of the system keeps running.
"""

import logging
from pathlib import Path

import cv2
import os
from dotenv import load_dotenv
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "ai_models" / "yolo26n.onnx"

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

_TARGET_CLASS = os.getenv("TARGET_CLASS", "plant")
_USE_YOLO_WITH_TRACK = os.getenv("USE_YOLO_WITH_TRACK", "True").lower() == "true"
# We lower the default confidence here since we rely on tracking to filter noise
_DEFAULT_CONFIDENCE = 0.2


class PlantDetector:
    """YOLO-powered plant / object detector."""

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        confidence: float = _DEFAULT_CONFIDENCE,
    ):
        self.confidence = confidence
        self.enabled = False
        
        self.target_class_str = _TARGET_CLASS
        self.use_track = _USE_YOLO_WITH_TRACK
        self.target_class_ids = None

        logger.info("Loading YOLO model from %s …", model_path)
        try:
            self.model = YOLO(str(model_path))
            self.enabled = True
            logger.info("YOLO model loaded successfully")
            
            # Resolve target class ID
            if self.target_class_str:
                for cls_id, cls_name in self.model.names.items():
                    if cls_name == self.target_class_str:
                        self.target_class_ids = [cls_id]
                        logger.info("Target class '%s' found with ID: %d", self.target_class_str, cls_id)
                        break
                if self.target_class_ids is None:
                    logger.warning("Target class '%s' not found in model.", self.target_class_str)
                    
        except Exception as e:
            logger.warning("Could not load YOLO model — AI disabled: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame) -> list[dict]:
        """
        Run detection on a raw OpenCV frame.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``class``       : detected class name (str)
            - ``confidence``  : detection confidence 0–1 (float)
            - ``box``         : [x1, y1, x2, y2] bounding box (list[float])
        """
        if not self.enabled or frame is None:
            return []

        if self.use_track:
            results = self.model.track(
                frame, imgsz=416, tracker="bytetrack.yaml", persist=True,
                verbose=False, conf=self.confidence, classes=self.target_class_ids
            )
        else:
            results = self.model(
                frame, imgsz=416, verbose=False, conf=self.confidence, classes=self.target_class_ids
            )
            
        result = results[0]

        detections = []
        for box in result.boxes:
            det = {
                "class": self.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": box.xyxy[0].tolist(),
                "id": int(box.id[0]) if box.id is not None else None
            }
            detections.append(det)

        logger.debug("Detected %d object(s)", len(detections))
        return detections

    def annotate_frame(self, frame, detections: list[dict]):
        """
        Draw bounding boxes and labels onto a frame copy.

        Returns the annotated frame (original is not modified).
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            id_str = f' [ID:{det["id"]}]' if det.get("id") is not None else ''
            label = f'{det["class"]}{id_str} {det["confidence"]:.0%}'

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        return annotated

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Release model resources."""
        self.enabled = False
        self.model = None
        logger.info("PlantDetector closed")


# ------------------------------------------------------------------
# Quick self-test (python -m components.ai)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)
    detector = PlantDetector()

    if not detector.enabled:
        print("✘ Model not loaded — place yolo11n.pt in ai_models/")
        exit(1)

    # Test with camera if available
    try:
        from components.camera import RobotCamera

        camera = RobotCamera()
        print("Waiting for frame…")
        time.sleep(2)

        frame = camera.get_raw_frame()
        if frame is not None:
            detections = detector.detect(frame)
            print(f"✔ {len(detections)} detection(s):")
            for d in detections:
                print(f"   {d['class']}  {d['confidence']:.0%}  {d['box']}")
        else:
            print("✘ No frame from camera")

        camera.close()
    except Exception as e:
        print(f"Camera not available, skipping live test: {e}")

    detector.close()
