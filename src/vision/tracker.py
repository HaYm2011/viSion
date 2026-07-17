import logging
import cv2
import json
import os
from typing import List, Dict, Any, Tuple
from src.db.database import log_event

logger = logging.getLogger(__name__)

class VisionTracker:
    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self._load_config(config_path)
        self.zones = self.config.get("zones", [])
        self.objects_to_track = self.config.get("track_objects", ["keys", "wallet", "phone"])

        # Dictionary to store the last known zone for each object
        # Format: {"keys": "counter", "wallet": "desk"}
        self.last_known_states: Dict[str, str] = {}

        # We would initialize YOLO-World here, but for this skeleton we just mock it
        # from ultralytics import YOLO
        # self.model = YOLO("yolov8s-world.pt")
        # self.model.set_classes(self.objects_to_track)
        logger.info(f"Initialized VisionTracker tracking {len(self.objects_to_track)} objects.")

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {path}, using defaults. Error: {e}")
            return {"zones": [], "track_objects": ["keys", "wallet", "phone"]}

    def process_frame(self, frame, save_dir: str = "data/crops") -> List[Dict[str, Any]]:
        """
        Process a single frame. Detect objects, check against zones, and log transitions.
        Returns a list of detected objects and their current zones.
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Detect objects (Mocked for now)
        detections = self._detect_objects(frame)

        current_states = {}

        # 2. Map detections to zones
        for det in detections:
            obj_name = det["name"]
            bbox = det["bbox"]
            confidence = det["confidence"]

            zone_id = self._get_zone_for_bbox(bbox)

            if zone_id:
                current_states[obj_name] = {
                    "zone_id": zone_id,
                    "confidence": confidence,
                    "bbox": bbox
                }

        # 3. Check for state transitions and log events
        self._check_state_transitions(current_states, frame, save_dir)

        return list(current_states.values())

    def _detect_objects(self, frame) -> List[Dict[str, Any]]:
        """
        Wrapper around the YOLO-World detection logic.
        MOCK IMPLEMENTATION for skeleton.
        Returns list of dicts: {"name": "keys", "bbox": [x1, y1, x2, y2], "confidence": 0.95}
        """
        # In a real app:
        # results = self.model(frame)
        # detections = []
        # for r in results:
        #     for box in r.boxes:
        #         name = self.model.names[int(box.cls)]
        #         detections.append({"name": name, "bbox": box.xyxy[0].tolist(), "confidence": float(box.conf)})
        # return detections

        # Mock detection for testing
        return []

    def _get_zone_for_bbox(self, bbox: List[int]) -> str:
        """
        Given a bounding box [x1, y1, x2, y2], determine which zone it falls into.
        Uses center point of the bbox.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        for zone in self.zones:
            zx1, zy1, zx2, zy2 = zone["bbox"]
            if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                return zone["id"]

        return "unknown"

    def _check_state_transitions(self, current_states: Dict[str, Dict[str, Any]], frame, save_dir: str):
        """
        Compare current_states with self.last_known_states.
        Log to database if an object moved, appeared, or disappeared.
        """

        # Check for appeared or moved objects
        for obj_name, state in current_states.items():
            current_zone = state["zone_id"]
            last_zone = self.last_known_states.get(obj_name)

            if current_zone != last_zone:
                # State transition detected!
                event_type = "moved" if last_zone else "appeared"

                # Optionally save a crop
                crop_path = None
                if frame is not None:
                    # Mock crop save
                    # x1, y1, x2, y2 = map(int, state["bbox"])
                    # crop = frame[y1:y2, x1:x2]
                    crop_path = os.path.join(save_dir, f"{obj_name}_{current_zone}.jpg")
                    # cv2.imwrite(crop_path, crop)

                logger.info(f"Transition: {obj_name} {event_type} at {current_zone}")
                log_event(
                    object_name=obj_name,
                    event_type=event_type,
                    zone_id=current_zone,
                    confidence=state["confidence"],
                    frame_path=crop_path
                )

                self.last_known_states[obj_name] = current_zone

        # We don't immediately log "disappeared" because of occlusion or failed detections.
        # A robust system would wait N frames before deciding it's truly gone.
        # For simplicity, we only log positive presence transitions (appeared/moved)
        # and trust the "last known location".
