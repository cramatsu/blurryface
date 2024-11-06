from pathlib import Path
from typing import List

from ultralytics import YOLO

model_path = Path("app/models/best.onnx")


class YOLOModel:
    def __init__(self):
        self._model = YOLO(model_path.absolute(), task="detect", verbose=False)
        pass

    def detect_faces(self, image_path: str, conf: float, iou: float) -> List[tuple[int, ...]] | None:
        results = self._model.predict(image_path, conf=conf, iou=iou, verbose=False)

        has_faces = any(result.boxes for result in results)

        if not has_faces:
            return None

        face_boxes = [tuple(map(int, box.xyxy[0])) for result in results for box in result.boxes]

        return face_boxes
