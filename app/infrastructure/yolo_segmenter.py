import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import BEST_PT_PATH, DEVICE
from app.domain.protocols import SegmenterProtocol
from app.domain.schemas import SegmentationResult


class YOLOv8Segmenter:
    def __init__(
        self,
        model_path: str = BEST_PT_PATH,
        device: torch.device = DEVICE,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device

    def detect(self, image: np.ndarray) -> SegmentationResult:
        result = self.model(image, verbose=False)[0]

        if result.masks is None:
            raise ValueError("No se detecto la silueta del bovino en la imagen.")

        mask_original = result.masks.data[0].cpu().numpy()
        alto_orig, ancho_orig = image.shape[:2]
        mask_resized = cv2.resize(
            mask_original,
            (ancho_orig, alto_orig),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        area_pixels = int(np.sum(mask_binary == 1))

        if result.boxes is not None and len(result.boxes.conf) > 0:
            confidence = float(result.boxes.conf[0].cpu().numpy())
        else:
            confidence = 0.0

        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].cpu().numpy())
        else:
            rows, cols = np.where(mask_binary == 1)
            if len(rows) == 0:
                raise ValueError("No se pudo encontrar la region del bovino.")
            x1, y1 = np.min(cols), np.min(rows)
            x2, y2 = np.max(cols), np.max(rows)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ancho_orig, x2), min(alto_orig, y2)
        crop = image[y1:y2, x1:x2].copy()

        return SegmentationResult(
            mask=mask_binary,
            crop=crop,
            area_pixels=area_pixels,
            confidence=confidence,
        )
