import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import BEST_PT_PATH, DEVICE
from app.domain.protocols import SegmenterProtocol
from app.domain.schemas import SegmentationResult


class YOLOv8Segmenter:
    """Adaptador del modelo YOLOv8-seg al SegmenterProtocol del dominio."""

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

        return SegmentationResult(
            mask=mask_binary,
            area_pixels=area_pixels,
            confidence=confidence,
        )
