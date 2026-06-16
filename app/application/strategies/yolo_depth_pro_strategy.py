import cv2
import numpy as np

from app.domain.protocols import DepthEstimatorProtocol, EstimationStrategy
from app.domain.schemas import SegmentationResult


class YOLODepthProStrategy:
    """Estrategia concreta de estimación usando Depth Pro sobre la máscara de YOLO."""

    def __init__(self, depth_estimator: DepthEstimatorProtocol):
        self.depth_estimator = depth_estimator

    def estimate(
        self,
        image: np.ndarray,
        segmentation_result: SegmentationResult,
        breed_constant: float,
    ) -> dict:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_result = self.depth_estimator.estimate(
            image_rgb,
            segmentation_result.area_pixels,
            segmentation_result.mask,
        )

        estimated_weight_kg = breed_constant * (depth_result.real_area_m2 ** 1.5)

        return {
            "success": True,
            "data": {
                "peso_estimado_kg": round(estimated_weight_kg, 2),
                "confianza_yolo": round(segmentation_result.confidence, 4),
            },
        }
