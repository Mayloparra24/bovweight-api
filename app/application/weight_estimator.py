import cv2
import numpy as np

from app.domain.protocols import DepthEstimatorProtocol, SegmenterProtocol


class WeightEstimatorService:
    def __init__(
        self,
        segmenter: SegmenterProtocol,
        depth_estimator: DepthEstimatorProtocol,
    ):
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator

    def predict(self, image: np.ndarray, breed_constant: float) -> dict:
        seg_result = self.segmenter.detect(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_result = self.depth_estimator.estimate(
            image_rgb,
            seg_result.area_pixels,
            seg_result.mask,
        )

        estimated_weight_kg = breed_constant * (depth_result.real_area_m2 ** 1.5)

        return {
            "success": True,
            "data": {
                "peso_estimado_kg": round(estimated_weight_kg, 2),
                "confianza_yolo": round(seg_result.confidence, 4),
            },
        }
