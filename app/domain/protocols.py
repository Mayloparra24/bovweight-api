from typing import Protocol

import numpy as np

from .schemas import DepthResult, SegmentationResult


class SegmenterProtocol(Protocol):
    """Adaptador del modelo de segmentación (por ejemplo, YOLOv8-seg)."""

    def detect(self, image: np.ndarray) -> SegmentationResult: ...


class DepthEstimatorProtocol(Protocol):
    """Adaptador del modelo de profundidad (por ejemplo, Depth Pro)."""

    def estimate(
        self,
        image_rgb: np.ndarray,
        area_pixels: int,
        mask: np.ndarray,
    ) -> DepthResult: ...


class EstimationStrategy(Protocol):
    """Estrategia intercambiable de estimación de peso."""

    def estimate(
        self,
        image: np.ndarray,
        segmentation_result: SegmentationResult,
        breed_constant: float,
    ) -> dict: ...
