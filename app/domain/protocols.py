from typing import Protocol

import numpy as np

from .schemas import DepthResult, SegmentationResult


class SegmenterProtocol(Protocol):
    def detect(self, image: np.ndarray) -> SegmentationResult: ...


class DepthEstimatorProtocol(Protocol):
    def estimate(
        self,
        image_rgb: np.ndarray,
        area_pixels: int,
        mask: np.ndarray,
    ) -> DepthResult: ...
