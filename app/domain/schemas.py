from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    area_pixels: int
    confidence: float


@dataclass(frozen=True)
class DepthResult:
    avg_distance_m: float
    focal_length_px: float
    real_area_m2: float
