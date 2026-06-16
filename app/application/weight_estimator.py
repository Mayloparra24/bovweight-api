import numpy as np

from app.domain.protocols import EstimationStrategy
from app.domain.schemas import SegmentationResult


class WeightEstimatorService:
    """Contexto del patrón Strategy para estimación de peso."""

    def __init__(self, strategy: EstimationStrategy):
        self.strategy = strategy

    def predict(
        self,
        image: np.ndarray,
        segmentation_result: SegmentationResult,
        breed_constant: float,
    ) -> dict:
        return self.strategy.estimate(image, segmentation_result, breed_constant)
