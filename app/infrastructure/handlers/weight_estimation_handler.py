from typing import Optional

from app.application.weight_estimator import WeightEstimatorService

from .base import ValidationContext, ValidationHandler


class WeightEstimationHandler(ValidationHandler):
    """Calcula el peso estimado usando la estrategia configurada."""

    def __init__(self, service: WeightEstimatorService):
        super().__init__()
        self.service = service

    def handle(self, context: ValidationContext) -> Optional[dict]:
        if context.segmentation_result is None:
            raise ValueError("No hay resultado de segmentación disponible.")

        context.result = self.service.predict(
            context.image,
            context.segmentation_result,
            context.breed_constant,
        )
        return context.result
