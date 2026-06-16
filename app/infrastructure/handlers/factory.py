from app.application.weight_estimator import WeightEstimatorService
from app.domain.protocols import SegmenterProtocol

from .base import ValidationHandler
from .bovine_detection_handler import BovineDetectionHandler
from .format_handler import FormatHandler
from .resolution_handler import ResolutionHandler
from .weight_estimation_handler import WeightEstimationHandler


class ValidationChainFactory:
    """Factory Method que ensambla la cadena de validación en orden."""

    @staticmethod
    def build(
        segmenter: SegmenterProtocol,
        service: WeightEstimatorService,
    ) -> ValidationHandler:
        format_handler = FormatHandler()
        resolution_handler = ResolutionHandler()
        bovine_handler = BovineDetectionHandler(segmenter)
        weight_handler = WeightEstimationHandler(service)

        format_handler.set_next(resolution_handler)
        resolution_handler.set_next(bovine_handler)
        bovine_handler.set_next(weight_handler)

        return format_handler
