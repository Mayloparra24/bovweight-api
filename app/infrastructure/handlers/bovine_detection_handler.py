from typing import Optional

from app.domain.protocols import SegmenterProtocol

from .base import ValidationContext, ValidationHandler


class BovineDetectionHandler(ValidationHandler):
    """Confirma que existe al menos un bovino claramente visible."""

    def __init__(self, segmenter: SegmenterProtocol):
        super().__init__()
        self.segmenter = segmenter

    def handle(self, context: ValidationContext) -> Optional[dict]:
        context.segmentation_result = self.segmenter.detect(context.image)
        return self._call_next(context)
