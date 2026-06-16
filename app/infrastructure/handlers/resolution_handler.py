from typing import Optional

from .base import ValidationContext, ValidationHandler


class ResolutionHandler(ValidationHandler):
    """Valida que la imagen tenga resolución mínima suficiente."""

    MIN_WIDTH = 256
    MIN_HEIGHT = 256

    def handle(self, context: ValidationContext) -> Optional[dict]:
        height, width = context.image.shape[:2]

        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            raise ValueError(
                f"La imagen es demasiado pequeña: {width}x{height}. "
                f"Mínimo requerido: {self.MIN_WIDTH}x{self.MIN_HEIGHT}."
            )

        return self._call_next(context)
