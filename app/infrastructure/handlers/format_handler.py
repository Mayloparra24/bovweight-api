from typing import Optional

from .base import ValidationContext, ValidationHandler


class FormatHandler(ValidationHandler):
    """Valida que el archivo sea una imagen en formato soportado."""

    ALLOWED_TYPES = {
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp",
        "image/tiff",
    }

    def handle(self, context: ValidationContext) -> Optional[dict]:
        content_type = context.content_type
        if content_type is None:
            raise ValueError("No se especificó el tipo de contenido de la imagen.")

        clean_type = content_type.split(";")[0].strip()
        if clean_type not in self.ALLOWED_TYPES:
            raise ValueError(
                f"Tipo de archivo no permitido: {content_type}. Use JPEG o PNG."
            )

        return self._call_next(context)
