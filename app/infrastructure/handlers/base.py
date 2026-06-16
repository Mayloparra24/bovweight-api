from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.domain.schemas import SegmentationResult


@dataclass
class ValidationContext:
    """Contexto que viaja a través de la cadena de responsabilidad."""

    image: np.ndarray
    breed_constant: float
    content_type: Optional[str] = None
    segmentation_result: Optional[SegmentationResult] = None
    result: Optional[dict] = None


class ValidationHandler(ABC):
    """Eslabón base de la cadena de validación."""

    def __init__(self):
        self._next: Optional["ValidationHandler"] = None

    def set_next(self, handler: "ValidationHandler") -> "ValidationHandler":
        self._next = handler
        return handler

    @abstractmethod
    def handle(self, context: ValidationContext) -> Optional[dict]:
        ...

    def _call_next(self, context: ValidationContext) -> Optional[dict]:
        if self._next is None:
            return None
        return self._next.handle(context)
