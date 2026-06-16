import time

import numpy as np
import torch

from app.application.strategies import YOLODepthProStrategy
from app.application.weight_estimator import WeightEstimatorService
from app.infrastructure.depth_pro_estimator import DepthProEstimator
from app.infrastructure.handlers import ValidationChainFactory, ValidationHandler
from app.infrastructure.yolo_segmenter import YOLOv8Segmenter

_yolo_segmenter = None
_depth_estimator = None
_strategy = None
_service = None
_chain = None


def initialize_validation_chain() -> ValidationHandler:
    """Carga modelos y construye la cadena de validación al arrancar."""
    global _yolo_segmenter, _depth_estimator, _strategy, _service, _chain

    if _chain is None:
        _yolo_segmenter = YOLOv8Segmenter()
        _depth_estimator = DepthProEstimator()
        _strategy = YOLODepthProStrategy(depth_estimator=_depth_estimator)
        _service = WeightEstimatorService(strategy=_strategy)
        _chain = ValidationChainFactory.build(
            segmenter=_yolo_segmenter,
            service=_service,
        )

        _warmup_models(_yolo_segmenter, _depth_estimator)

    return _chain


def _warmup_models(
    segmenter: YOLOv8Segmenter,
    depth_estimator: DepthProEstimator,
) -> None:
    t0 = time.perf_counter()

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        segmenter.model(dummy, verbose=False)
        t1 = time.perf_counter()
        print(f"[warmup] YOLO: {t1 - t0:.2f}s")

        tensor = depth_estimator.transform(dummy).to(depth_estimator.device).half()
        t2 = time.perf_counter()
        print(f"[warmup] Depth Pro transform: {t2 - t1:.2f}s")

        with torch.no_grad():
            depth_estimator.model.infer(tensor, f_px=None)
        t3 = time.perf_counter()
        print(f"[warmup] Depth Pro infer: {t3 - t2:.2f}s")
    except Exception as e:
        print(f"[warmup] Error durante warm-up (no crítico): {e}")

    print(f"[warmup] Total: {time.perf_counter() - t0:.2f}s")


def get_validation_chain() -> ValidationHandler:
    if _chain is None:
        initialize_validation_chain()
    return _chain
