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


def get_validation_chain() -> ValidationHandler:
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

    return _chain
