from app.infrastructure.yolo_segmenter import YOLOv8Segmenter
from app.infrastructure.depth_pro_estimator import DepthProEstimator
from app.application.weight_estimator import WeightEstimatorService

_yolo_segmenter = None
_depth_estimator = None
_weight_estimator = None


def get_weight_estimator() -> WeightEstimatorService:
    global _yolo_segmenter, _depth_estimator, _weight_estimator

    if _weight_estimator is None:
        _yolo_segmenter = YOLOv8Segmenter()
        _depth_estimator = DepthProEstimator()
        _weight_estimator = WeightEstimatorService(
            segmenter=_yolo_segmenter,
            depth_estimator=_depth_estimator,
        )

    return _weight_estimator
