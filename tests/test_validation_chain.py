import unittest
from unittest.mock import MagicMock

import numpy as np

from app.application.weight_estimator import WeightEstimatorService
from app.domain.schemas import SegmentationResult
from app.infrastructure.handlers import (
    BovineDetectionHandler,
    FormatHandler,
    ResolutionHandler,
    ValidationContext,
    ValidationChainFactory,
    WeightEstimationHandler,
)


class TestFormatHandler(unittest.TestCase):
    def test_accepts_jpeg(self):
        handler = FormatHandler()
        context = ValidationContext(
            image=np.zeros((300, 300, 3), dtype=np.uint8),
            breed_constant=1.0,
            content_type="image/jpeg",
        )
        self.assertIsNone(handler.handle(context))

    def test_rejects_plain_text(self):
        handler = FormatHandler()
        context = ValidationContext(
            image=np.zeros((300, 300, 3), dtype=np.uint8),
            breed_constant=1.0,
            content_type="text/plain",
        )
        with self.assertRaises(ValueError):
            handler.handle(context)


class TestResolutionHandler(unittest.TestCase):
    def test_accepts_large_image(self):
        handler = ResolutionHandler()
        context = ValidationContext(
            image=np.zeros((300, 300, 3), dtype=np.uint8),
            breed_constant=1.0,
        )
        self.assertIsNone(handler.handle(context))

    def test_rejects_small_image(self):
        handler = ResolutionHandler()
        context = ValidationContext(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            breed_constant=1.0,
        )
        with self.assertRaises(ValueError):
            handler.handle(context)


class TestBovineDetectionHandler(unittest.TestCase):
    def test_detects_bovine(self):
        segmenter = MagicMock()
        segmenter.detect.return_value = SegmentationResult(
            mask=np.ones((300, 300), dtype=np.uint8),
            area_pixels=100,
            confidence=0.9,
        )

        handler = BovineDetectionHandler(segmenter)
        context = ValidationContext(
            image=np.zeros((300, 300, 3), dtype=np.uint8),
            breed_constant=1.0,
        )
        handler.handle(context)

        segmenter.detect.assert_called_once()
        self.assertIsNotNone(context.segmentation_result)


class TestValidationChainFactory(unittest.TestCase):
    def test_chain_returns_result(self):
        segmenter = MagicMock()
        segmenter.detect.return_value = SegmentationResult(
            mask=np.ones((300, 300), dtype=np.uint8),
            area_pixels=100,
            confidence=0.9,
        )

        strategy = MagicMock()
        strategy.estimate.return_value = {
            "success": True,
            "data": {"peso_estimado_kg": 500.0},
        }
        service = WeightEstimatorService(strategy=strategy)

        chain = ValidationChainFactory.build(segmenter, service)
        context = ValidationContext(
            image=np.zeros((300, 300, 3), dtype=np.uint8),
            breed_constant=1.0,
            content_type="image/png",
        )

        result = chain.handle(context)

        self.assertEqual(result["data"]["peso_estimado_kg"], 500.0)
        strategy.estimate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
