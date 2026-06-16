import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from app.infrastructure.handlers import ValidationHandler


class MockHandler(ValidationHandler):
    def __init__(self, result=None, exc=None):
        super().__init__()
        self.result = result
        self.exc = exc

    def handle(self, context):
        if self.exc:
            raise self.exc
        return self.result


class TestPredictEndpoint(unittest.TestCase):
    def _fake_decode_success(self, contents, flags):
        return np.zeros((300, 300, 3), dtype=np.uint8)

    def test_predict_weight_success(self):
        with patch("app.api.predict.cv2.imdecode", side_effect=self._fake_decode_success):
            with patch(
                "app.api.predict.get_validation_chain",
                return_value=MockHandler(result={"success": True, "data": {"peso_estimado_kg": 450.0}}),
            ):
                from app.main import app

                client = TestClient(app)
                response = client.post(
                    "/api/v1/predict-weight",
                    data={"constante_raza": "0.95"},
                    files={"file": ("vaca.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["peso_estimado_kg"], 450.0)

    def test_predict_weight_invalid_image(self):
        with patch("app.api.predict.cv2.imdecode", return_value=None):
            with patch(
                "app.api.predict.get_validation_chain",
                return_value=MockHandler(),
            ):
                from app.main import app

                client = TestClient(app)
                response = client.post(
                    "/api/v1/predict-weight",
                    data={"constante_raza": "0.95"},
                    files={"file": ("vaca.jpg", io.BytesIO(b"not-an-image"), "image/jpeg")},
                )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
