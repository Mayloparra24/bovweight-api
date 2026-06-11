import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import get_weight_estimator
from app.application.weight_estimator import WeightEstimatorService

router = APIRouter(tags=["prediction"])

TIPOS_PERMITIDOS = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
}


@router.post("/api/v1/predict-weight")
async def predict_weight(
    file: UploadFile = File(...),
    constante_raza: float = Form(...),
    service: WeightEstimatorService = Depends(get_weight_estimator),
):
    content_type = file.content_type
    if content_type and content_type.split(";")[0].strip() not in TIPOS_PERMITIDOS:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_IMAGE_TYPE",
                    "message": f"Tipo de archivo no permitido: {content_type}. Use JPEG o PNG.",
                },
            },
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_IMAGE",
                    "message": "No se pudo decodificar la imagen enviada.",
                },
            },
        )

    return service.predict(image, constante_raza)
