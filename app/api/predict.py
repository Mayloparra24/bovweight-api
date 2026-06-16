import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_validation_chain
from app.infrastructure.handlers import ValidationContext, ValidationHandler

router = APIRouter(tags=["prediction"])


@router.post("/api/v1/predict-weight")
async def predict_weight(
    file: UploadFile = File(...),
    constante_raza: float = Form(...),
    chain: ValidationHandler = Depends(get_validation_chain),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen enviada.")

    context = ValidationContext(
        image=image,
        breed_constant=constante_raza,
        content_type=file.content_type,
    )

    return chain.handle(context)
