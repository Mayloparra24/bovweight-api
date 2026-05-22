import time
import math
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from app.segmentacion import detectar_bovino
from app.profundidad import estimar_profundidad

app = FastAPI(
    title="BovWeight CR API",
    description="Microservicio de estimacion de peso bovino con YOLOv8-seg y Depth Pro",
    version="1.0.0",
)

TIPOS_PERMITIDOS = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
    "image/tiff",
}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/predict-weight")
async def predict_weight(
    file: UploadFile = File(...),
    constante_raza: float = Form(...),
):
    if file.content_type and file.content_type.split(";")[0].strip() not in TIPOS_PERMITIDOS:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": {
                    "code": "INVALID_IMAGE_TYPE",
                    "message": f"Tipo de archivo no permitido: {file.content_type}. Use JPEG o PNG.",
                },
            },
        )

    inicio_total = time.time()

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if imagen is None:
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

        resultado_seg = detectar_bovino(imagen)
        mascara = resultado_seg["mascara"]
        area_pixeles = resultado_seg["area_pixeles"]
        confianza = resultado_seg["confianza"]

        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado_depth = estimar_profundidad(imagen_rgb, area_pixeles, mascara)

        area_real_m2 = resultado_depth["area_real_m2"]
        peso_estimado_kg = constante_raza * (area_real_m2 ** 1.5)

        tiempo_total = time.time() - inicio_total

        return {
            "success": True,
            "data": {
                "peso_estimado_kg": round(peso_estimado_kg, 2),
                "confianza_yolo": round(confianza, 4),
                "tiempo_total_segundos": round(tiempo_total, 2),
            },
        }

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": {
                    "code": "NO_BOVINE_DETECTED",
                    "message": str(e),
                },
            },
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"Error interno del servidor: {str(e)}",
                },
            },
        )
