import sys
import os
import cv2
import numpy as np
import torch

_DEPTH_PRO_SRC = "/content/ml-depth-pro/src"
if os.path.isdir(_DEPTH_PRO_SRC) and _DEPTH_PRO_SRC not in sys.path:
    sys.path.insert(0, _DEPTH_PRO_SRC)

import depth_pro

RESOLUCION_OPTIMIZADA = int(os.environ.get("BOVWEIGHT_RESOLUCION", 1024))

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELO_DEPTH, TRANSFORMACION = depth_pro.create_model_and_transforms()
MODELO_DEPTH = MODELO_DEPTH.to(dispositivo).half()
MODELO_DEPTH.eval()


def estimar_profundidad(
    imagen_rgb: np.ndarray,
    area_pixeles: int,
    mascara: np.ndarray,
) -> dict:
    alto_orig, ancho_orig = imagen_rgb.shape[:2]

    imagen_small = cv2.resize(
        imagen_rgb,
        (RESOLUCION_OPTIMIZADA, RESOLUCION_OPTIMIZADA),
        interpolation=cv2.INTER_AREA,
    )
    imagen_tensor = TRANSFORMACION(imagen_small).to(dispositivo).half()

    with torch.no_grad():
        prediccion = MODELO_DEPTH.infer(imagen_tensor, f_px=None)
        mapa_profundidad_small = prediccion["depth"].float().cpu().numpy()
        distancia_focal_small = prediccion["focallength_px"].item()

    mapa_profundidad = cv2.resize(
        mapa_profundidad_small,
        (ancho_orig, alto_orig),
        interpolation=cv2.INTER_LINEAR,
    )

    distancia_focal = distancia_focal_small * (1536.0 / RESOLUCION_OPTIMIZADA)

    distancias_bovino = mapa_profundidad[mascara == 1]
    if len(distancias_bovino) == 0:
        raise ValueError(
            "No se pudo calcular la profundidad sobre la region del bovino."
        )

    distancia_promedio_metros = float(np.median(distancias_bovino))
    factor_escala = (distancia_promedio_metros / distancia_focal) ** 2
    area_real_m2 = area_pixeles * factor_escala

    return {
        "distancia_promedio_metros": distancia_promedio_metros,
        "distancia_focal_px": float(distancia_focal),
        "area_real_m2": float(area_real_m2),
    }
