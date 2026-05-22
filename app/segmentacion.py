import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_PT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
MODELO_YOLO = YOLO(BEST_PT_PATH)
MODELO_YOLO.to(dispositivo)


def detectar_bovino(imagen: np.ndarray) -> dict:
    resultado = MODELO_YOLO(imagen, verbose=False)[0]

    if resultado.masks is None:
        raise ValueError("No se detecto la silueta del bovino en la imagen.")

    mascara_original = resultado.masks.data[0].cpu().numpy()
    alto_orig, ancho_orig = imagen.shape[:2]
    mascara_redimensionada = cv2.resize(
        mascara_original,
        (ancho_orig, alto_orig),
        interpolation=cv2.INTER_NEAREST,
    )
    mascara_binaria = (mascara_redimensionada > 0.5).astype(np.uint8)
    area_pixeles = int(np.sum(mascara_binaria == 1))

    if resultado.boxes is not None and len(resultado.boxes.conf) > 0:
        confianza = float(resultado.boxes.conf[0].cpu().numpy())
    else:
        confianza = 0.0

    if resultado.boxes is not None and len(resultado.boxes.xyxy) > 0:
        x1, y1, x2, y2 = map(int, resultado.boxes.xyxy[0].cpu().numpy())
    else:
        filas, columnas = np.where(mascara_binaria == 1)
        if len(filas) == 0:
            raise ValueError("No se pudo encontrar la region del bovino.")
        x1, y1 = np.min(columnas), np.min(filas)
        x2, y2 = np.max(columnas), np.max(filas)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ancho_orig, x2), min(alto_orig, y2)
    crop = imagen[y1:y2, x1:x2].copy()

    return {
        "mascara": mascara_binaria,
        "crop": crop,
        "area_pixeles": area_pixeles,
        "confianza": confianza,
    }
