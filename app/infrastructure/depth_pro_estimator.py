import sys
import os
import cv2
import numpy as np
import torch

from app.config import DEPTH_PRO_SRC, RESOLUCION_OPTIMIZADA, DEVICE
from app.domain.protocols import DepthEstimatorProtocol
from app.domain.schemas import DepthResult

if os.path.isdir(DEPTH_PRO_SRC) and DEPTH_PRO_SRC not in sys.path:
    sys.path.insert(0, DEPTH_PRO_SRC)

import depth_pro


class DepthProEstimator:
    """Adaptador del modelo Depth Pro al DepthEstimatorProtocol del dominio."""

    def __init__(
        self,
        resolution: int = RESOLUCION_OPTIMIZADA,
        device: torch.device = DEVICE,
    ):
        self.resolution = resolution
        self.device = device
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(DEPTH_PRO_SRC))
        self.model, self.transform = depth_pro.create_model_and_transforms()
        os.chdir(original_dir)
        self.model = self.model.to(device).half()
        self.model.eval()

    def estimate(
        self,
        image_rgb: np.ndarray,
        area_pixels: int,
        mask: np.ndarray,
    ) -> DepthResult:
        alto_orig, ancho_orig = image_rgb.shape[:2]

        image_small = cv2.resize(
            image_rgb,
            (self.resolution, self.resolution),
            interpolation=cv2.INTER_AREA,
        )
        image_tensor = self.transform(image_small).to(self.device).half()

        with torch.no_grad():
            prediction = self.model.infer(image_tensor, f_px=None)
            depth_map_small = prediction["depth"].float().cpu().numpy()
            focal_small = prediction["focallength_px"].item()

        depth_map = cv2.resize(
            depth_map_small,
            (ancho_orig, alto_orig),
            interpolation=cv2.INTER_LINEAR,
        )

        focal_length = focal_small * (1536.0 / self.resolution)

        distances_bovine = depth_map[mask == 1]
        if len(distances_bovine) == 0:
            raise ValueError(
                "No se pudo calcular la profundidad sobre la region del bovino."
            )

        avg_distance_m = float(np.median(distances_bovine))
        scale_factor = (avg_distance_m / focal_length) ** 2
        real_area_m2 = area_pixels * scale_factor

        return DepthResult(
            avg_distance_m=avg_distance_m,
            focal_length_px=float(focal_length),
            real_area_m2=float(real_area_m2),
        )
