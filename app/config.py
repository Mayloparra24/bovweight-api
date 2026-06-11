import os
import torch

RESOLUCION_OPTIMIZADA: int = int(os.environ.get("BOVWEIGHT_RESOLUCION", 1024))

BEST_PT_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

DEPTH_PRO_SRC: str = os.environ.get(
    "DEPTH_PRO_SRC",
    "/content/ml-depth-pro/src",
)

DEPTH_PRO_CHECKPOINT: str = os.environ.get(
    "DEPTH_PRO_CHECKPOINT",
    "/content/ml-depth-pro/checkpoints/depth_pro.pt",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
