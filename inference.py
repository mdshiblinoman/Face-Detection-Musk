from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import CLASS_NAMES, IMAGE_SIZE, build_model


MODEL_PATH = Path(__file__).parent / "models" / "face_mask_model.pt"


def load_model() -> tuple[torch.nn.Module, tuple[str, ...]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    model = build_model("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict(
    model: torch.nn.Module,
    image: Image.Image,
    class_names: tuple[str, ...],
) -> tuple[str, float]:
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    inputs = torch.from_numpy(np.asarray(resized, dtype="float32") / 255.0)
    inputs = inputs.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    index = int(probabilities.argmax())
    return class_names[index], float(probabilities[index])
