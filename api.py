from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from inference import load_model, predict


app = FastAPI(title="Face Mask Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

try:
    model, class_names = load_model()
    model_error: str | None = None
except FileNotFoundError as error:
    model = None
    class_names = ()
    model_error = str(error)


@app.get("/health")
def health() -> dict[str, str]:
    if model_error is not None:
        raise HTTPException(status_code=503, detail=model_error)
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict[str, str | float]:
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model unavailable")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPG, PNG, and WEBP images are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error
    label, confidence = predict(model, image, class_names)
    return {"label": label, "confidence": confidence}
