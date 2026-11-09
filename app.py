from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model.pkl"
artifact = joblib.load(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = artifact["feature_extractor"].to(device)
feature_extractor.eval()
classifier = artifact["classifier"]
classes = artifact["classes"]
image_transform = artifact["transform"]

app = FastAPI(
    title="Flowers102 Prediction API",
    description="Classification d'images de fleurs avec ResNet18 et Flowers102.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Flowers102 API is running.", "status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k doit être supérieur ou égal à 1.")

    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas une image valide.") from exc
    finally:
        await file.close()

    input_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = feature_extractor(input_tensor).cpu().numpy()

    probabilities = classifier.predict_proba(features)[0]
    best_indices = np.argsort(probabilities)[::-1][: min(top_k, len(classes))]
    predicted_index = int(best_indices[0])

    return {
        "predicted_species": classes[predicted_index],
        "confidence_score": float(probabilities[predicted_index]),
        "top_predictions": [
            {
                "predicted_species": classes[int(index)],
                "confidence_score": float(probabilities[index]),
            }
            for index in best_indices
        ],
    }
