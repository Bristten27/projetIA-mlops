import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Charger le modèle préalablement entraîné et sauvegardé
model = joblib.load("model.pkl")

# 2. Initialiser l'application FastAPI
app = FastAPI(
    title="API MLOps - Prédiction Iris",
    description="API pour prédire la classe d'une fleur Iris via un modèle Scikit-Learn",
    version="1.0"
)

# Configuration CORS pour autoriser les requêtes externes (depuis Hugging Face)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

# 3. Définir le format des données d'entrée (Payload JSON)
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 4. Route racine (Health Check)
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API MLOps ! Le service est opérationnel."}

# 5. Route de prédiction
@app.post("/predict")
def predict(features: IrisFeatures):
    # Préparer les données pour le modèle
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    # Faire la prédiction
    prediction = model.predict(input_data)
    target_names = ['setosa', 'versicolor', 'virginica']
    predicted_class = target_names[int(prediction[0])]
    
    return {
        "prediction_index": int(prediction[0]),
        "predicted_species": predicted_class
    }