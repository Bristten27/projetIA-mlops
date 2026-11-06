# Utiliser une image Python officielle légère
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application et le modèle entraîné
COPY app.py .
COPY model.pkl .

# Exposer le port sur lequel l'API va tourner
EXPOSE 8000

# Commande pour lancer l'application avec Uvicorn
CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "8000"]
