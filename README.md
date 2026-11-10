# Projet_IA - Pipeline MLOps End-to-End (Oxford 102 Flowers Classification)

## Membres du groupe

* **421 H-TOL** ANDRIANJARA Jean de Bristten
* **495 H-TOL** TIVERNE Solonomena Stéphane
* **467 H-TOL** RAMBININTSOA Flavien Alfred
* **435 H-TOL** RAFIDISON Jean Aime
* **455 H-TOL** ODONARDO DESIRE Andriamiadana Avenir

---

## 📌 Description du Projet

Ce projet implémente un pipeline MLOps complet et automatisé pour la classification d'images de fleurs basée sur le jeu de données **Oxford 102 Flowers**. Il couvre l'ensemble du cycle de vie d'un modèle de Deep Learning et de Machine Learning : de l'extraction de caractéristiques par PyTorch, l'entraînement et le versionnage, jusqu'à l'intégration continue (CI/CD GitHub Actions), la containerisation Docker RAM-optimisée, le déploiement cloud d'une API haute performance sur **Render** et l'exposition d'une interface utilisateur professionnelle sur **Hugging Face Spaces**.

---

## Voici la capture de notre pipeline CI/CD

<img width="1919" height="878" alt="Capture Pipeline" src="https://github.com/user-attachments/assets/e76588f4-5f3b-4eef-9274-3571537fb1d3" />

---

## 🌐 Liens de Déploiement & Démos

* **Backend API (Render) :** [https://projetia-mlops.onrender.com](https://projetia-mlops.onrender.com) (Documentation interactive Swagger disponible sur `/docs`)
* **Frontend / Interface Utilisateur (Hugging Face Spaces) :** Hébergé et connecté en temps réel à l'API backend.

---

## 🗂️ Structure du Projet

```text
projetIA-mlops/
├── .github/
│   └── workflows/
│       └── main.yml             # Pipeline CI/CD GitHub Actions (tests & build)
├── app.py                       # API FastAPI pour l'inférence en temps réel (Top-K)
├── Dockerfile                   # Configuration Docker optimisée (CPU-only PyTorch, <512Mo RAM)
├── model.pkl                    # Modèle de classification entraîné sérialisé
├── requirements.txt             # Dépendances allégées pour la production
└── README.md                    # Documentation du projet

```

---

## 🛠️ Composants & Architecture MLOps

### 1. Feature Extraction & Modélisation

* **Dataset :** Oxford 102 Flower Dataset (classification multi-classes d'espèces florales).
* **Pipeline de vision :** Utilisation de **PyTorch / Torchvision** pour le prétraitement des images (redimensionnement, normalisation) et couplage avec un modèle de classification **Scikit-learn** sérialisé via `joblib` (`model.pkl`).

### 2. API d'Inférence (FastAPI)

* Le fichier `app.py` expose une API web moderne construite avec **FastAPI**, permettant de recevoir une image (multipart/form-data) et de retourner l'espèce prédite ainsi que le **Top-K** des prédictions avec leurs scores de confiance.

### 3. Intégration Continue (CI/CD - GitHub Actions)

* Le workflow configuré dans `.github/workflows/main.yml` s'exécute automatiquement à chaque modification (`push`) pour valider l'intégrité du code, des dépendances et s'assurer que le chargement du modèle fonctionne sans erreur.

### 4. Containerisation (Docker) & Déploiement (Render)

* **Optimisation RAM :** Utilisation d'une image `python:3.11-slim` et installation explicite de **PyTorch en version CPU-only** (`--index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)`) pour respecter strictement la limite des 512 Mo de RAM de l'offre gratuite Render.
* **Hébergement :** Déployé et configuré avec liaison dynamique de port (`$PORT`) sur **Render**.

### 5. Interface Utilisateur (Hugging Face Spaces - Statique)

* Une interface web moderne et professionnelle développée en HTML/JS (Tailwind CSS) hébergée sur un Hugging Face Space statique, permettant aux utilisateurs et au jury de tester directement le modèle par glisser-déposer d'images.

---

## 🚀 Installation & Utilisation Locale

### 1. Cloner le dépôt et installer les dépendances

```bash
git clone https://github.com/Bristten27/projetIA-mlops.git
cd projetIA-mlops
pip install -r requirements.txt

```

### 2. Lancer l'API localement

```bash
uvicorn app:app --reload --port 8000

```

Accédez ensuite à la documentation interactive sur **`[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`**.

### 3. Exécuter avec Docker

* **Construction de l'image :**

```bash
docker build -t projetia-mlops .

```

* **Lancement du conteneur :**

```bash
docker run -p 8000:8000 -e PORT=8000 projetia-mlops

```

---

## 📦 Dépendances Principales (`requirements.txt`)

* **FastAPI** & **Uvicorn** (Framework et serveur ASGI pour l'API)
* **Scikit-Learn** & **Joblib** (Modélisation et sérialisation du classifieur)
* **PyTorch** & **Torchvision (CPU-only)** (Traitement d'images et tenseurs)
* **Pillow** (Manipulation et décodage d'images)
* **Python-Multipart** (Gestion des fichiers d'images uploadés)
