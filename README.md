# Projet_IA - Pipeline MLOps End-to-End (Iris Classification & Monitoring)

# Membres du groupe
421 H-TOL ANDRIANJARA Jean de Bristten

495 H-TOL TIVERNE Solonomena Stéphane

467 H-TOL RAMBININTSOA Flavien Alfred

435 H-TOL RAFIDISON Jean Aime

455 H-TOL ODONARDO DESIRE Andriamiadana Avenir

## Description
Pipeline MLOps complet et automatisé pour la classification du jeu de données Iris. Le projet intègre le versionnage des données, le suivi des expériences, l'intégration continue (CI/CD), la containerisation Docker, une API d'inférence web et le monitoring de la dérive des données.

## Structure du projet
```text
projetIA-mlops/
├── .github/
│   └── workflows/
│       └── main.yml               # Pipeline CI/CD GitHub Actions
├── mlruns/                        # Artifacts et suivi des exécutions MLflow
├── app.py                         # API FastAPI pour l'inférence en temps réel
├── Dockerfile                     # Configuration de la containerisation Docker
├── docker-compose.yml             # Orchestration des services
├── dvc.yaml                       # Pipeline de données DVC
├── evidently_report.html          # Rapport de monitoring et Data Drift (Evidently AI)
├── iris_dataset.csv               # Jeu de données Iris
├── iris_dataset.csv.dvc           # Fichier de suivi DVC des données
├── model.pkl                      # Modèle entraîné sérialisé
├── requirements.txt               # Dépendances du projet
└── README.md                      # Ce fichier


## Données & Versionnage (DVC)
- **Dataset**: `iris_dataset.csv`
- **Outil**: DVC (Data Version Control) pour assurer la traçabilité et le versionnage des données.
- **Pipeline**: Dééfini via `dvc.yaml` pour lier les étapes de préparation et d'entraînement.

## Suivi des Expériences (MLflow)
Le projet utilise **MLflow** pour enregistrer et suivre les paramètres, les métriques et les artifacts des modèles entraînés (stockés dans le dossier `mlruns/`).

## API d'Inférence (FastAPI - app.py)
Le fichier `app.py` expose une API web construite avec **FastAPI** permettant de consommer le modèle de classification en temps réel.
- **Route racine (`/`)**: Vérification de l'état opérationnel du service.

**Lancement local de l'API**:
```bash
uvicorn app:app --reload

Intégration Continue (CI/CD - GitHub Actions)
Le workflow configuré dans .github/workflows/main.yml s'exécute automatiquement à chaque push sur la branche main pour valider l'intégrité du code et des dépendances.

Containerisation (Docker)
L'application est entièrement conteneurisée pour garantir un déploiement fluide et reproductible.

Commandes Docker:

Construction de l'image:

Bash
docker build -t projetia-mlops .
Exécution du conteneur:

Bash
docker run -p 8000:8000 projetia-mlops uvicorn app:app --host 0.0.0.0 --port 8000
Monitoring (Evidently AI)
Le rapport evidently_report.html généré dans le projet permet d'analyser la qualité des données et de détecter d'éventuelles dérives (Data Drift) par rapport aux données d'entraînement initiales.

Installation
Bash
pip install -r requirements.txt
Dépendances principales
FastAPI

Uvicorn

Scikit-Learn

MLflow

DVC

Evidently

Pandas
