# Projet_IA - Pipeline MLOps End-to-End (Iris Classification & Monitoring)

## Membres du groupe
* **421 H-TOL** ANDRIANJARA Jean de Bristten
* **495 H-TOL** TIVERNE Solonomena Stéphane
* **467 H-TOL** RAMBININTSOA Flavien Alfred
* **435 H-TOL** RAFIDISON Jean Aime
* **455 H-TOL** ODONARDO DESIRE Andriamiadana Avenir

---

## 📌 Description du Projet
Ce projet implémente un pipeline MLOps complet et automatisé pour la classification du jeu de données Iris. Il couvre l'ensemble du cycle de vie d'un modèle d'intelligence artificielle : du versionnage des données et du suivi des expériences jusqu'à l'intégration continue (CI/CD), la containerisation Docker, le déploiement d'une API web haute performance et le monitoring de la dérive des données.

---

## 🗂️ Structure du Projet
```text
projetIA-mlops/
├── .github/
│   └── workflows/
│       └── main.yml             # Pipeline CI/CD GitHub Actions
├── mlruns/                      # Artifacts et suivi des exécutions MLflow
├── app.py                       # API FastAPI pour l'inférence en temps réel
├── Dockerfile                   # Configuration de la containerisation Docker
├── docker-compose.yml           # Orchestration des services
├── dvc.yaml                     # Pipeline de données DVC
├── evidently_report.html        # Rapport de monitoring et Data Drift (Evidently AI)
├── iris_dataset.csv             # Jeu de données Iris
├── iris_dataset.csv.dvc         # Fichier de suivi DVC des données
├── model.pkl                    # Modèle entraîné sérialisé
├── requirements.txt             # Dépendances du projet
└── README.md                    # Documentation du projet

```

---

## 🛠️ Composants & Architecture MLOps

### 1. Données & Versionnage (DVC)

* **Dataset :** `iris_dataset.csv`
* **Outil :** DVC (*Data Version Control*) pour assurer la traçabilité rigoureuse et le versionnage des fichiers de données.
* **Pipeline :** Orchestré via `dvc.yaml` pour lier de manière reproductible les étapes de préparation et d'entraînement.

### 2. Suivi des Expériences (MLflow)

* Le projet intègre **MLflow** pour enregistrer et suivre l'ensemble des paramètres, des métriques de performance et des artifacts des modèles entraînés (centralisés dans le dossier `mlruns/`).

### 3. API d'Inférence (FastAPI)

* Le fichier `app.py` expose une API web moderne construite avec **FastAPI**, permettant de consommer le modèle de classification en temps réel via des requêtes HTTP POST.
* **Route racine (`/`) :** Vérification de l'état opérationnel du service.
* **Route de prédiction (`/predict`) :** Reçoit les dimensions des sépales et pétales et retourne l'espèce prédite (`setosa`, `versicolor`, `virginica`).

### 4. Intégration Continue (CI/CD - GitHub Actions)

* Le workflow configuré dans `.github/workflows/main.yml` s'exécute automatiquement à chaque modification (`push`) sur la branche principale pour valider l'intégrité du code, des tests et des dépendances.

### 5. Containerisation (Docker)

* L'application est entièrement conteneurisée pour garantir un déploiement fluide, isolé et reproductible sur n'importe quel environnement de production (Render, Hugging Face, etc.).

### 6. Monitoring & Data Drift (Evidently AI)

* Le rapport `evidently_report.html` généré dans le projet permet d'analyser la qualité statistique des données et de détecter d'éventuelles dérives (*Data Drift*) par rapport au jeu de données d'entraînement initial.

---

## 🚀 Installation & Utilisation Locale

### 1. Cloner le dépôt et installer les dépendances

```bash
git clone [https://github.com/Bristten27/projetIA-mlops.git](https://github.com/Bristten27/projetIA-mlops.git)
cd projetIA-mlops
pip install -r requirements.txt

```

### 2. Lancer l'API localement

```bash
uvicorn app:app --reload

```

Accédez ensuite à la documentation interactive sur **`http://127.0.0.1:8000/docs`**.

### 3. Exécuter avec Docker

* **Construction de l'image :**
```bash
docker build -t projetia-mlops .

```


* **Lancement du conteneur :**
```bash
docker run -p 8000:8000 projetia-mlops uvicorn app:app --host 0.0.0.0 --port 8000

```



---

## 📦 Dépendances Principales (`requirements.txt`)

* **FastAPI** & **Uvicorn** (Framework et serveur ASGI)
* **Scikit-Learn** (Modélisation ML)
* **MLflow** (Tracking des expériences)
* **DVC** (Versionnage des données)
* **Evidently** (Monitoring de la dérive des données)
* **Pandas** & **NumPy** (Manipulation et calcul numérique)
* **Joblib** (Sérialisation du modèle)

```

```