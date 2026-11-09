import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "model.pkl"


def build_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_datasets(transform):
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "flowers-102").exists():
        print(f"Téléchargement du dataset Oxford 102 dans {data_dir}...")
        download = True
    else:
        print(f"Dataset déjà présent dans {data_dir}. Chargement depuis le stockage local...")
        download = False

    train_dataset = torchvision.datasets.Flowers102(
        root=str(data_dir),
        split="train",
        transform=transform,
        download=download,
    )
    val_dataset = torchvision.datasets.Flowers102(
        root=str(data_dir),
        split="val",
        transform=transform,
        download=False,
    )
    return train_dataset, val_dataset


def build_feature_extractor(device):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


def extract_features(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def main():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(str(ROOT_DIR / "mlruns"))
    mlflow.set_experiment("flowers102_training")

    transform = build_transform()
    train_dataset, val_dataset = get_datasets(transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")

    feature_extractor = build_feature_extractor(device)
    X_train, y_train = extract_features(feature_extractor, train_dataset, device, batch_size=64)
    X_val, y_val = extract_features(feature_extractor, val_dataset, device, batch_size=64)

    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0,
    )

    with mlflow.start_run(run_name="flowers102_resnet18_logreg") as run:
        classifier.fit(X_train, y_train)

        train_pred = classifier.predict(X_train)
        val_pred = classifier.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)

        hyperparams = {
            "learning_rate": 0.01,
            "epochs": 20,
            "batch_size": 64,
            "optimizer": "lbfgs",
            "model_name": "resnet18_imagenet_pretrained",
            "num_classes": len(train_dataset.classes),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }

        mlflow.log_params(hyperparams)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.sklearn.log_model(classifier, artifact_path="classifier_model")

        class_to_idx = {label: idx for idx, label in enumerate(train_dataset.classes)}
        artifact = {
            "classifier": classifier,
            "feature_extractor": feature_extractor,
            "classes": train_dataset.classes,
            "class_to_idx": class_to_idx,
            "transform": transform,
            "device": str(device),
        }
        joblib.dump(artifact, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="artifacts")

        print(f"Run MLflow : {run.info.run_id}")
        print(f"Accuracy train : {train_accuracy:.4f}")
        print(f"Accuracy val : {val_accuracy:.4f}")
        print(f"Modèle enregistré : {MODEL_PATH}")

    return MODEL_PATH


if __name__ == "__main__":
    main()
