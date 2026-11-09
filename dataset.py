from pathlib import Path

try:
    import torchvision
    import torchvision.transforms as transforms
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Torchvision est manquant. Installez les dépendances du projet avec : "
        "pip install -r requirements.txt"
    ) from exc

# Transformations pour redimensionner les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Stockage du dataset au niveau du projet, sans dépendre du dossier d'exécution courant.
data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

flowers_dir = data_dir / "flowers-102"

if flowers_dir.exists():
    print(f"Dataset déjà présent dans {flowers_dir}. Chargement depuis le stockage local...")
    download = False
else:
    print(f"Téléchargement du dataset Oxford 102 dans {data_dir}...")
    download = True

try:
    train_dataset = torchvision.datasets.Flowers102(
        root=str(data_dir),
        split='train',
        transform=transform,
        download=download,
    )
except Exception as exc:
    raise RuntimeError(
        f"Le dataset Oxford 102 n'a pas pu être téléchargé ou chargé dans {data_dir}. "
        "Vérifie ta connexion internet et que les fichiers de données sont bien présents."
    ) from exc

print(f"Dataset prêt ! {len(train_dataset)} images chargées.")