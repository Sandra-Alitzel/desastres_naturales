# scripts/train_model.py

import json
from pathlib import Path

from joblib import dump

# Para que "src" sea visible al ejecutar desde la raíz del repo:
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import EVENTS
from src.dataset import build_dataset
from src.model import train_damage_classifier


def main():
    print("=== Entrenando modelo de daño (DamageLens) ===")

    # 1. Construir dataset a nivel edificio (train)
    X, y = build_dataset(
        events=EVENTS,
        split="train",
        max_images_per_event=3  # puedes ajustar esto
    )

    # 2. Entrenar clasificador
    clf, metrics = train_damage_classifier(X, y)

    # 3. Crear carpeta models/
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    # 4. Guardar modelo
    model_path = models_dir / "damage_clf.pkl"
    dump(clf, model_path)
    print(f"[OK] Modelo guardado en: {model_path}")

    # 5. Guardar métricas
    metrics_path = models_dir / "damage_metrics.json"
    metrics_serializable = {
        "f1_macro": float(metrics["f1_macro"]),
        "mcc": float(metrics["mcc"]),
        "kappa": float(metrics["kappa"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"[OK] Métricas guardadas en: {metrics_path}")
    print("=== Entrenamiento completo ===")


if __name__ == "__main__":
    main()
