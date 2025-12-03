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
    """Entrena y guarda el modelo de clasificación de daños.

    Este script se encarga de:
    1. Construir el dataset de entrenamiento y test.
    2. Entrenar el clasificador de daños.
    3. Guardar el modelo entrenado y las métricas de evaluación.
    """
    print("=== Entrenando modelo de daño (DamageLens) ===")

    # Construimos el dataset a nivel edificio (train)
    X, y = build_dataset(
        events=EVENTS,
        split="train",
        # Se realiza un sampleo dinámico
        max_samples_per_class=4000,
        augment=True
    )
    X_test, y_test = build_dataset(
        events=EVENTS,
        split="test",
        augment=False
    )

    # Entrenamiento del modelo
    clf, metrics = train_damage_classifier(X, y, X_test, y_test)

    # Guardado del modelo y métricas
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "damage_clf.pkl"
    dump(clf, model_path)
    print(f"[OK] Modelo guardado en: {model_path}")

    metrics_path = models_dir / "damage_metrics.json"
    metrics_serializable = {
        "f1_macro": float(metrics["f1_macro"]),
        "mcc": float(metrics["mcc"]),
        "kappa": float(metrics["kappa"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "report": metrics["report"]
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=4)
    print(f"[OK] Métricas guardadas en: {metrics_path}")
    
    print("=== Entrenamiento completo ===")

if __name__ == "__main__":
    main()