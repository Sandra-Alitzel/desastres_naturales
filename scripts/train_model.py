# scripts/train_model.py

import json
from pathlib import Path

from joblib import dump

# Para que "src" sea visible al ejecutar desde la raíz del repo:
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import EVENTS
from src.dataset import build_dataset, build_dataset_from_files
from src.model import train_damage_classifier, train_damage_classifier_presplit
from src.data_io import list_event_files

try:
    MAX_IMAGES = int(sys.argv[1])
except:
    print("[WARN] Max images usage parameter missing using default = 3 for testing")
    MAX_IMAGES = 3

DATA_DIR = ROOT_DIR /  Path("Datos") # Adjust if your folder is named differently
EVENTS = ["mexico-earthquake", "santa-rosa-wildfire"] # Add all your event names

def main():
    print("=== Entrenando modelo de daño (DamageLens) ===")

    # ---------------------------------------------------------
    # 1. GATHER FILE PATHS
    # ---------------------------------------------------------
    print("\n[1] Scanning directories...")

    train_files = list_event_files("train", EVENTS[0]) + list_event_files("train", EVENTS[1])
    hold_files  = list_event_files("hold", EVENTS[0]) + list_event_files("hold", EVENTS[1])
    test_files  = list_event_files("test", EVENTS[0]) + list_event_files("test", EVENTS[1])

    print(f"  Found {len(train_files)} training files")
    print(f"  Found {len(hold_files)} holdout/validation files")
    print(f"  Found {len(test_files)} test files")

    """
    train_files = train_files[:MAX_IMAGES]
    hold_files = hold_files[:MAX_IMAGES]
    test_files = test_files[:MAX_IMAGES]
    """

    if len(train_files) == 0:
        print("[ERR] No training files found. Check paths.")
        sys.exit(1)

    # A. TRAIN SET -> Enable Augmentation (Tiered Strategy)
    print("\n[2] Building TRAIN set (Augmentation ENABLED)...")
    X_train, y_train = build_dataset_from_files(train_files, "train", augment=True)

    # B. HOLD SET -> Disable Augmentation (Raw data for validation)
    print("\n[3] Building HOLD set (Augmentation DISABLED)...")
    X_hold, y_hold = build_dataset_from_files(hold_files, "hold", augment=False)

    # C. TEST SET -> Disable Augmentation (Raw data for final metric)
    print("\n[4] Building TEST set (Augmentation DISABLED)...")
    X_test, y_test = build_dataset_from_files(test_files, "test", augment=False)

    # 5. Train
    clf, metrics = train_damage_classifier_presplit(X_train, y_train, X_hold, y_hold)

    print("\n[6] Final Evaluation on TEST set...")
    if len(y_test) > 0:
        # Use the trained clf to predict on the separate Test set
        y_test_pred = clf.predict(X_test)

        # Calculate metrics specifically for Test
        from sklearn.metrics import classification_report, confusion_matrix
        print("--- TEST SET REPORT ---")
        print(classification_report(y_test, y_test_pred))
        print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))
    else:
        print("[WARN] No labeled data found in 'test' folder (or folder empty).")


    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "damage_clf_final.pkl"
    dump(clf, model_path)
    print(f"\n[OK] Model saved to {model_path}")

    print("=== Entrenamiento completo ===")

if __name__ == "__main__":
    main()
