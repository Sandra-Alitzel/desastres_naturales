# src/model.py
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    cohen_kappa_score, classification_report
)

from .config import INV_DAMAGE_MAP

def train_damage_classifier_presplit(X_train, y_train, X_val, y_val, random_state=42):

    print("Training Random Forest...")
    print(f"Train set shape: {X_train.shape}")
    print(f"Val set shape:   {X_val.shape}")

    # 2. Define the Pipeline with SMOTE
    # SMOTE happens ONLY on X_train, never on validation data.
    # k_neighbors must be smaller than the number of samples in the minority class.
    # Since you have classes with ~3 samples, we must lower k_neighbors.
    smote = SMOTE(random_state=random_state, k_neighbors=1)

    # We use 'balanced' class_weight to handle any remaining imbalance
    # after our augmentation strategy.
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", smote), # Generates synthetic feature vectors here
        ("rf", RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced"
        ))
    ])

    clf.fit(X_train, y_train)

    print("Predicting validation set...")
    y_pred = clf.predict(X_val)

    # Metrics Calculation
    present_labels = np.unique(y_val)
    present_names = [INV_DAMAGE_MAP.get(c, str(c)) for c in present_labels]

    print("Clases presentes en validación:", present_labels)
    conf_mat = confusion_matrix(y_val, y_pred, labels=present_labels)
    print("Matriz de confusión:\n", conf_mat)

    print("\nReporte de clasificación:")
    print(classification_report(
        y_val, y_pred,
        labels=present_labels,
        target_names=present_names
    ))

    f1 = f1_score(y_val, y_pred, average="macro")
    mcc = matthews_corrcoef(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)

    metrics = {
        "f1_macro": f1,
        "mcc": mcc,
        "kappa": kappa,
        "confusion_matrix": conf_mat,
    }

    return clf, metrics

def train_damage_classifier(X, y, test_size=0.2, random_state=42):
    if len(y) == 0:
        raise RuntimeError("No hay datos en y. Revisa build_dataset / rutas.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample"
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    present_labels = np.unique(y_val)
    present_names = [INV_DAMAGE_MAP[c] for c in present_labels]

    print("Clases presentes en validación:", present_labels, present_names)

    conf_mat = confusion_matrix(y_val, y_pred, labels=present_labels)
    print("Matriz de confusión (solo clases presentes):\n", conf_mat)

    print("\nReporte de clasificación:")
    print(classification_report(
        y_val, y_pred,
        labels=present_labels,
        target_names=present_names
    ))

    f1 = f1_score(y_val, y_pred, average="macro")
    mcc = matthews_corrcoef(y_val, y_pred)
    kappa = cohen_kappa_score(y_val, y_pred)

    print("F1 macro:", f1)
    print("MCC:", mcc)
    print("Kappa:", kappa)
    print("Clases que conoce el modelo:", clf.classes_)

    metrics = {
        "f1_macro": f1,
        "mcc": mcc,
        "kappa": kappa,
        "confusion_matrix": conf_mat,
    }

    return clf, metrics
