# src/model.py (Versión Final Corregida)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    cohen_kappa_score, classification_report
)

from .config import INV_DAMAGE_MAP


def train_damage_classifier(X_train, y_train, X_test, y_test, test_size=0.2, random_state=42):
    if len(y_train) == 0:
        raise RuntimeError("No hay datos en y. Revisa build_dataset / rutas.")

    # 1. División del dataset en Entrenamiento y Validación
    #X_train, X_val, y_train, y_val = train_test_split(
    #    X, y, test_size=test_size, stratify=y, random_state=random_state
    #)

    # 2. Definición y Entrenamiento del Clasificador
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
    y_pred = clf.predict(X_test)

    # 3. Cálculo de Métricas
    present_labels = np.unique(y_test)
    present_names = [INV_DAMAGE_MAP[c] for c in present_labels]

    print("Clases presentes en validación:", present_labels, present_names)

    conf_mat = confusion_matrix(y_test, y_pred, labels=present_labels)
    print("Matriz de confusión (solo clases presentes):\n", conf_mat)

    # Reporte de clasificación (string para consola)
    report_str = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names
    )
    print("\nReporte de clasificación:")
    print(report_str)

    # Reporte de clasificación (diccionario para JSON)
    report_dict = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True # <-- Esto arregla el error de JSON
    )

    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"\nF1 macro: {f1}")
    print(f"MCC: {mcc}")
    print(f"Kappa: {kappa}")
    
    metrics = {
        "f1_macro": f1,
        "mcc": mcc,
        "kappa": kappa,
        # Guardamos la matriz como lista para serializar
        "confusion_matrix": conf_mat, 
        # Guardamos el diccionario para JSON
        "report": report_dict 
    }

    return clf, metrics