import json
import cv2
import numpy as np
from shapely import wkt
import random 

from .config import DAMAGE_MAP, INV_DAMAGE_MAP
from .data_io import list_event_files, get_label_path
from .features import extract_DMS


def augment_patch(patch):
    """
    Aplica transformaciones geométricas aleatorias (flip horizontal y rotación)
    a un parche de imagen para generar nuevas muestras.
    """
    # Flip horizontal aleatorio
    if random.random() > 0.5:
        patch = cv2.flip(patch, 1)

    # Rotación aleatoria (90, 180, 270 grados)
    # 0 = no rota, 1 = 90°, 2 = 180°, 3 = 270°
    rotation_choice = random.choice([0, 1, 2, 3])
    if rotation_choice == 1:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_choice == 2:
        patch = cv2.rotate(patch, cv2.ROTATE_180)
    elif rotation_choice == 3:
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return patch


def polygons_to_patches_and_labels(img, data):
    """Extrae parches de edificios y sus etiquetas de daño.

    A partir de una imagen y los datos de anotación (polígonos), esta
    función recorta los parches de la imagen que corresponden a cada
    edificio y extrae su etiqueta de clasificación de daño.

    Parameters
    ----------
    img : np.ndarray
        Imagen de entrada (post-desastre) en formato NumPy.
    data : dict
        Diccionario con los datos de las etiquetas (polígonos y
        metadatos) en formato GeoJSON.

    Returns
    -------
    tuple[list[np.ndarray], list[int]]
        Una tupla conteniendo:
        - patches (list[np.ndarray]): Lista de parches de imagen,
          cada uno representando un edificio.
        - labels (list[int]): Lista de etiquetas numéricas de daño
          correspondientes a cada parche.
    """
    patches = []
    labels = []
    h, w = img.shape[:2]

    for feat in data["features"]["xy"]:
        props = feat.get("properties", {})
        subtype = props.get("subtype", "no-damage")
        if subtype not in DAMAGE_MAP:
            continue
        damage_class = DAMAGE_MAP[subtype]

        poly = wkt.loads(feat["wkt"])
        coords = np.array(poly.exterior.coords, dtype=np.int32)

        x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
        x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w - 1, x_max)
        y_max = min(h - 1, y_max)
        if x_max <= x_min or y_max <= y_min:
            continue

        patch = img[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            continue

        patches.append(patch)
        labels.append(damage_class)

    return patches, labels


def build_dataset(events, split="train", max_images_per_event=None, max_samples_per_class=2000, augment: bool = False):
    """
    Construye el dataset a nivel edificio, implementando Augmentation (Oversampling)
    para clases minoritarias y Subsampling para la clase mayoritaria.
    """
    # Colectar todos los patches y labels originales, agrupados por clase.
    patches_by_class = {c: [] for c in DAMAGE_MAP.values()}
    
    for ev in events:
        img_files_all = list_event_files(split, ev)
        
        # Aplicar el límite de imágenes (si max_images_per_event es None, usa todas)
        img_files = img_files_all
        if max_images_per_event is not None:
             img_files = img_files_all[:max_images_per_event]

        print(f"{split} / {ev} -> {len(img_files)} imágenes a procesar de {len(img_files_all)} disponibles")

        for img_path in img_files:
            label_path = get_label_path(split, img_path)
            if not label_path.exists():
                print(f"[WARN] No existe label: {label_path}")
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[WARN] No pude leer imagen: {img_path}")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            with open(label_path) as f:
                data = json.load(f)

            patches, labels = polygons_to_patches_and_labels(img, data)
            for p, lab in zip(patches, labels):
                # Almacenamos el patch (imagen), no las features
                if lab in patches_by_class:
                    patches_by_class[lab].append(p)

    # Balanceo: Augmentation (Oversampling) y Subsampling
    final_patches = []
    final_labels = []
    if augment:
        print(f"\n--- Stats para {split} con Balanceo (Augmentation / Sampling) ---")
    else:
        print(f"\n ---Stats para {split}---")
    target_N = max_samples_per_class

    for i, (class_id, original_patches) in enumerate(patches_by_class.items(), start=1):
        class_name = INV_DAMAGE_MAP[class_id]
        original_count = len(original_patches)

        if original_count == 0:
            print(f"[{class_name} (id={class_id})]: 0 muestras. Saltando.")
            continue

        # Augmentation / Oversampling (Clases Minoritarias)
        if (original_count < target_N) and augment:
            augmented_patches = list(original_patches)

            # Generar nuevas muestras por aumento
            cycle_count = 0
            while len(augmented_patches) < target_N:
                # Ciclamos sobre los patches originales disponibles
                source_patch = original_patches[cycle_count % original_count]

                # Aplicamos aumento
                new_patch = augment_patch(source_patch)
                augmented_patches.append(new_patch)
                cycle_count += 1

            N_final = len(augmented_patches)
            final_patches.extend(augmented_patches)
            final_labels.extend([class_id] * N_final)
            print(f"[{class_name} (id={class_id})]: {original_count} -> {N_final} muestras (Oversampling)")

        # Subsampling (Clase Mayoritaria)
        elif (original_count > target_N) and augment:
            # Seleccionar aleatoriamente el número objetivo de muestras
            indices = random.sample(range(original_count), target_N)
            subsampled_patches = [original_patches[i] for i in indices]

            final_patches.extend(subsampled_patches)
            final_labels.extend([class_id] * target_N)
            print(f"[{class_name} (id={class_id})]: {original_count} -> {target_N} muestras (Subsampling)")

        else:
            final_patches.extend(original_patches)
            final_labels.extend([class_id] * original_count)
            print(f"[{class_name} (id={class_id})]: {original_count} -> {original_count} muestras (No sampling)")

    # Extraer características DMS del dataset
    print("\n *** Extrayendo características DMS del dataset ***")
    X = np.array([extract_DMS(p) for p in final_patches])
    y = np.array(final_labels)
    
    print("\nTotal edificios después de balanceo:", len(y))
    if len(y) > 0:
        # np.bincount(y) solo funciona si y es una lista de enteros
        print("Distribución de clases después de balanceo:", np.bincount(y))

    return X, y