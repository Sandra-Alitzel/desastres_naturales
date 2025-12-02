# src/dataset.py
import json
import cv2
import numpy as np
from shapely import wkt

from .config import DATA_DIR, DAMAGE_MAP
from .data_io import list_event_files, get_label_path
from .features import extract_DMS


def polygons_to_patches_and_labels(img, data):
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


def build_dataset(events, split="train", max_images_per_event=3):
    """
    Construye el dataset de entrenamiento/validación a nivel edificio.
    """
    X, y = [], []

    for ev in events:
        img_files = list_event_files(split, ev)[:max_images_per_event]
        print(f"{split} / {ev} -> {len(img_files)} imágenes")

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
                feats = extract_DMS(p)
                X.append(feats)
                y.append(lab)

    X = np.array(X)
    y = np.array(y)

    print("Total edificios:", len(y))
    if len(y) > 0:
        print("Distribución de clases:", np.bincount(y))

    return X, y
