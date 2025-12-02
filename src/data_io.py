# src/data_io.py
from pathlib import Path
import json
import random

import cv2
import numpy as np
from shapely import wkt

from .config import DATA_DIR, DAMAGE_MAP


def list_event_files(split: str, event: str):
    """
    Devuelve la lista de imágenes POST-DESASTRE para un evento dado.

    Estructura esperada:
      DATA_DIR / <split> / "images" / "<event>_*_post_disaster.png"
    """
    img_dir = DATA_DIR / split / "images"

    if not img_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de imágenes: {img_dir}")

    pattern = f"{event}_*_post_disaster.png"
    files = sorted(img_dir.glob(pattern))
    if not files:
        print(f"[WARN] No encontré {pattern} en {img_dir}")
    return files


def get_label_path(split: str, img_path: Path):
    """
    A partir del path de una imagen post_desastre (.png)
    devuelve el path del JSON correspondiente post_desastre.
    """
    label_dir = DATA_DIR / split / "labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de labels: {label_dir}")

    label_path = label_dir / img_path.name.replace(".png", ".json")
    return label_path


def load_image_and_label(event: str, split: str = "train"):
    """
    Elige aleatoriamente UNA imagen post_desastre de un evento
    y carga la imagen RGB y el JSON.

    Devuelve:
      img_rgb, data_json, img_path, label_path
    """
    files = list_event_files(split, event)
    if not files:
        raise RuntimeError(f"No encontré imágenes de {event} en {split}")

    img_path = random.choice(files)
    label_path = get_label_path(split, img_path)

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(label_path) as f:
        data = json.load(f)

    return img, data, img_path, label_path


def load_pre_image_from_post_path(img_post_path: Path):
    """
    A partir de un path POST-DESASTRE, intenta cargar la imagen PRE-DESASTRE
    correspondiente cambiando 'post_disaster' -> 'pre_disaster' en el nombre.
    """
    pre_name = img_post_path.name.replace("post_disaster", "pre_disaster")
    pre_path = img_post_path.with_name(pre_name)

    if not pre_path.exists():
        print(f"[WARN] No encontré pre-desastre: {pre_path}")
        return None

    img_pre = cv2.imread(str(pre_path))
    if img_pre is None:
        print(f"[WARN] No pude leer {pre_path}")
        return None

    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
    return img_pre


def json_to_mask_and_polygons(data, shape):
    """
    A partir del JSON de xBD genera:

      - mask: máscara binaria de edificios (h x w)
      - polys: lista de polígonos (numpy arrays de Nx2)
      - damage_labels: lista de subtypes (string) por polígono
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    polys = []
    damage_labels = []

    for feat in data["features"]["xy"]:
        props = feat.get("properties", {})
        subtype = props.get("subtype", "no-damage")

        poly = wkt.loads(feat["wkt"])
        coords = np.array(poly.exterior.coords, dtype=np.int32)

        cv2.fillPoly(mask, [coords], 1)
        polys.append(coords)
        damage_labels.append(subtype)

    return mask, polys, damage_labels
