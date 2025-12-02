# src/mdm.py
import cv2
import numpy as np
from shapely import wkt
import matplotlib.cm as mpl_cm

from .config import DAMAGE_MAP
from .features import extract_DMS


def damage_score_from_proba(proba, class_labels):
    class_labels = np.array(class_labels, dtype=np.float32)
    return float((proba * (class_labels / 3.0)).sum())


def create_MDM_for_image(img, data, model, alpha=0.6):
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    class_labels = model.classes_

    for feat in data["features"]["xy"]:
        props = feat.get("properties", {})
        subtype = props.get("subtype", "no-damage")
        if subtype not in DAMAGE_MAP:
            continue

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

        feats = extract_DMS(patch).reshape(1, -1)
        proba = model.predict_proba(feats)[0]

        score = damage_score_from_proba(proba, class_labels)

        mask_building = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_building, [coords], 1)
        heatmap[mask_building == 1] = score

    hm_norm = np.clip(heatmap, 0, 1)
    cmap = mpl_cm.get_cmap("jet")
    hm_color = (cmap(hm_norm)[..., :3] * 255).astype(np.uint8)

    img_float = img.astype(np.float32)
    overlay = ((1 - alpha) * img_float + alpha * hm_color).astype(np.uint8)

    return overlay, hm_norm
