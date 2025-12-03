# src/dataset.py
import json
import cv2
import numpy as np
from shapely import wkt

from .config import DATA_DIR, DAMAGE_MAP
from .data_io import list_event_files, get_label_path
from .features import extract_DMS

from tqdm import tqdm

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


def generate_augmented_patches(patch_img):
    """
    Generates 7 additional variations of the input image patch:
    - Rotations: 90, 180, 270 degrees
    - Flips: Horizontal, Vertical
    - Flips + Rotation
    """
    augmented = []

    # 1. Rotations
    r90 = cv2.rotate(patch_img, cv2.ROTATE_90_CLOCKWISE)
    r180 = cv2.rotate(patch_img, cv2.ROTATE_180)
    r270 = cv2.rotate(patch_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    augmented.extend([r90, r180, r270])

    # 2. Flips (Horizontal, Vertical, Both)
    f_h = cv2.flip(patch_img, 1)
    f_v = cv2.flip(patch_img, 0)
    f_hv = cv2.flip(patch_img, -1) # Both

    augmented.extend([f_h, f_v, f_hv])

    # 3. Extra: Flip then Rotate (to maximize variations)
    f_h_r90 = cv2.rotate(f_h, cv2.ROTATE_90_CLOCKWISE)
    augmented.append(f_h_r90)

    return augmented

def build_dataset_from_files(file_list, split: str, augment=False):
    """
    Builds X, y from a specific list of files.
    - augment=True: Enables tiered augmentation (ONLY for training set).
    - augment=False: Returns raw data (ONLY for validation set).
    """
    X, y = [], []


    print(f"Processing {len(file_list)} files. Augmentation={augment}")

    pbar = tqdm(file_list, desc="Extracting Features", unit="img")
    total_aug_samples = 0
    for img_path in pbar:
        pbar.set_postfix({
            "Total Samples": len(X),
            "Augmented": total_aug_samples
        })
        # 1. Load Image & Label
        # (Assuming your get_label_path / polygons functions are available)
        label_path = get_label_path(split, img_path) # path logic might need adjust depending on how you pass files
        if not label_path.exists(): continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with open(label_path) as f:
            data = json.load(f)

        patches, labels = polygons_to_patches_and_labels(img, data)

        for p, lab in zip(patches, labels):

            # --- A. ALWAYS ADD ORIGINAL ---
            # This is the "ground truth"
            feats = extract_DMS(p)
            if not np.isnan(feats).any():
                X.append(feats)
                y.append(lab)

            # --- B. AUGMENTATION (Only if enabled) ---
            if augment and lab != 0:
                multiplier = 0

                # STRATEGY BASED ON YOUR DISTRIBUTION:
                # Class 0: ~38k (No Aug)
                # Class 3: ~2.7k -> x1 (adds 7 vars)  -> ~21k total
                # Class 1: ~159  -> x10 (adds 70 vars) -> ~11k total
                # Class 2: ~56   -> x20 (adds 140 vars)-> ~8k total

                if lab == 3:   multiplier = 1
                elif lab == 1: multiplier = 10
                elif lab == 2: multiplier = 20

                if multiplier > 0:
                    base_aug = generate_augmented_patches(p) # Generates 7 images

                    # Duplicate list to meet multiplier requirement
                    final_batch = base_aug * multiplier

                    batch_count = 0
                    for aug_p in final_batch:
                        aug_feats = extract_DMS(aug_p)
                        if not np.isnan(aug_feats).any():
                            X.append(aug_feats)
                            y.append(lab)
                            batch_count += 1

                    total_aug_samples += batch_count

                    #Optional: Log if we hit a critical class to verify it's working
                    if lab == 2:
                        tqdm.write(f"  [+] Augmented Major Damage ({img_path.name}): +{batch_count} samples")
    print(f"\nDone. Extracted {len(y)} samples.")
    return np.array(X), np.array(y)


def build_dataset(events, split="train", max_images_per_event=3, data_augmentation: bool=False):
    """
    Construye el dataset de entrenamiento/validación a nivel edificio.
    """
    X, y = [], []

    # Global counters for summary
    total_patches = 0
    total_augmented = 0
    class_counts = {} # Dynamic counter

    print(f"\n{'='*40}")
    print(f"BUILDING DATASET: {split.upper()}")
    print(f"{'='*40}")

    for ev_idx, ev in enumerate(events):
        img_files = list_event_files(split, ev)[:max_images_per_event]
        print(f"\n[{ev_idx+1}/{len(events)}] Event: {ev}")
        print(f"{split} -> Found {len(img_files)} images to process")

        for i, img_path in enumerate(img_files):
            label_path = get_label_path(split, img_path)

            print(f"  Processing image {i+1}/{len(img_files)}: {img_path.name}", end="... ")

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

            # Count buildings in this specific image
            n_patches_img = len(patches)
            n_damage_img = sum(1 for l in labels if l > 0)

            status_msg = "OK"
            if n_damage_img > 0:
                status_msg = f"OK (Found {n_damage_img} DAMAGED buildings!)"
            print(status_msg)

            for p, lab in zip(patches, labels):
                # Always add the original image features
                feats = extract_DMS(p)
                X.append(feats)
                y.append(lab)

                # Update counters
                class_counts[lab] = class_counts.get(lab, 0) + 1
                total_patches += 1

                # DATA AUGMENTATION STRATEGY
                if data_augmentation and lab != 0:
                    multiplier = 0 # Default: Don't augment

                    # STRATEGY CALCULATIONS:
                    # Class 3 (2709) -> x1  -> ~21k samples (Good)
                    # Class 1 (159)  -> x10 -> ~11k samples (Good)
                    # Class 2 (56)   -> x20 -> ~8k  samples (Acceptable)

                    if lab == 3:   # Destroyed
                        multiplier = 1
                    elif lab == 1: # Minor Damage
                        multiplier = 10
                    elif lab == 2: # Major Damage
                        multiplier = 20

                    # If the class is NOT 'no-damage' (0), generate variations
                    # This helps balance classes 1, 2, and 3
                    if split == "train" and multiplier > 0:
                        # Generate geometric variations (rotations/flips)
                        base_aug_patches = generate_augmented_patches(p)
                        final_aug_patches = base_aug_patches * multiplier

                        aug_count = 0
                        for aug_p in final_aug_patches:
                            aug_feats = extract_DMS(aug_p)
                            if not np.isnan(aug_feats).any():
                                X.append(aug_feats)
                                y.append(lab)
                                aug_count += 1

                        total_augmented += aug_count
                        # Log specifically when we boost a rare class
                        print(f"    -> Augmented Class {lab}: +{aug_count} samples")

    X = np.array(X)
    y = np.array(y)

    print(f"\n{'='*40}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*40}")
    print(f"Total Feature Vectors: {len(y)}")
    print(f" - Original Patches:   {total_patches}")
    if data_augmentation:
        print(f" - Augmented Patches:  {total_augmented}")
    print("-" * 20)
    print("Class Distribution (Final):")

    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        # Calculate percentage
        pct = (c / len(y)) * 100
        print(f"  Class {u}: {c:5d} samples ({pct:.2f}%)")

    print(f"{'='*40}\n")
    """print("Total edificios:", len(y))
    if len(y) > 0:
        print("Distribución de clases:", np.bincount(y))"""

    return X, y
