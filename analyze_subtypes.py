from pathlib import Path
import json
from collections import Counter

import polars as pl

from src.config import DATA_DIR  # DATA_DIR = raíz_del_proyecto / "Datos"


def iter_label_files(split: str = "hold"):
    """
    Itera sobre todos los JSON *_post_disaster.json
    en Datos/{split}/labels.
    """
    labels_dir = DATA_DIR / split / "labels"
    pattern = "*_post_disaster.json"
    return sorted(labels_dir.glob(pattern))


def extract_subtypes_from_json(json_path: Path):
    """
    Devuelve una lista de subtypes encontrados en un archivo JSON.

    Tiene en cuenta que xBD puede tener:
        features["xy"]  o  features["lng_lat"]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    feats_xy = data.get("features", {}).get("xy", [])
    feats_ll = data.get("features", {}).get("lng_lat", [])

    subtypes = []

    for feat in feats_xy:
        props = feat.get("properties", {})
        st = props.get("subtype", "unknown")
        subtypes.append(st)

    for feat in feats_ll:
        props = feat.get("properties", {})
        st = props.get("subtype", "unknown")
        subtypes.append(st)

    return subtypes


def main():
    split = "train"
    print(f"Analizando subtypes en split: {split}")

    all_subtypes = []
    file_paths = iter_label_files(split)

    if not file_paths:
        print(f"[WARN] No se encontraron JSON en {DATA_DIR / split / 'labels'}")
        return

    for json_path in file_paths:
        subtypes = extract_subtypes_from_json(json_path)
        all_subtypes.extend(subtypes)

    counter = Counter(all_subtypes)
    total = sum(counter.values())

    print("\n✔ Subtypes encontrados:")
    for st, cnt in counter.most_common():
        print(f"  - {st:15s}: {cnt:6d} ({cnt/total:6.2%})")

    # Crear DataFrame en Polars
    df = pl.DataFrame(
        {
            "subtype": list(counter.keys()),
            "count": list(counter.values()),
        }
    ).with_columns(
        (pl.col("count") / total).alias("proportion")
    ).sort("count", descending=True)

    print("\n📊 Tabla en Polars:")
    print(df)

    # Opcional: guardar a CSV
    out_path = Path("figuras") / "subtype_distribution_train.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.write_csv(out_path)
    print(f"\n[OK] Tabla guardada en: {out_path}")


if __name__ == "__main__":
    main()
