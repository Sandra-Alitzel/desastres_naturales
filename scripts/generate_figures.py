# scripts/generate_figures.py

from pathlib import Path

from joblib import load
import matplotlib.pyplot as plt

# Asegurar acceso a src/
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import EVENTS
from src.data_io import load_image_and_label, load_pre_image_from_post_path
from src.mdm import create_MDM_for_image
from src.visualization import plot_pre_post_MDM_with_veg
from src.spectral import visualize_spectral_indices_pre_post


def ensure_output_dir():
    figuras_dir = ROOT_DIR / "figuras"
    figuras_dir.mkdir(exist_ok=True)
    return figuras_dir


def load_trained_model():
    model_path = ROOT_DIR / "models" / "damage_clf.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Ejecuta primero: python scripts/train_model.py"
        )
    clf = load(model_path)
    print(f"[OK] Modelo cargado desde: {model_path}")
    return clf


def generate_earthquake_figures(clf, figuras_dir):
    print("=== Figuras: Mexico Earthquake ===")
    img_quake, data_quake, img_q_path, lbl_q_path = load_image_and_label(
        "mexico-earthquake", "hold"
    )
    overlay_quake, hm_quake = create_MDM_for_image(img_quake, data_quake, clf)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_quake)
    plt.axis("off")
    plt.title("Mexico Earthquake – Post desastre")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_quake)
    plt.axis("off")
    plt.title("Mexico Earthquake – MDM edificios")

    plt.tight_layout()
    out_path = figuras_dir / "mexico_earthquake_MDM.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figura guardada: {out_path}")


def generate_wildfire_figures(clf, figuras_dir):
    print("=== Figuras: Santa Rosa Wildfire ===")
    img_post, data_post, img_post_path, lbl_post_path = load_image_and_label(
        "santa-rosa-wildfire", "hold"
    )
    img_pre = load_pre_image_from_post_path(img_post_path)

    overlay_fire, hm_fire = create_MDM_for_image(img_post, data_post, clf)

    # Figura 1: Pre / Post / MDM + vegetación quemada
    plot_pre_post_MDM_with_veg(img_pre, img_post, hm_fire)
    out_path1 = figuras_dir / "santarosa_pre_post_MDM_veg.png"
    plt.savefig(out_path1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figura guardada: {out_path1}")

    # Figura 2: NDVI/SAVI Pre/Post con colorbars explicativas
    if img_pre is not None:
        visualize_spectral_indices_pre_post(
            img_pre,
            img_post,
            title_prefix="Santa Rosa – "
        )
        out_path2 = figuras_dir / "santarosa_NDVI_SAVI_pre_post.png"
        plt.savefig(out_path2, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Figura guardada: {out_path2}")
    else:
        print("[WARN] No hay imagen pre-desastre; no se generaron figuras espectrales.")


def main():
    figuras_dir = ensure_output_dir()
    clf = load_trained_model()

    generate_earthquake_figures(clf, figuras_dir)
    generate_wildfire_figures(clf, figuras_dir)

    print("=== Generación de figuras completada ===")


if __name__ == "__main__":
    main()
