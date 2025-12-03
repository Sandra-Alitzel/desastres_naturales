## `streamlit_app.py`


import sys
from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load

# ------------------------------------------
# Asegurar acceso a src/ desde cualquier lugar
# ------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[0]
if not (ROOT_DIR / "src").exists():
    ROOT_DIR = ROOT_DIR.parent
sys.path.append(str(ROOT_DIR))

from src.config import EVENTS
from src.data_io import list_event_files, get_label_path, load_pre_image_from_post_path
from src.mdm import create_MDM_for_image
from src.spectral import compute_spectral_indices
from src.vegetation import detect_burned_vegetation


# ==========================================
# CACHES
# ==========================================


@st.cache_resource
def load_trained_model():
    model_path = ROOT_DIR / "models" / "damage_clf.pkl"
    if not model_path.exists():
        st.error(
            f"No se encontró el modelo en {model_path}. "
            "Ejecuta primero: `python scripts/train_model.py`."
        )
        return None
    clf = load(model_path)
    return clf


@st.cache_data
def list_event_files_cached(split: str, event: str):
    try:
        files = list_event_files(split, event)
        return [str(f) for f in files]
    except Exception as e:
        st.error(f"Error listando archivos: {e}")
        return []


@st.cache_data
def load_image_and_json(path_str: str, split: str):
    img_path = Path(path_str)
    label_path = get_label_path(split, img_path)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"No pude leer la imagen: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with open(label_path) as f:
        data = json.load(f)

    img_pre = load_pre_image_from_post_path(img_path)
    return img_rgb, data, img_pre, img_path, label_path


# ==========================================
# FUNCIONES DE PLOTEO (FIGURAS PARA STREAMLIT)
# ==========================================


def fig_pre_post_mdm_veg(img_pre, img_post, hm_buildings):
    veg_burned = None
    if img_pre is not None:
        veg_burned = detect_burned_vegetation(img_pre, img_post)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Pre
    ax1.imshow(img_pre if img_pre is not None else img_post)
    ax1.set_title("Pre-desastre")
    ax1.axis("off")

    # Post
    ax2.imshow(img_post)
    ax2.set_title("Post-desastre")
    ax2.axis("off")

    # Post + MDM + veg
    ax3.imshow(img_post)
    im_build = ax3.imshow(hm_buildings, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.6)
    if veg_burned is not None:
        ax3.imshow(veg_burned, cmap="hot", vmin=0.0, vmax=1.0, alpha=0.6)
    ax3.set_title("MDM edificios + vegetación")
    ax3.axis("off")

    cbar = fig.colorbar(im_build, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Damage score (0=intacto, 1=destruido)", fontsize=8)

    return fig


def fig_ndvi_savi_pre_post(img_pre, img_post):
    NDVI_pre, SAVI_pre = compute_spectral_indices(img_pre)
    NDVI_post, SAVI_post = compute_spectral_indices(img_post)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    vmin, vmax = -0.5, 1.0

    ticks_ndvi = [-0.3, 0.0, 0.3, 0.6, 0.9]
    ticks_savi = [-0.3, 0.0, 0.3, 0.6, 0.9]

    # NDVI Pre
    im0 = axes[0, 0].imshow(NDVI_pre, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("NDVI Pre")
    axes[0, 0].axis("off")
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar0.set_ticks(ticks_ndvi)
    cbar0.set_ticklabels(
        [
            "-0.3\nagua/suelo",
            "0.0\nsin vegetación",
            "0.3\nveg. escasa",
            "0.6\nveg. moderada",
            "0.9\nveg. densa",
        ]
    )
    cbar0.set_label("NDVI", fontsize=8)

    # SAVI Pre
    im1 = axes[0, 1].imshow(SAVI_pre, cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("SAVI Pre")
    axes[0, 1].axis("off")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_ticks(ticks_savi)
    cbar1.set_ticklabels(
        [
            "-0.3\nsuelo desnudo",
            "0.0\nbaja cobertura",
            "0.3\nveg. baja",
            "0.6\nveg. media",
            "0.9\nveg. alta",
        ]
    )
    cbar1.set_label("SAVI", fontsize=8)

    # NDVI Post
    im2 = axes[1, 0].imshow(NDVI_post, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("NDVI Post")
    axes[1, 0].axis("off")
    cbar2 = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar2.set_ticks(ticks_ndvi)
    cbar2.set_ticklabels(
        [
            "-0.3\nagua/suelo",
            "0.0\nsin vegetación",
            "0.3\nveg. escasa",
            "0.6\nveg. moderada",
            "0.9\nveg. densa",
        ]
    )
    cbar2.set_label("NDVI", fontsize=8)

    # SAVI Post
    im3 = axes[1, 1].imshow(SAVI_post, cmap="magma", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("SAVI Post")
    axes[1, 1].axis("off")
    cbar3 = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar3.set_ticks(ticks_savi)
    cbar3.set_ticklabels(
        [
            "-0.3\nsuelo desnudo",
            "0.0\nbaja cobertura",
            "0.3\nveg. baja",
            "0.6\nveg. media",
            "0.9\nveg. alta",
        ]
    )
    cbar3.set_label("SAVI", fontsize=8)

    plt.tight_layout()
    return fig


# ==========================================
# APP
# ==========================================


def main():
    st.set_page_config(page_title="DamageLens xBD", layout="wide", page_icon="🛰")
    st.title(":material/satellite_alt: DamageLens – Multi-Scale Damage Mapping on xBD")

    st.sidebar.header("Configuración")

    # Selección de evento y split
    event = st.sidebar.selectbox("Evento", options=EVENTS, index=0)
    split = st.sidebar.selectbox("Split", options=["train", "test", "hold"], index=2)

    # Cargar archivos disponibles
    file_list = list_event_files_cached(split, event)
    if not file_list:
        st.warning("No se encontraron imágenes para esta combinación.")
        return

    # Seleccionar índice de imagen
    idx = st.sidebar.slider("Índice de imagen", 0, len(file_list) - 1, 0)
    st.sidebar.write(f":material/image: \n`{Path(file_list[idx]).name}`")

    st.sidebar.markdown("""
    ## Equipo
    
    - Sandra Alitzel Vázquez Chávez
    - Diego A. Barriga Martínez
    - David Alexis Duran Ruiz
    - Tlacaelel Jaime Flores Villaseñor
    """)

    # Cargar modelo
    clf = load_trained_model()
    if clf is None:
        return

    # Cargar imagen + JSON + pre
    try:
        img_post, data_post, img_pre, img_path, lbl_path = load_image_and_json(
            file_list[idx], split
        )
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return

    # Crear MDM para esa imagen
    overlay, hm = create_MDM_for_image(img_post, data_post, clf)

    # Tabs de visualización
    tab1, tab2, tab3 = st.tabs(["Pre/Post + MDM", "Overlay MDM solo", "NDVI/SAVI"])

    with tab1:
        st.subheader("Pre / Post / MDM + Vegetación quemada")
        fig1 = fig_pre_post_mdm_veg(img_pre, img_post, hm)
        st.pyplot(fig1)

    with tab2:
        st.subheader("Overlay de MDM sobre imagen Post")
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(img_post)
        ax1.set_title("Post-desastre")
        ax1.axis("off")
        ax2.imshow(overlay)
        ax2.set_title("MDM edificios")
        ax2.axis("off")
        plt.tight_layout()
        st.pyplot(fig2)

    with tab3:
        if img_pre is None:
            st.warning("No hay imagen pre-desastre para calcular índices espectrales.")
        else:
            st.subheader("Índices espectrales simulados (NDVI / SAVI)")
            st.caption(
                "NDVI y SAVI se calculan usando el canal verde como proxy de NIR. "
                "Las barras de color interpretan los rangos: suelo, vegetación escasa, "
                "vegetación densa, etc."
            )
            fig3 = fig_ndvi_savi_pre_post(img_pre, img_post)
            st.pyplot(fig3)


if __name__ == "__main__":
    main()
