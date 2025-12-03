import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2

from .data_io import json_to_mask_and_polygons
from .vegetation import detect_burned_vegetation


def show_mask_and_contours(img, data):
    """Muestra la máscara de edificios y sus contornos sobre la imagen.

    Genera una figura con dos subplots:
    1. La máscara binaria de los edificios.
    2. Los contornos de los edificios dibujados sobre la imagen original.

    Parameters
    ----------
    img : np.ndarray
        Imagen sobre la cual se visualizarán los contornos.
    data : dict
        Datos de etiquetas en formato GeoJSON que contienen los polígonos.
    """
    mask, polys, _ = json_to_mask_and_polygons(data, img.shape)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara binaria de edificios")
    plt.axis("off")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)

    plt.subplot(1, 2, 2)
    plt.imshow(img_contours)
    plt.title("Contornos sobre imagen")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pre_post_MDM_with_veg(img_pre, img_post, hm_buildings):
    """Visualiza las imágenes pre y post-desastre junto con el mapa de daños.

    Crea una figura con tres subplots:
    1. Imagen pre-desastre.
    2. Imagen post-desastre.
    3. Imagen post-desastre con el mapa de daños (MDM) de edificios y la
       detección de vegetación quemada superpuestos.

    Parameters
    ----------
    img_pre : np.ndarray or None
        Imagen pre-desastre en formato RGB. Si es None, se muestra la
        imagen post-desastre en su lugar.
    img_post : np.ndarray
        Imagen post-desastre en formato RGB.
    hm_buildings : np.ndarray
        Mapa de calor (heatmap) del daño en edificios, con valores en [0, 1].
    """
    veg_burned = None
    if img_pre is not None:
        veg_burned = detect_burned_vegetation(img_pre, img_post)

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.05])
    ax  = [fig.add_subplot(gs[i]) for i in range(3)]

    # Pre
    ax[0].imshow(img_pre if img_pre is not None else img_post)
    ax[0].set_title("Pre-desastre", fontsize=16)
    ax[0].axis("off")

    # Post
    ax[1].imshow(img_post)
    ax[1].set_title("Post-desastre", fontsize=16)
    ax[1].axis("off")

    # Post + edificios + vegetación
    ax[2].imshow(img_post)
    im_build = ax[2].imshow(hm_buildings, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.6)

    if veg_burned is not None:
        ax[2].imshow(veg_burned, cmap="hot", vmin=0.0, vmax=1.0, alpha=0.7)

    ax[2].set_title("MDM edificios + vegetación quemada", fontsize=16)
    ax[2].axis("off")

    cbar = fig.colorbar(im_build, ax=ax[2], fraction=0.046, pad=0.04)
    cbar.set_label("Damage score edificios (0 = sin daño, 1 = destruido)", fontsize=12)

    ticks = np.linspace(0.0, 1.0, 5)
    labels_nivel = [
        "prob. intacto",
        "daño leve",
        "daño moderado",
        "daño severo",
        "destruido",
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}\n{lab}" for t, lab in zip(ticks, labels_nivel)])

    plt.tight_layout()
    plt.show()
