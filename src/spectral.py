# src/spectral.py
import numpy as np
import matplotlib.pyplot as plt


def compute_spectral_indices(img):
    img_f = img.astype(np.float32) / 255.0

    R = img_f[..., 0]
    G = img_f[..., 1]
    B = img_f[..., 2]

    NIR = G

    NDVI = (NIR - R) / (NIR + R + 1e-6)
    NDVI = np.clip(NDVI, -1.0, 1.0)

    L = 0.5
    SAVI = ((NIR - R) / (NIR + R + L + 1e-6)) * (1.0 + L)
    SAVI = np.clip(SAVI, -1.0, 1.0)

    return NDVI, SAVI


def visualize_spectral_indices_pre_post(img_pre, img_post, title_prefix=""):
    NDVI_pre, SAVI_pre   = compute_spectral_indices(img_pre)
    NDVI_post, SAVI_post = compute_spectral_indices(img_post)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    vmin, vmax = -0.5, 1.0

    # NDVI Pre
    im0 = axes[0, 0].imshow(NDVI_pre, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"{title_prefix}NDVI Pre", fontsize=14)
    axes[0, 0].axis("off")
    cbar0 = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    ticks_ndvi = [-0.3, 0.0, 0.3, 0.6, 0.9]
    cbar0.set_ticks(ticks_ndvi)
    cbar0.set_ticklabels([
        "-0.3\nagua/suelo",
        "0.0\nsin vegetación",
        "0.3\nvegetación escasa",
        "0.6\nvegetación moderada",
        "0.9\nvegetación densa",
    ])
    cbar0.set_label("NDVI", fontsize=10)

    # SAVI Pre
    im1 = axes[0, 1].imshow(SAVI_pre, cmap="magma", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"{title_prefix}SAVI Pre", fontsize=14)
    axes[0, 1].axis("off")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    ticks_savi = [-0.3, 0.0, 0.3, 0.6, 0.9]
    cbar1.set_ticks(ticks_savi)
    cbar1.set_ticklabels([
        "-0.3\nsuelo desnudo",
        "0.0\nmuy baja cobertura",
        "0.3\nvegetación baja",
        "0.6\nvegetación media",
        "0.9\nvegetación alta",
    ])
    cbar1.set_label("SAVI", fontsize=10)

    # NDVI Post
    im2 = axes[1, 0].imshow(NDVI_post, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"{title_prefix}NDVI Post", fontsize=14)
    axes[1, 0].axis("off")
    cbar2 = fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar2.set_ticks(ticks_ndvi)
    cbar2.set_ticklabels([
        "-0.3\nagua/suelo",
        "0.0\nsin vegetación",
        "0.3\nvegetación escasa",
        "0.6\nvegetación moderada",
        "0.9\nvegetación densa",
    ])
    cbar2.set_label("NDVI", fontsize=10)

    # SAVI Post
    im3 = axes[1, 1].imshow(SAVI_post, cmap="magma", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"{title_prefix}SAVI Post", fontsize=14)
    axes[1, 1].axis("off")
    cbar3 = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar3.set_ticks(ticks_savi)
    cbar3.set_ticklabels([
        "-0.3\nsuelo desnudo",
        "0.0\nmuy baja cobertura",
        "0.3\nvegetación baja",
        "0.6\nvegetación media",
        "0.9\nvegetación alta",
    ])
    cbar3.set_label("SAVI", fontsize=10)

    plt.tight_layout()
    plt.show()
