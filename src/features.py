import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

from .config import IMG_SIZE


def edge_features(gray):
    """Extrae características de bordes usando filtros Sobel y Laplaciano.

    Calcula la magnitud del gradiente con Sobel y la varianza del Laplaciano
    para capturar información sobre la presencia y fuerza de los bordes.

    Parameters
    ----------
    gray : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    np.ndarray
        Vector de características concatenando el histograma de magnitud del
        gradiente y la varianza del Laplaciano.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = mag / (mag.max() + 1e-6)

    hist, _ = np.histogram(mag, bins=16, range=(0, 1), density=True)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = np.var(lap)

    return np.concatenate([hist, [lap_var]])


def lbp_features(gray, P=8, R=1):
    """Calcula el histograma de Patrones Binarios Locales (LBP).

    LBP es un descriptor de textura que captura patrones locales.

    Parameters
    ----------
    gray : np.ndarray
        Imagen en escala de grises.
    P : int, optional
        Número de puntos en un vecindario circular.
    R : float, optional
        Radio del vecindario circular.

    Returns
    -------
    np.ndarray
        Histograma normalizado de los patrones LBP.
    """
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def haralick_features(gray):
    """Extrae características de textura de Haralick de la matriz GLCM.

    Estas características describen la textura de la imagen basándose en
    la co-ocurrencia de niveles de gris.

    Parameters
    ----------
    gray : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    np.ndarray
        Vector con las propiedades de textura de Haralick.
    """
    gray_small = cv2.resize(gray, (64, 64))
    gray_q = (gray_small / 4).astype(np.uint8)  # 0..63

    glcm = graycomatrix(
        gray_q,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    feats = [graycoprops(glcm, p).mean() for p in props]
    return np.array(feats)


def fourier_features(gray):
    """Extrae características del espectro de frecuencias de Fourier.

    Calcula la magnitud media en bandas de baja, media y alta frecuencia
    para capturar la estructura global de la imagen.

    Parameters
    ----------
    gray : np.ndarray
        Imagen en escala de grises.

    Returns
    -------
    np.ndarray
        Vector con las magnitudes medias de las bandas de frecuencia.
    """
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = np.log1p(mag)
    mag = mag / (mag.max() + 1e-6)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    r1, r2 = 10, 30
    low  = mag[R <= r1].mean()
    mid  = mag[(R > r1) & (R <= r2)].mean()
    high = mag[R > r2].mean()

    return np.array([low, mid, high])


def laplacian_pyramid_features(gray, levels=4):
    """Extrae la varianza de cada nivel de la pirámide Laplaciana.

    La pirámide Laplaciana descompone la imagen en diferentes escalas,
    y su varianza en cada nivel da información sobre la textura y detalles.

    Parameters
    ----------
    gray : np.ndarray
        Imagen en escala de grises.
    levels : int, optional
        Número de niveles de la pirámide.

    Returns
    -------
    np.ndarray
        Vector con la varianza de cada nivel de la pirámide.
    """
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    gp = [gray.astype(np.float32)]
    for _ in range(1, levels):
        gp.append(cv2.pyrDown(gp[-1]))

    lp = []
    for i in range(levels - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        up = cv2.pyrUp(gp[i + 1], dstsize=size)
        lap = gp[i] - up
        lp.append(lap)

    return np.array([np.var(l) for l in lp])


def color_features(rgb):
    """Extrae histogramas de color del espacio de color HSV.

    Parameters
    ----------
    rgb : np.ndarray
        Imagen en formato RGB.

    Returns
    -------
    np.ndarray
        Vector de características concatenando los histogramas de los
        canales H, S y V.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 180), density=True)
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=16, range=(0, 255), density=True)
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=16, range=(0, 255), density=True)
    return np.concatenate([h_hist, s_hist, v_hist])


def extract_DMS(patch_rgb):
    """Extrae un vector de características combinado (Damage Morpho-Spectral).

    Este vector DMS combina características de bordes, textura (LBP, Haralick),
    frecuencia (Fourier), escala (Pirámide Laplaciana) y color (HSV).

    Parameters
    ----------
    patch_rgb : np.ndarray
        Parche de imagen en formato RGB.

    Returns
    -------
    np.ndarray
        Vector de características DMS final.
    """
    patch_rgb = cv2.resize(patch_rgb, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)

    f_edges = edge_features(gray)
    f_lbp   = lbp_features(gray)
    f_har   = haralick_features(gray)
    f_four  = fourier_features(gray)
    f_lap   = laplacian_pyramid_features(gray)
    f_col   = color_features(patch_rgb)

    return np.concatenate([f_edges, f_lbp, f_har, f_four, f_lap, f_col])
