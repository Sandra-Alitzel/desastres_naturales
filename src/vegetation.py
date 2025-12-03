import cv2
import numpy as np


def detect_burned_vegetation(img_pre, img_post):
    """Detecta vegetación quemada comparando imágenes pre y post desastre.

    Identifica áreas que eran vegetación en la imagen pre-desastre y que
    muestran una reducción significativa en el canal verde en la imagen
    post-desastre, lo que es un indicador de quema.

    Parameters
    ----------
    img_pre : np.ndarray
        Imagen pre-desastre en formato RGB.
    img_post : np.ndarray
        Imagen post-desastre en formato RGB.

    Returns
    -------
    np.ndarray
        Mapa de puntuación de vegetación quemada, con valores en el rango
        [0.0, 1.0], donde 1.0 indica una alta probabilidad de quema.
        Las áreas no consideradas vegetación en la imagen pre-desastre
        tienen una puntuación de 0.
    """
    pre_hsv = cv2.cvtColor(img_pre, cv2.COLOR_RGB2HSV)

    lower_green = np.array([30, 30, 30], dtype=np.uint8)
    upper_green = np.array([100, 255, 255], dtype=np.uint8)

    pre_veg_mask = cv2.inRange(pre_hsv, lower_green, upper_green)
    pre_veg_bool = pre_veg_mask > 0

    g_pre  = img_pre[..., 1].astype(np.float32)
    g_post = img_post[..., 1].astype(np.float32)

    delta_g = g_pre - g_post
    delta_g = np.clip(delta_g, 0, None)

    burned_score = delta_g / 80.0
    burned_score = np.clip(burned_score, 0.0, 1.0)

    burned_score[~pre_veg_bool] = 0.0
    burned_score = cv2.GaussianBlur(burned_score, (7, 7), 0)

    return burned_score
