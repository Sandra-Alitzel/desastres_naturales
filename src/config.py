# src/config.py
from pathlib import Path

# Directorio raíz del repo: .../tu_proyecto/
ROOT_DIR = Path(__file__).resolve().parents[1]

# Carpeta de datos (no hardcode a tu máquina, solo /Datos/)
DATA_DIR = ROOT_DIR / "Datos"

# Subcarpetas estándar en xBD
SPLITS = ["train", "test", "hold"]
EVENTS = ["mexico-earthquake", "santa-rosa-wildfire"]

# Tamaño base para parches de edificios
IMG_SIZE = 256

# Mapeo de niveles de daño
DAMAGE_MAP = {
    "no-damage":     0,
    "minor-damage":  1,
    "major-damage":  2,
    "destroyed":     3,
}
INV_DAMAGE_MAP = {v: k for k, v in DAMAGE_MAP.items()}