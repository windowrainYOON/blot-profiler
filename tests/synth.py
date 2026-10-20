"""Synthetic chemiluminescence blot + white-light marker photo (imager style)."""

import numpy as np


def chemi_blot(H=1024, W=1408, seed=0):
    """Bright bands on a black background, uint16 (like a CCD imager export).

    Rows: Vinculin (faint, y~180), Erk doublet (y~360/385), Kras (strong, y~590,
    empty first lane); 8 lanes; ladder lanes left of x~150 are blank here.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 300.0)
    lanes = 260 + np.arange(8) * 135
    vin = [0.5, 0.55, 0.5, 0.6, 0.5, 0.55, 0.5, 0.6]
    kras = [0.0, 1.0, 1.1, 1.0, 1.3, 1.2, 0.9, 0.8]
    for i, cx in enumerate(lanes):
        prof = np.exp(-(((xx - cx) / 48.0) ** 6))
        img += 3000 * vin[i] * prof * np.exp(-((yy - 180) / 7.0) ** 2)
        img += 9000 * prof * np.exp(-((yy - 360) / 8.0) ** 2)
        img += 12000 * prof * np.exp(-((yy - 385) / 8.0) ** 2)
        img += 40000 * kras[i] * prof * np.exp(-((yy - 590) / 11.0) ** 2)
    img += rng.normal(0, 60, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16), lanes


LADDER_Y = {250: 70, 130: 130, 100: 165, 70: 230, 55: 300, 40: 390, 35: 430, 25: 540, 15: 680, 10: 760}


def marker_photo(H=1024, W=1408, seed=1):
    """White-light membrane photo: grey membrane, dark prestained ladder at the left."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 52000.0)
    img[:, :40] = 20000  # outside the membrane
    for y in LADDER_Y.values():
        img -= 25000 * np.exp(-(((xx - 95) / 30.0) ** 6)) * np.exp(-((yy - y) / 5.0) ** 2)
    img += rng.normal(0, 300, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)
