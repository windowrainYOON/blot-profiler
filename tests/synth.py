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


GEL_LANES = 12


def gel_photo(H=1024, W=1280, seed=2):
    """Agarose gel photo, already inverted (dark bands on light grey), as RGB uint8.

    The gel sits in a clear tray on a white background; 12 wells along the top;
    lanes 1 and 5 are ladders (10 bands), lanes 2-4 and 6-8 carry one strong
    band (lanes 3 and 7 a bit higher plus a faint band below), lanes 9-12 are empty.
    Returns (rgb, lane x centres, ladder band ys).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 245.0)
    gx0, gx1, gy0, gy1 = int(0.17 * W), int(0.83 * W), int(0.28 * H), int(0.73 * H)
    img[gy0 - 30:gy1 + 30, gx0 - 30:gx1 + 30] = 225.0  # tray
    img[gy0:gy1, gx0:gx1] = 200.0  # gel
    lanes = gx0 + 40 + np.arange(GEL_LANES) * (gx1 - gx0 - 80) / (GEL_LANES - 1)
    for cx in lanes:  # wells
        img[(yy > gy0 + 12) & (yy < gy0 + 24) & (np.abs(xx - cx) < 16)] -= 40
    ladder_y = gy0 + 50 + np.cumsum([0, 22, 22, 24, 26, 28, 32, 36, 42, 50])
    dark = np.zeros((H, W))
    for li, cx in enumerate(lanes):
        prof = np.exp(-(((xx - cx) / 14.0) ** 6))
        if li in (0, 4):
            for y in ladder_y:
                dark += 70 * prof * np.exp(-((yy - y) / 3.0) ** 2)
        elif li in (1, 2, 3, 5, 6, 7):
            y = gy0 + 200 - (12 if li in (2, 6) else 0)
            dark += 110 * prof * np.exp(-((yy - y) / 5.0) ** 2)
            if li in (2, 6):
                dark += 35 * prof * np.exp(-((yy - (y + 45)) / 3.5) ** 2)
    img -= dark
    img += rng.normal(0, 2.0, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb, lanes, ladder_y
