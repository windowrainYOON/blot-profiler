"""Draw the app icon (a stylised blot) as a 1024x1024 PNG: python packaging/make_icon.py out.png"""

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((64, 64, S - 64, S - 64), radius=200, fill=(245, 245, 242, 255), outline=(60, 60, 70, 255), width=18)
bands = Image.new("L", (S, S), 0)
bd = ImageDraw.Draw(bands)
rows = [(330, 1.0, (0.9, 1.0, 0.95, 1.0)), (520, 1.0, (0.2, 0.6, 1.0, 0.8)), (710, 1.3, (0.0, 0.9, 0.7, 1.0))]
for y, thick, ints in rows:
    for i, v in enumerate(ints):
        x = int(512 + (i - 1.5) * 185)
        if v > 0:
            bd.ellipse((x - 70, y - 22 * thick, x + 70, y + 22 * thick), fill=int(230 * v))
bands = bands.filter(ImageFilter.GaussianBlur(10))
dark = Image.new("RGBA", (S, S), (25, 25, 35, 255))
img.paste(dark, (0, 0), bands)
for y, thick, _ in rows[1:2]:
    for i in range(4):
        x = int(512 + (i - 1.5) * 185)
        d.rectangle((x - 80, y - 45, x + 80, y + 45), outline=(220, 40, 60, 255), width=12)
img.save(sys.argv[1] if len(sys.argv) > 1 else "icon.png")
