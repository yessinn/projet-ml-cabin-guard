"""Histogrames de gradients sur une image entière, sans annotation en entrée."""

import numpy as np
from PIL import Image, ImageOps

SIZE = 128
CELL = 8
BINS = 9
EPS = 1e-6


def hog_image(path):
    """Retourner 8 100 coefficients à partir des pixels de l’image."""
    with Image.open(path) as source, ImageOps.exif_transpose(source) as oriented:
        gray = ImageOps.contain(oriented.convert("L"), (SIZE, SIZE), Image.Resampling.BILINEAR)
    width, height = gray.size
    left, top = (SIZE - width) // 2, (SIZE - height) // 2
    pixels = np.zeros((SIZE, SIZE), dtype=np.float32)
    pixels[top:top + height, left:left + width] = np.asarray(gray, dtype=np.float32) / 255
    dy, dx = np.gradient(pixels)
    magnitudes = np.hypot(dx, dy)
    valid = np.zeros_like(pixels, dtype=bool)
    valid[top + 1:top + height - 1, left + 1:left + width - 1] = True
    magnitudes[~valid] = 0

    angle = np.mod(np.arctan2(dy, dx), np.pi)
    orientation = np.minimum((angle * (BINS / np.pi)).astype(np.intp), BINS - 1)
    cells_per_side = SIZE // CELL
    cell_axis = np.arange(SIZE) // CELL
    cell_index = cell_axis[:, None] * cells_per_side + cell_axis[None, :]
    indices = cell_index * BINS + orientation
    histograms = np.bincount(indices.ravel(), weights=magnitudes.ravel(),
                             minlength=cells_per_side**2 * BINS).reshape(cells_per_side, cells_per_side, BINS)

    blocks = np.lib.stride_tricks.sliding_window_view(histograms, (2, 2), axis=(0, 1))
    blocks = blocks.transpose(0, 1, 3, 4, 2).reshape(-1, 4 * BINS)
    normalized = blocks / np.sqrt(np.sum(blocks**2, axis=1, keepdims=True) + EPS**2)
    normalized = np.minimum(normalized, 0.2)
    normalized /= np.sqrt(np.sum(normalized**2, axis=1, keepdims=True) + EPS**2)
    return normalized.ravel().astype(np.float32)
