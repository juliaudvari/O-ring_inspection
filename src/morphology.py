from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _validate_kernel(ksize: int, iterations: int) -> None:
    # basic arg checks
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer >= 1.")
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")


def binary_dilation(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    _validate_kernel(ksize, iterations)
    output = mask.astype(bool, copy=False)
    pad = ksize // 2

    for _ in range(iterations):
        # grow fg area
        padded = np.pad(output, pad_width=pad, mode="constant", constant_values=False)
        windows = sliding_window_view(padded, (ksize, ksize))
        output = np.any(windows, axis=(-2, -1))
    return output


def binary_erosion(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    _validate_kernel(ksize, iterations)
    output = mask.astype(bool, copy=False)
    pad = ksize // 2

    for _ in range(iterations):
        # shrink fg area
        padded = np.pad(output, pad_width=pad, mode="constant", constant_values=False)
        windows = sliding_window_view(padded, (ksize, ksize))
        output = np.all(windows, axis=(-2, -1))
    return output


def binary_closing(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    # fill small holes
    dilated = binary_dilation(mask, ksize=ksize, iterations=iterations)
    return binary_erosion(dilated, ksize=ksize, iterations=iterations)
