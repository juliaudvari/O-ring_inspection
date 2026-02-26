from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationResult:
    status: str
    reason: str
    continuity: float
    thickness_median: float
    thickness_mad: float
    outlier_fraction: float


def _compute_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(cols.mean())


def _sample_ray_thicknesses(mask: np.ndarray, centroid_rc: tuple[float, float], n_angles: int):
    height, width = mask.shape
    max_radius = int(np.ceil(np.hypot(height, width)))
    radii = np.arange(max_radius, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_angles, endpoint=False)

    hits = np.zeros(n_angles, dtype=bool)
    thicknesses = np.full(n_angles, np.nan, dtype=np.float64)

    center_row, center_col = centroid_rc
    for idx, angle in enumerate(angles):
        ray_rows = np.rint(center_row + radii * np.sin(angle)).astype(np.int32)
        ray_cols = np.rint(center_col + radii * np.cos(angle)).astype(np.int32)
        valid = (
            (ray_rows >= 0)
            & (ray_rows < height)
            & (ray_cols >= 0)
            & (ray_cols < width)
        )
        if not np.any(valid):
            continue
        ray_rows = ray_rows[valid]
        ray_cols = ray_cols[valid]

        values = mask[ray_rows, ray_cols]
        fg_idx = np.flatnonzero(values)
        if fg_idx.size == 0:
            continue

        hits[idx] = True
        thicknesses[idx] = float(fg_idx[-1] - fg_idx[0] + 1)

    return thicknesses, hits


def classify_ring(
    mask: np.ndarray,
    n_angles: int = 360,
    continuity_threshold: float = 0.98,
    outlier_fraction_threshold: float = 0.10,
    relative_mad_threshold: float = 0.18,
) -> ClassificationResult:
    centroid = _compute_centroid(mask)
    if centroid is None:
        return ClassificationResult("FAIL", "no foreground", 0.0, 0.0, 0.0, 1.0)

    thicknesses, hits = _sample_ray_thicknesses(mask, centroid, n_angles=n_angles)
    valid = thicknesses[hits]
    continuity = float(np.mean(hits))
    if valid.size == 0:
        return ClassificationResult("FAIL", "no ray hits", continuity, 0.0, 0.0, 1.0)

    thickness_median = float(np.median(valid))
    thickness_mad = float(np.median(np.abs(valid - thickness_median)))
    if thickness_mad < 1e-8:
        outlier_fraction = float(np.mean(np.abs(valid - thickness_median) / (thickness_median + 1e-8) > 0.15))
    else:
        robust_sigma = 1.4826 * thickness_mad
        outlier_fraction = float(np.mean(np.abs(valid - thickness_median) / robust_sigma > 3.5))
    relative_mad = thickness_mad / (thickness_median + 1e-8)

    fail = []
    if continuity < continuity_threshold:
        fail.append("low continuity")
    if outlier_fraction > outlier_fraction_threshold:
        fail.append("many outliers")
    if relative_mad > relative_mad_threshold:
        fail.append("high mad")

    if fail:
        return ClassificationResult(
            "FAIL",
            "; ".join(fail),
            continuity,
            thickness_median,
            thickness_mad,
            outlier_fraction,
        )

    return ClassificationResult(
        "PASS",
        "stable thickness",
        continuity,
        thickness_median,
        thickness_mad,
        outlier_fraction,
    )
