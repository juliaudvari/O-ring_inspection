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
    max_angular_gap: float
    centroid_rc: tuple[float, float]


def _compute_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    # center of fg pixels
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(cols.mean())


def _sample_ray_thicknesses(
    mask: np.ndarray, centroid_rc: tuple[float, float], n_angles: int
) -> tuple[np.ndarray, np.ndarray]:
    # set ray grid
    height, width = mask.shape
    max_radius = int(np.ceil(np.hypot(height, width)))
    radii = np.arange(max_radius, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, num=n_angles, endpoint=False)

    hits = np.zeros(n_angles, dtype=bool)
    thicknesses = np.full(n_angles, np.nan, dtype=np.float64)

    center_row, center_col = centroid_rc

    # trace each ray
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

        if ray_rows.size > 1:
            deltas = np.diff(np.column_stack((ray_rows, ray_cols)), axis=0)
            keep = np.concatenate(([True], np.any(deltas != 0, axis=1)))
            ray_rows = ray_rows[keep]
            ray_cols = ray_cols[keep]

        values = mask[ray_rows, ray_cols]
        foreground_indices = np.flatnonzero(values)
        if foreground_indices.size == 0:
            continue

        # first to last hit
        hits[idx] = True
        first = foreground_indices[0]
        last = foreground_indices[-1]
        thicknesses[idx] = float(last - first + 1)

    return thicknesses, hits


def _max_angular_gap(hits: np.ndarray, n_angles: int) -> float:
    # longest run of no-hit angles in degrees
    if not np.any(~hits):
        return 0.0
    deg_per_angle = 360.0 / n_angles
    # wrap treat as circular
    ext = np.concatenate([hits, hits])
    run = 0
    max_run = 0
    for v in ext:
        if not v:
            run += 1
        else:
            max_run = max(max_run, run)
            run = 0
    max_run = max(max_run, run)
    return min(max_run * deg_per_angle, 360.0)


def _compute_outlier_fraction(valid_thicknesses: np.ndarray, median: float, mad: float) -> float:
    # no data edge case
    if valid_thicknesses.size == 0:
        return 1.0
    if median <= 0:
        return 1.0

    # near flat case
    if mad < 1e-8:
        relative_error = np.abs(valid_thicknesses - median) / median
        return float(np.mean(relative_error > 0.15))

    # robust outlier test
    robust_sigma = 1.4826 * mad
    robust_z = np.abs(valid_thicknesses - median) / robust_sigma
    return float(np.mean(robust_z > 3.5))


def classify_ring(
    mask: np.ndarray,
    n_angles: int = 360,
    continuity_threshold: float = 0.99,
    outlier_fraction_threshold: float = 0.10,
    relative_mad_threshold: float = 0.18,
    max_gap_threshold: float = 12.0,
    all_component_stats: list | None = None,
) -> ClassificationResult:
    # two arcs = broken ring
    if all_component_stats and len(all_component_stats) >= 2:
        areas = sorted((int(s["area"]) for s in all_component_stats), reverse=True)
        if areas[1] > 0.08 * areas[0]:
            return ClassificationResult(
                status="FAIL",
                reason="ring split into two arcs (gap detected)",
                continuity=0.0,
                thickness_median=0.0,
                thickness_mad=0.0,
                outlier_fraction=1.0,
                max_angular_gap=360.0,
                centroid_rc=(0.0, 0.0),
            )

    # get ring center
    centroid = _compute_centroid(mask)
    if centroid is None:
        return ClassificationResult(
            status="FAIL",
            reason="no foreground pixels in largest component",
            continuity=0.0,
            thickness_median=0.0,
            thickness_mad=0.0,
            outlier_fraction=1.0,
            max_angular_gap=360.0,
            centroid_rc=(0.0, 0.0),
        )

    # get thickness stats
    thicknesses, hits = _sample_ray_thicknesses(mask, centroid, n_angles=n_angles)
    valid_thicknesses = thicknesses[hits]
    continuity = float(np.mean(hits))
    max_gap = _max_angular_gap(hits, n_angles)

    if valid_thicknesses.size == 0:
        return ClassificationResult(
            status="FAIL",
            reason="no radial intersections with component",
            continuity=continuity,
            thickness_median=0.0,
            thickness_mad=0.0,
            outlier_fraction=1.0,
            max_angular_gap=max_gap,
            centroid_rc=centroid,
        )

    # build features
    thickness_median = float(np.median(valid_thicknesses))
    thickness_mad = float(np.median(np.abs(valid_thicknesses - thickness_median)))
    outlier_fraction = _compute_outlier_fraction(
        valid_thicknesses, thickness_median, thickness_mad
    )
    relative_mad = thickness_mad / (thickness_median + 1e-8)
    min_thickness = float(np.min(valid_thicknesses))
    # rule checks
    fail_reasons: list[str] = []
    if continuity < continuity_threshold:
        fail_reasons.append(
            f"continuity too low ({continuity:.3f} < {continuity_threshold:.3f})"
        )
    if max_gap > max_gap_threshold:
        fail_reasons.append(
            f"gap in ring ({max_gap:.1f} deg > {max_gap_threshold:.1f} deg)"
        )
    if thickness_median > 1e-6 and min_thickness < 0.5 * thickness_median:
        fail_reasons.append(
            f"thin spot (min={min_thickness:.1f} << med={thickness_median:.1f})"
        )
    if outlier_fraction > outlier_fraction_threshold:
        fail_reasons.append(
            "too many thickness outliers "
            f"({outlier_fraction:.3f} > {outlier_fraction_threshold:.3f})"
        )
    if relative_mad > relative_mad_threshold:
        fail_reasons.append(
            f"thickness variability too high ({relative_mad:.3f} > {relative_mad_threshold:.3f})"
        )

    # final class
    if fail_reasons:
        return ClassificationResult(
            status="FAIL",
            reason="; ".join(fail_reasons),
            continuity=continuity,
            thickness_median=thickness_median,
            thickness_mad=thickness_mad,
            outlier_fraction=outlier_fraction,
            max_angular_gap=max_gap,
            centroid_rc=centroid,
        )

    return ClassificationResult(
        status="PASS",
        reason="high continuity and stable thickness profile",
        continuity=continuity,
        thickness_median=thickness_median,
        thickness_mad=thickness_mad,
        outlier_fraction=outlier_fraction,
        max_angular_gap=max_gap,
        centroid_rc=centroid,
    )
