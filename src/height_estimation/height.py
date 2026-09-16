from math import isfinite


def calculate_height_cm(height_px: float, cm_per_pixel: float) -> float:
    return height_px * cm_per_pixel


def calculate_perspective_height_cm(
    head: tuple[float, float],
    feet: tuple[float, float],
) -> float:
    if not all(
        isfinite(value)
        for point in (head, feet)
        for value in point
    ):
        raise ValueError("perspective endpoints must have finite coordinates")

    if head[1] == feet[1]:
        raise ValueError("perspective endpoints must have different heights")

    return abs(feet[1] - head[1])


def validate_perspective_result(
    head: tuple[float, float],
    feet: tuple[float, float],
    reprojection_error_cm: float,
    max_reprojection_error_cm: float = 2.0,
) -> None:
    calculate_perspective_height_cm(head, feet)

    if not isfinite(reprojection_error_cm):
        raise ValueError("perspective reprojection error must be finite")
    if reprojection_error_cm < 0:
        raise ValueError("perspective reprojection error cannot be negative")
    if reprojection_error_cm > max_reprojection_error_cm:
        raise ValueError("perspective reprojection error is too high")
