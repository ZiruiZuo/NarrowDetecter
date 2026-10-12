from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonMath import vtkMatrix3x3

from .image_io import LoadedVolume


def loaded_volume_to_vtk(volume: LoadedVolume) -> vtkImageData:
    array = np.ascontiguousarray(volume.array_zyx, dtype=np.float32)
    expected = tuple(reversed(volume.geometry.size_xyz))
    if array.ndim != 3 or array.shape != expected:
        raise ValueError("The volume array shape does not match its geometry.")

    image = vtkImageData()
    image.SetDimensions(*volume.geometry.size_xyz)
    image.SetSpacing(*volume.geometry.spacing_xyz)
    image.SetOrigin(*volume.geometry.origin_lps_xyz)
    direction = vtkMatrix3x3()
    matrix = volume.geometry.direction_matrix
    for row in range(3):
        for column in range(3):
            direction.SetElement(row, column, float(matrix[row, column]))
    image.SetDirectionMatrix(direction)
    image.GetPointData().SetScalars(
        numpy_to_vtk(array.ravel(order="C"), deep=True)
    )
    return image


def orthogonal_plane_axes(
    orientation: Literal["axial", "coronal", "sagittal"] | str,
) -> np.ndarray:
    axes = {
        "axial": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "coronal": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "sagittal": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    }
    try:
        return np.asarray(axes[str(orientation).lower()], dtype=float)
    except KeyError as error:
        raise ValueError(f"Unsupported orientation: {orientation}") from error


def _display_to_world(renderer, x: float, y: float, depth: float) -> np.ndarray:
    renderer.SetDisplayPoint(float(x), float(y), float(depth))
    renderer.DisplayToWorld()
    point = np.asarray(renderer.GetWorldPoint(), dtype=float)
    if point.shape != (4,) or not np.all(np.isfinite(point)) or point[3] == 0:
        raise ValueError("Unable to project the display position into physical space.")
    return point[:3] / point[3]


def display_position_to_lps(
    renderer,
    display_xy: Sequence[float],
    plane_origin_lps: Sequence[float],
    plane_normal_lps: Sequence[float],
) -> tuple[float, float, float]:
    display = np.asarray(display_xy, dtype=float)
    origin = np.asarray(plane_origin_lps, dtype=float)
    normal = np.asarray(plane_normal_lps, dtype=float)
    if display.shape != (2,) or origin.shape != (3,) or normal.shape != (3,):
        raise ValueError("Display position, plane origin, and normal have invalid shapes.")
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Plane normal must be finite and nonzero.")
    normal /= length
    near = _display_to_world(renderer, display[0], display[1], 0.0)
    far = _display_to_world(renderer, display[0], display[1], 1.0)
    ray = far - near
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-12:
        raise ValueError("The display ray is parallel to the slice plane.")
    scale = float(np.dot(normal, origin - near) / denominator)
    point = near + scale * ray
    return tuple(float(value) for value in point)

