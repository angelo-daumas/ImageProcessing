import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union, Sequence

Matrix = NDArray[np.float64]
ArrayPoint = NDArray[np.float64]
Vector = NDArray[np.float64]
PointLike = Union[ArrayPoint, Tuple[float,float]]

class AffineTransform:
    """
    A class that represents a 2D affine transformation.

    Args:
        matrix (3x3 Matrix): The matrix that represents the transformation mathematically.
    
    Attributes:
        matrix (3x3 Matrix): The matrix that represents the transformation mathematically.
    """
    matrix:Matrix

    def __init__(self, matrix:Matrix) -> None:
        self.matrix = np.array(matrix)  # type: ignore
        pass

    def apply(self, point:PointLike) -> PointLike:
        """Applies the transformation to a 2D point, and returns the resulting 2D point."""
        return tuple((self.matrix @ [*point, 1.])[0:2])  # type: ignore

    def applyAll(self, points:Sequence[PointLike]) -> NDArray[np.float64]:
        """Applies the transformation to a sequence of 2D points, using NumPy to speed up computation."""
        homogeneous:NDArray[np.float64]  = np.hstack([points, np.ones([len(points), 1])])  # type: ignore
        return (homogeneous @ self.matrix.T)[:,:2]

def get_transformed_bounds(img, transform):
    """
    Returns the transformed bounds of an image, after applying the given affine transformation.

    Two points are returned: the new origin of the image and the new diagonal. Their coordinates are relative to the original origin point.
    """
    maxX, maxY = img.shape[0:2] - np.array(1)

    x_coords,y_coords = transform.applyAll([[0, 0], [maxX, 0], [maxX, maxY], [0, maxY]]).T
    x_coords = np.sort(x_coords)
    y_coords = np.sort(y_coords)

    return np.array([x_coords[0], y_coords[0]]), np.array([x_coords[-1], y_coords[-1]])

def average_nearest_pixels(img, point):
    """Returns the averaged values of a point's neighbouring pixels in an image."""
    x0, y0 = np.floor(point).astype(int)

    neighbors = img[x0:x0+2, y0:y0+2]

    x, y = point
    wx = np.array([[x0+1 - x], [x - x0]])
    wy = np.array([[y0+1 - y], [y - y0]])
    weights = wx @ wy.T

    # Einsum for elementwise multiplication, broadcast to higher dimentions (for RGB images)
    # Equivalent to (weights * neighbors), but works even if neighbors is 3D or higher.
    return np.sum(np.einsum('ij,ij...->ij...', weights, neighbors))

def bilinear_interp(img, transform):
    """Performs billinear interpolation on an image, using inverse mapping."""
    inverse = AffineTransform(np.linalg.inv(transform.matrix))
    maxX, maxY = img.shape[0:2] - np.array(1)

    new_origin, new_diagonal = get_transformed_bounds(img, transform)
    width, height = np.round(new_diagonal - new_origin).astype(int)

    shape = list(img.shape)
    shape[0:2] = width, height
    result = np.ones(shape)*255
    for r in range(width):
        for c in range(height):
            x,y = point = inverse.apply(new_origin + [r, c])

            if x >= 0 and x < maxX and y >= 0 and y < maxY:
                result[r,c] = average_nearest_pixels(img, point)

    return result
