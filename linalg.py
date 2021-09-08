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
