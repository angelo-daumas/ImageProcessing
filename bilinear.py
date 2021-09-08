import numpy as np
from linalg import AffineTransform

def get_transformed_bounds(img, transform):
    maxX, maxY = img.shape[0:2] - np.array(1)

    x_coords,y_coords = transform.applyAll([[0, 0], [maxX, 0], [maxX, maxY], [0, maxY]]).T
    x_coords = np.sort(x_coords)
    y_coords = np.sort(y_coords)

    return np.array([x_coords[0], y_coords[0]]), np.array([x_coords[-1], y_coords[-1]])

def average_nearest_pixels(img, point):
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
