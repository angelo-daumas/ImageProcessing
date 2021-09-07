import numpy as np
from linalg import AffineTransform
from kernels import boxfilter

original = []
transform = AffineTransform([]])  # type: ignore
transform.matrix = np.linalg.inv(transform.matrix)  # type: ignore
points = transform.applyAll(original)

newmatrix = np.zeros((row, col, 3), np.uint8)
for r in range(row):
    for c in range(col):
        if offset > 0:
            offset = -1 * offset          
        pt = np.array([r+offset,c,1]) #Adjust the offset.
        newpt = np.matmul(invRot, pt) #Reverse map by reverse rotation and pick up color.

        #Check the bounds of the inverse pts we got and if they lie in the original image,
        #then copy the color from that original pt to the new matrix/image.
        if (newpt[0] >= 0 and newpt[0] < (yLen - 1) and newpt[1] >= 0 and newpt[1] < (xLen - 1)):
            x = np.asarray(newpt[1])
            y = np.asarray(newpt[0])

            x0 = np.floor(x).astype(int)
            x1 = x0 + 1
            y0 = np.floor(y).astype(int)
            y1 = y0 + 1

            Ia = img[y0, x0]
            Ib = img[y1, x0]
            Ic = img[y0, x1]
            Id = img[y1, x1]

            color1 = (x1-x) * (y1-y) * Ia
            color2 = (x1-x) * (y-y0) * Ib
            color3 = (x-x0) * (y1-y) * Ic
            color4 = (x-x0) * (y-y0) * Id

            weightedAvgColor = color1 + color2 + color3 + color4
            newmatrix[r][c] = weightedAvgColor 


import itertools
def testing(img, transform):
    inverse = AffineTransform(np.linalg.inv(transform.matrix))
    maxX = img.shape[0] - 1
    maxY = img.shape[1] - 1

    x,y = transform.applyAll([[0, 0], [maxX, 0], [maxX, maxY], [0, maxY]]).T
    r0 = np.min(x)
    c0 = np.min(y)
    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)

    for r in range(width):
        for c in range(height):
            pt = inverse @ np.array([r+r0, c+c0, 1])

            if pt[0] >= 0 and pt[0] < img.shape[0]-1 and pt[1] >= 0 pt[1] < img.shape[1]-1:
                x, y = pt

                x0, y0 = np.floor(pt).astype(int)
                x1, y1 = x0+1, y0+1

                neighbors = img[x0:x0+1, y0:y0+2]
                wx = np.array([[x0+1 - x], [x - x0]])
                wy = np.array([[y0+1 - y], [y - y0]])
                weights = wx @ wy.T

                # Einsum for elementwise multiplication, broadcast to higher dimentions
                final = sum(np.einsum('ij,ij...->ij...', weights, neighbors))
