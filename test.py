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

