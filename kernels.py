import itertools
import numpy as np
from numpy.typing import NDArray
from typing import Callable,Any

def extend_image(pixels, radius) -> NDArray[np.uint8]:
    padded_pixels: NDArray[np.uint8] = np.pad(pixels, ((radius, radius),(radius,radius),(0,0)))  # type: ignore

    for i in range(0, radius):
        padded_pixels[i] = padded_pixels[radius]
        padded_pixels[:,i] = padded_pixels[:,radius]
    for i in range(-radius, 0):
        padded_pixels[i] = padded_pixels[-radius-1]
        padded_pixels[:,i] = padded_pixels[:,-radius-1]

    return padded_pixels

def boxfilter(pixels:NDArray[np.uint8], radius:int=1):
    result:NDArray[np.uint16] = np.zeros(pixels.shape, dtype=np.uint16)  # type: ignore
    padded_pixels: NDArray[np.uint8] = extend_image(pixels, radius)

    print(padded_pixels[...,0])
    
    get_slice: Callable[[int], slice] = lambda k: slice(radius+k,k-radius if k < radius else None)
    for i,j in itertools.product(range(-radius,radius+1), range(-radius,radius+1)):
        # print(weighted_pixels[sliceX, sliceY,0])
        result += padded_pixels[get_slice(i), get_slice(j)]

    return np.array(result / (2*radius + 1)**2 , dtype=np.uint8)


def kernelfilter(pixels:NDArray[np.uint8], kernel:NDArray[Any]) -> NDArray[np.uint8]:
    radius = len(kernel)//2
    result:NDArray[np.uint32] = np.zeros(pixels.shape, dtype=np.int32)  # type: ignore
    padded_pixels: NDArray[np.uint8] = extend_image(pixels, radius)

    get_slice: Callable[[int], slice] = lambda k: slice(radius+k,k-radius if k < radius else None)
    for i,j in itertools.product(range(-radius,radius+1), range(-radius,radius+1)):
        # print(weighted_pixels[sliceX, sliceY,0])
        x = kernel[radius+i,radius+j]*padded_pixels[get_slice(i), get_slice(j)]
        print(x.dtype)
        result += x

    return np.array(result, dtype=np.uint8)

sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8)
sobel_kernel2 = np.flip(sobel_kernel.T)

def sobel_filter():
    pass

laplace_kernel = 1/4 * np.array([[0,1,0], [1,-4,1], [0,1,0]])
    