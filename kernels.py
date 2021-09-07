import itertools
import numpy as np
from numpy.typing import NDArray
from typing import Callable,Any,List,Tuple

def extend_image(pixels:NDArray[np.uint8], radius:int) -> NDArray[np.uint8]:
    padding:List[Tuple[int,int]] = [(0,0)]*len(pixels.shape)
    padding[0] = padding[1] = (radius,radius)
    padded_pixels: NDArray[np.uint8] = np.pad(pixels, padding)  # type: ignore

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
    
    get_slice: Callable[[int], slice] = lambda k: slice(radius+k,k-radius if k < radius else None)
    for i,j in itertools.product(range(-radius,radius+1), range(-radius,radius+1)):
        # print(weighted_pixels[sliceX, sliceY,0])
        result += padded_pixels[get_slice(i), get_slice(j)]

    return np.array(result / (2*radius + 1)**2 , dtype=np.uint8)


def convolve(pixels:NDArray[np.uint8], kernel:NDArray[Any]) -> NDArray[float]:
    radius = len(kernel)//2
    result:NDArray[float] = np.zeros(pixels.shape, dtype=float)  # type: ignore
    padded_pixels: NDArray[np.uint8] = extend_image(pixels, radius)

    get_slice: Callable[[int], slice] = lambda k: slice(radius+k,k-radius if k < radius else None)
    for i,j in itertools.product(range(-radius,radius+1), range(-radius,radius+1)):
        # print(weighted_pixels[sliceX, sliceY,0])
        x = (kernel[radius+i,radius+j]*padded_pixels[get_slice(i), get_slice(j)]).astype(float)
        # print(x.dtype)
        result += x

    return result

sobel_kernel_x = 1/8 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8)
sobel_kernel_y = 1/8 * np.flip(sobel_kernel.T)

laplace_kernel = 1/4 * np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
gaussian_kernel = 1/16 * np.array([[1,2,1],[2,4,2],[1,2,1]])

def sobel_filter(pixels):
    pixels = convolve(pixels, gaussian_kernel)

    dx = convolve(pixels, sobel_kernel_x)
    dy = convolve(pixels, sobel_kernel_y)

    return np.sqrt(dx**2 + dy**2)
