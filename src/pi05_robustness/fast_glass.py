"""Bitwise-equivalent acceleration candidate for pinned LIBERO-Plus glass blur.

Not enabled automatically. The reference's NumPy tuple assignment aliases views:
the second assignment reads the already updated pixel, so the operation copies
the neighbour rather than performing a conventional swap. Preserve that behavior.
"""
import numpy as np
from numba import njit
from skimage.filters import gaussian


@njit(cache=True)
def _copy_pixels(image, shifts, delta):
    height,width,channels=image.shape
    for iteration in range(shifts.shape[0]):
        for ih in range(shifts.shape[1]):
            h=height-delta-ih
            for iw in range(shifts.shape[2]):
                w=width-delta-iw
                hp=h+shifts[iteration,ih,iw,1]
                wp=w+shifts[iteration,ih,iw,0]
                for channel in range(channels):
                    image[h,w,channel]=image[hp,wp,channel]
    return image


def glass_blur(image,severity=1):
    sigma,delta,iterations=[(.5,1,3),(.7,1,3),(.9,2,3),(1.,2,2),(1.1,3,2),
                            (1.3,3,2),(1.5,4,2),(1.8,4,2),(2.2,5,1),(2.5,5,1)][severity-1]
    x=np.uint8(gaussian(np.array(image)/255.,sigma=sigma,channel_axis=-1)*255)
    height,width=x.shape[:2]
    shifts=np.random.randint(-delta,delta,size=(iterations,height-2*delta,width-2*delta,2))
    _copy_pixels(x,shifts,delta)
    return np.clip(gaussian(x/255.,sigma=sigma,channel_axis=-1),0,1)*255
