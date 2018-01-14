import numpy as np
from scipy.ndimage import correlate
from math import ceil
from PIL import Image
from PIL.Image import ANTIALIAS
from numba import cuda
from cuda_utils import DoG_norm
from cpu_utils import DoG_norm_CPU


def DoG_normalization(img):
    img = img.astype(np.float32)
    img_out = np.zeros(img.shape).astype(np.float32)
    img_sz = np.array([img.shape[0], img.shape[1]], dtype=np.uint8)

    blockdim = (10, 10)
    griddim = (int(ceil(img.shape[0] / blockdim[0])), int(ceil(img.shape[1] / blockdim[1])))

    d_img_in = cuda.to_device(img)
    d_img_out = cuda.to_device(img_out)
    d_img_sz = cuda.to_device(img_sz)
    DoG_norm[griddim, blockdim](d_img_out, d_img_in, d_img_sz, 8)
    d_img_out.to_host()

    return img_out

def DoG_normalization_CPU(img):
    img = img.astype(np.float32)
    img_out = np.zeros(img.shape).astype(np.float32)
    img_sz = np.array([img.shape[0], img.shape[1]], dtype=np.uint8)
    img_out = DoG_norm_CPU(img_out, img, img_sz, 8)
    return img_out

def DoG_filter(path_img, filt, img_size, total_time, num_layers):
    """
        DoG filter implementation based on Kheradpisheh, S.R., et al. 'STDP-based spiking deep neural networks 
        for object recognition'. arXiv:1611.01421v1 (Nov, 2016)
    """

    # Open image, convert to grayscale and resize
    img = Image.open(path_img)
    img = img.convert('L')
    img = img.resize(img_size, ANTIALIAS)
    img = np.asarray(img.getdata(), dtype=np.float64).reshape((img.size[1], img.size[0]))

    # Apply filter
    img = correlate(img, filt, mode='constant')

    # Border
    border = np.zeros(img.shape)
    border[5:-5, 5:-5] = 1.
    img = img * border

    # Threshold
    img = (img >= 15).astype(int) * img
    img = np.abs(img)  # Convert -0. to 0.

    # DoG Normalization
    # img_out = DoG_normalization(img)
    # img_out = DoG_normalization_CPU(img)
    img_out = img

    # Convert to spike times
    I = np.argsort(1 / img_out.flatten())  # Get indices of sorted latencies
    lat = np.sort(1 / img_out.flatten())  # Get sorted latencies
    I = np.delete(I, np.where(lat == np.inf))  # Remove infinite latencies indexes
    II = np.unravel_index(I, img_out.shape)  # Get the row, column and depth of the latencies in order
    t_step = np.ceil(np.arange(I.size) / ((I.size) / (total_time - num_layers))).astype(np.uint8)
    II += (t_step,)
    spike_times = np.zeros((img_out.shape[0], img_out.shape[1], total_time))
    spike_times[II] = 1

    return spike_times

def DoG(size, s1, s2):
    """
        Generates a filter window of size size x size with std of s1 and s2
    """
    r = np.arange(size)+1
    x = np.tile(r, [size, 1])
    y = x.T
    d2 = (x-size/2.-0.5)**2 + (y-size/2.-0.5)**2
    filt = 1/np.sqrt(2*np.pi) * (1/s1 * np.exp(-d2/(2*(s1**2))) - 1/s2 * np.exp(-d2/(2*(s2**2))))
    filt -= np.mean(filt[:])
    filt /= np.amax(filt[:])
    return filt
