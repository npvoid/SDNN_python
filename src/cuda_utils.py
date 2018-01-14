import numpy as np
from numba import *
from numba import cuda

"""
    CUDA Kernel implementations 
"""


@cuda.jit(argtypes=[uint8[:, :, :], float32[:, :, :], uint8[:, :, :], float32[:, :, :, :],
                    uint32, float32])
def conv_step(S, V, s, w, stride, th):

    idx, idy, idz = cuda.grid(3)
    if idx > V.shape[0] - 1:
        return
    if idy > V.shape[1] - 1:
        return
    if idz > V.shape[2] - 1:
        return

    if V[idx, idy, idz] > th:
        V[idx, idy, idz] = 0.

    result = 0.
    for k in range(w.shape[2]):
        for j in range(w.shape[1]):
            for i in range(w.shape[0]):
                result += w[i, j, k, idz] * s[idx*stride + i, idy*stride+j, k]

    V[idx, idy, idz] += result
    if V[idx, idy, idz] > th:
        S[idx, idy, idz] = 1
    else:
        S[idx, idy, idz] = 0


@cuda.jit(argtypes=[uint8[:, :, :], uint8[:, :, :], float32[:, :, :],
                    uint32, float32])
def pool(S, s, w, stride, th):

    idx, idy, idz = cuda.grid(3)
    if idx > S.shape[0] - 1:
         return
    if idy > S.shape[1] - 1:
         return
    if idz > S.shape[2] - 1:
         return

    result = 0.
    for j in range(w.shape[1]):
        for i in range(w.shape[0]):
            result += w[i, j, idz] * s[idx*stride + i, idy*stride+j, idz]

    if result > th:
        S[idx, idy, idz] = 1
    else:
        S[idx, idy, idz] = 0


@cuda.jit(argtypes=[int32[:], uint8[:, :, :], float32[:, :, :, :], uint8[:, :, :],
                    float32[:], int16[:], int16[:],
                    uint32, uint32, float32, float32])
def STDP_learning(S_sz, s, w, K_STDP,  # Input arrays
                  maxval, maxind1, maxind2,  # Indices
                  stride, offset, a_minus, a_plus):  # Parameters

    idx, idy, idz = cuda.grid(3)
    if idx > S_sz[0] - 1:
        return
    if idy > S_sz[1] - 1:
        return
    if idz > S_sz[2] - 1:
        return

    if idx != maxind1[idz] or idy != maxind2[idz]:  # Check if this is the neuron we have to update (correct idx, idy for map idz)
        return

    for i in range(w.shape[3]):
        if (idz != i and maxind1[idz] <= maxind1[i] + offset and maxind1[idz] >= maxind1[i] - offset
            and maxind2[idz] <= maxind2[i] + offset and maxind2[idz] >= maxind2[i] - offset and maxval[i] > maxval[idz]):
            maxval[idz] = 0.

    if maxval[idz] > 0:
        # Weights STDP update
        for k in range(w.shape[2]):
            for j in range(w.shape[1]):
                for i in range(w.shape[0]):
                    input = s[idx * stride + i, idy * stride + j, k]
                    dw = input * a_minus * w[i, j, k, idz] * (1 - w[i, j, k, idz]) + \
                         input * a_plus * w[i, j, k, idz] * (1 - w[i, j, k, idz]) - \
                         a_minus * w[i, j, k, idz] * (1 - w[i, j, k, idz])
                    w[i, j, k, idz] += dw

        # Turn off the STDP for lateral neurons of the activated neuron in all planes
        for k in range(S_sz[2]):
            j = 0 if idy - offset < 0 else idy - offset
            while j <= (S_sz[1] - 1 if idy + offset > S_sz[1] - 1 else idy + offset):
                i = 0 if idx - offset < 0 else idx - offset
                while i <= (S_sz[0] - 1 if idx + offset > S_sz[0] - 1 else idx + offset):
                    K_STDP[i, j, k] = 0
                    i += 1
                j += 1

        # Turn off the STDP for all neurons in the plane of the activated neuron
        for j in range(S_sz[1]):
            for i in range(S_sz[0]):
                K_STDP[i, j, idz] = 0


@cuda.jit(argtypes=[uint8[:, :, :], float32[:, :, :], uint8[:, :]])
def lateral_inh(S, V, K_inh):

    idx, idy, idz = cuda.grid(3)
    if idx > V.shape[0] - 1:
        return
    if idy > V.shape[1] - 1:
        return
    if idz > V.shape[2] - 1:
        return

    # if neuron has not fired terminate the thread
    if S[idx, idy, idz] != 1:
        return

    # if a neuron in this position has fired before do not fire again
    if K_inh[idx, idy] == 0:
        S[idx, idy, idz] = 0
        return

    # neuron at this position but in other input map
    for k in range(V.shape[2]):
        if S[idx, idy, k] == 1 and V[idx, idy, idz] < V[idx, idy, k]:
            S[idx, idy, idz] = 0
            return
    K_inh[idx, idy] = 0


@cuda.jit(argtypes=[float32[:, :], float32[:, :], uint8[:], uint8])
def DoG_norm(img_out, img_in, image_size, win_size):

    idx, idy = cuda.grid(2)
    if idx > image_size[0] - 1:
         return
    if idy > image_size[1] - 1:
         return

    sumation = .0001
    j = 0 if idy-win_size < 0 else idy-win_size
    while j <= (image_size[1]-1 if idy+win_size > image_size[1]-1 else idy+win_size):
        i = 0 if idx - win_size < 0 else idx - win_size
        while i <= (image_size[0]-1 if idx+win_size > image_size[0]-1 else idx+win_size):
            sumation += img_in[i, j]
            i += 1
        j += 1

    mean = sumation / ((2*win_size+1)**2)
    img_out[idx, idy] = img_in[idx, idy] / mean
