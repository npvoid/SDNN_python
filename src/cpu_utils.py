import numpy as np
from numba import *

"""
    CPU Kernel implementations 
"""

@jit
def conv_step_CPU(S, V, s, w, stride, th):

    V[V > th] = 0.
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                V[i, j, k] += np.sum(w[:, :, :, k] * s[i * stride:i * stride + w.shape[0], j * stride:j * stride + w.shape[1], :])


    S = (V > th).astype(int)*np.ones(S.shape)
    return V, S

@jit
def pool_CPU(S, s, w, stride, th):

    V_tmp = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            for k in range(S.shape[2]):
                V_tmp[i, j, k] += np.sum(w[:, :, k] * s[i*stride:i*stride+w.shape[0], j*stride:j*stride+w.shape[1], k])

    S = (V_tmp > th).astype(int)*np.ones(S.shape)
    return S

@jit
def lateral_inh_CPU(S, V, K_inh):
    S_inh = np.ones(S.shape, dtype=S.dtype)
    K = np.ones(K_inh.shape, dtype=K_inh.dtype)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            for k in range(V.shape[2]):
                flag = False
                if S[i, j, k] != 1:
                    continue
                if K_inh[i, j] == 0:
                    S_inh[i, j, k] = 0
                    continue
                for kz in range(V.shape[2]):
                    if S[i, j, kz] == 1 and V[i, j, k] < V[i, j, kz]:
                        S_inh[i, j, k] = 0
                        flag = True
                if flag:
                    continue
                else:
                    K[i, j] = 0
    S *= S_inh
    K_inh *= K
    return S, K_inh



    # # if neuron has not fired terminate the thread
    # if S[idx, idy, idz] != 1:
    #     return
    #
    # # if a neuron in this position has fired before do not fire again
    # if K_inh[idx, idy] == 0:
    #     S[idx, idy, idz] = 0
    #     return
    #
    # # neuron at this position but in other input map
    # for k in range(V.shape[2]):
    #     if S[idx, idy, k] == 1 and V[idx, idy, idz] < V[idx, idy, k]:
    #         S[idx, idy, idz] = 0
    #         return
    # K_inh[idx, idy] = 0


@jit
def STDP_learning_CPU(S_sz, s, w, K_STDP,  # Input arrays
                  maxval, maxind1, maxind2,  # Indices
                  stride, offset, a_minus, a_plus):  # Parameters
    for idx in range(S_sz[0]):
        for idy in range(S_sz[1]):
            for idz in range(S_sz[2]):

                if idx != maxind1[idz] or idy != maxind2[idz]:  # Check if this is the neuron we have to update (correct idx, idy for map idz)
                    continue

                for i in range(w.shape[3]):
                    if (idz != i and maxind1[idz] <= maxind1[i] + offset
                        and maxind1[idz] >= maxind1[i] - offset
                        and maxind2[idz] <= maxind2[i] + offset
                        and maxind2[idz] >= maxind2[i] - offset
                        and maxval[i] > maxval[idz]):
                        maxval[idz] = 0.

                # Weights STDP update
                if maxval[idz] > 0:
                    # Weights STDP update
                    input = np.zeros(w[:, :, :, idz].shape)
                    if idy*stride >= S_sz[1] - w.shape[1] and idx*stride >= S_sz[0] - w.shape[0]:
                        ss = s[idx * stride:, idy * stride:, :]
                        input[:ss.shape[0], :ss.shape[1], :] = ss
                    elif idy*stride >= S_sz[1] - w.shape[1]:
                        ss = s[idx * stride:idx * stride + w.shape[0], idy * stride:, :]
                        input[:, :ss.shape[1], :] = ss
                    elif idx*stride >= S_sz[0] - w.shape[0]:
                        ss = s[idx * stride:, idy * stride:idy * stride + w.shape[1], :]
                        input[:ss.shape[0], :, :] = ss
                    else:
                        input = s[idx * stride:idx*stride+w.shape[0], idy*stride:idy*stride+w.shape[1], :]
                    dw = input * a_minus * w[:, :, :, idz] * (1 - w[:, :, :, idz]) + \
                         input * a_plus * w[:, :, :, idz] * (1 - w[:, :, :, idz]) - \
                         a_minus * w[:, :, :, idz] * (1 - w[:, :, :, idz])
                    w[:, :, :, idz] += dw

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
    return w, K_STDP

@jit
def DoG_norm_CPU(img_out, img_in, image_size, win_size):

    sumation = .0001
    for idx in range(img_in.shape[0]):
        for idy in range(img_in.shape[1]):
            j = 0 if idy-win_size < 0 else idy-win_size
            while j <= (image_size[1]-1 if idy+win_size > image_size[1]-1 else idy+win_size):
                i = 0 if idx - win_size < 0 else idx - win_size
                while i <= (image_size[0]-1 if idx+win_size > image_size[0]-1 else idx+win_size):
                    sumation += img_in[i, j]
                    i += 1
                j += 1
            mean = sumation / ((2*win_size+1)**2)
            img_out[idx, idy] = img_in[idx, idy] / mean
    return img_out
