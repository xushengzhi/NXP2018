#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
Author: Shengzhi Xu
Email: sz.xu@hotmail.com
Data: 12-3-2018
Project: 20171107EWI
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, pi, arcsin, arccos, arctan
from numpy import e, exp, log10, log, cov, rad2deg, deg2rad
from numpy.linalg import eig, svd, qr

class SpatialSmoothing(object):

    def __init__(self, data):
        self.data = data            # (element, snapshots)
        self.p, self.q = self.data.shape

    def spatial_smoothing(self, derank_p=1, derank_q=1):

        C = np.zeros((self.p-derank_p, self.p-derank_q), dtype=complex)
        for i in range(derank_p):
            for j in range(derank_q):
                sub_data = self.data[i:self.p-derank_p+i, j:self.q-derank_q+j]
                C = C + sub_data.dot(sub_data.T.conj())
        C = C/derank_p/derank_q
        J = np.fliplr(np.eye(C.shape[0]))

        return (C + J.dot(C.conj()).dot(J))/2

if __name__ == '__main__':

    # Test :: 1 target will be smoothing into 2 targets??

    freq = [0.2, 0.2, 0,2]
    angle = rad2deg([30, -20, 0])
    derank = 3


    X, Y = np.meshgrid( np.arange(30), np.arange(12))
    data = np.zeros((12,30), dtype=complex)
    for i in range(3):
        data += e**(2j*pi*( 0.5*sin(angle[i])* Y + freq[i]*X ))

    R = data.dot(data.conj().T)
    E, I = eig(R)

    C = SpatialSmoothing(data=data).spatial_smoothing(derank_p=derank, derank_q=derank)
    G, H = eig(C)

    plt.figure()
    plt.scatter(x=np.arange(12), y=log10(abs(np.sort(E.real))))
    plt.figure()
    plt.scatter(x=np.arange(12-derank), y=log10(abs(np.sort(G.real))))

    # plt.figure()
    # plt.imshow(abs(fftshift(fft2(data, s=[128,128]))))
    plt.show()