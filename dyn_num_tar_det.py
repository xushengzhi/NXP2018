# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:33:32 2018

@author: shengzhixu
"""

import os, sys, time

import numpy as np
import matplotlib.pyplot as plt
from pylab import pi, exp, sin, cos, log10, fftshift, fft
import scipy.io as spio
from scipy.signal import hamming, hann


x = np.arange(64)
y = np.exp(2j*pi*0.1*x) + 0.5*np.exp(-2j*pi*0.4*x)

HAM = hamming(64)
HAN = hann(64)

plt.figure()
plt.plot(10*log10(abs(fft(y, 1028))), label="no win")
plt.plot(10*log10(abs(fft(y*HAM, 1028))), label="hamming")
plt.plot(10*log10(abs(fft(y*HAN, 1028))), label="hann")
plt.legend()

plt.figure()
plt.plot(HAM, label="hamming")
plt.plot(HAN, label="hann")
plt.legend()

plt.show()