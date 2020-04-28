# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 10:01:59 2018

@author: shengzhixu
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from pylab import pi, exp, sin, cos, log10, fftshift, fft
import scipy.io as spio
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nxp_processing import NXPReal


fig = plt.figure(figsize=[14.5, 7])
#ax = fig.add_subplot(111)

# I like to position my colorbars this way, but you don't have to
#div = make_axes_locatable(ax)
#cax = div.append_axes('right', '5%', '5%')
#
#
#frames = []
#maxval = np.zeros(100)

#cv0 = frames[0]
#cf = ax.imshow(cv0)
#cb = fig.colorbar(cf, cax=cax)
#tx = ax.set_title('Frame 0')

def animate(i):
    plt.cla()
    plt.clf()
    save_fig = False
    derank = 0
    threshold_cof = 6
    cmap = 'jet'
    model = NXPReal(slice=i+1, direct_wave_range=3, end_wave_range=20)
    bad_data = model.load_data(path='D:/20171107EWI/BeatSignals')
    good_data = model.pre_processing(DATA=bad_data, window_fun='hamming')
    model.range_angle_1d_music(fig=fig, calibrated_data=good_data, derank_q=derank, derank_p=derank,
                               save_fig=save_fig, slow_time_cutoff=32, cmap=cmap, title="Slice "+str(i),
                               threshold_cof=threshold_cof, clim_am=20, dynamic_thresh=True)



ani = animation.FuncAnimation(fig, animate, frames=10, interval=1000)
plt.show()
#ani.save('dftmusic.mp4', dpi=100)
print('Done!!!!!!!!!')

