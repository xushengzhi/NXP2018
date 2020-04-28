#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
Author: Shengzhi Xu
Email: sz.xu@hotmail.com
Data: 7-3-2018
Project: 20171107EWI
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, deg2rad
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter

params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


class PolarPlot(object):
    
    '''
    To be added
    '''

    def __init__(self, fig, rect, angle_ticks, radius_ticks
                 , min_rad, max_rad, min_angle=0, max_angle=180):

        self.fig = fig
        self.rect = rect
        self.angle_ticks = angle_ticks
        self.radius_ticks = radius_ticks
        self.min_rad = min_rad
        self.max_rad = max_rad
        self.max_angle = deg2rad(max_angle)
        self.min_angle = deg2rad(min_angle)

    def polarplot(self):

        tr = PolarAxes.PolarTransform()

        grid_locator1 = FixedLocator([v for v,s in self.angle_ticks])
        tick_formatter1 = DictFormatter(dict(self.angle_ticks))

        grid_locator2 = FixedLocator([a for a,b in self.radius_ticks])
        tick_formatter2 = DictFormatter(dict(self.radius_ticks))

        grid_helper = floating_axes.GridHelperCurveLinear(tr,
                          extremes=(self.max_angle, self.min_angle, self.max_rad, self.min_rad),
                          grid_locator1=grid_locator1,
                          grid_locator2=grid_locator2,
                          tick_formatter1=tick_formatter1,
                          tick_formatter2=tick_formatter2)

        ax = floating_axes.FloatingSubplot(self.fig, self.rect, grid_helper=grid_helper)
        self.fig.add_subplot(ax)
        ax.grid(True, color='b', linewidth=0.2, linestyle='-')

        aux_ax = ax.get_aux_axes(tr)
        aux_ax.patch = ax.patch
        ax.patch.zorder = 0.9

        ax.axis['bottom'].set_label(r'Angle ($^{\circ}$)')
        ax.axis['bottom'].major_ticklabels.set_rotation(180)
        ax.axis['bottom'].label.set_rotation(180)
        ax.axis['bottom'].LABELPAD += 30
        ax.axis['left'].set_label('Range (m)')
        ax.axis['left'].label.set_rotation(0)
        ax.axis['left'].LABELPAD += 15
        ax.axis['left'].set_visible(True)

        return ax, aux_ax


class RangeAnglePolarPlot():
    
    '''
    To be added
    '''

    def __init__(self, fig,  data, rmin, rmax, rinc=5):
        self.fig = fig
        self.data = data
        self.rmin = rmin
        self.rmax = rmax
        self.rinc = rinc
        self.angle_size, self.range_size = data.shape

    def polar_plot(self, title='Angle - Range Map', levels=None, cmap='hot',
                   ylabel=None, normalizer=True):

        if levels == None:
            levels = np.linspace(0, 1, 21)

        # set ticks
        angle_ticks = range(0, 181, 10)
        angle_ticks_rads = deg2rad(angle_ticks)

        angle_ticks_for_plot = []
        radius_ticks_for_plot = []

        for i, a in enumerate(angle_ticks):
            angle_ticks_for_plot.append((angle_ticks_rads[i], r"$"+str(-a+90)+"$"))

        rad_tick_start_point = np.ceil(self.rmin/self.rinc)*self.rinc
        rad_tick_end_point = np.floor(self.rmax/self.rinc)*self.rinc
        radius_ticks = np.arange(rad_tick_start_point, rad_tick_end_point+1, self.rinc)
        for i, r in enumerate(radius_ticks):
            radius_ticks_for_plot.append((radius_ticks[i], r"$"+str(r) + r"$"))

        ax, aux = PolarPlot(fig=self.fig, rect=111, angle_ticks=angle_ticks_for_plot,
                            radius_ticks=radius_ticks_for_plot,
                            min_rad=self.rmin,
                            max_rad=self.rmax,
                            min_angle=0,
                            max_angle=180).polarplot()

        # Set Gird
        azimuth = deg2rad(np.linspace(0, 180, self.angle_size))
        zeniths = np.linspace(self.rmin, self.rmax, self.range_size)
        r, theta = np.meshgrid(zeniths, azimuth)

        # data normalize to 0~1
        if normalizer == True:
            data_max = np.max(self.data)
            data_min = np.min(self.data)
            self.data = (self.data - data_min)/(data_max- data_min)

        pic = aux.contourf(theta, r, self.data, levels=levels, cmap=cmap)

        # optional
        cbar = plt.colorbar(pic, orientation='vertical')
        # cbar.set_label('(dB)', fontsize=20)
        if ylabel is not None:
            cbar.ax.set_ylabel(ylabel, fontsize=16, rotation=-90, labelpad=18)
        if title is not None:
            plt.suptitle(t=title, fontsize=18, weight='bold')

        return pic
    
    
if __name__ == '__main__':
    pass
