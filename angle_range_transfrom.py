#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
Author: Shengzhi Xu
Email: sz.xu@hotmail.com
Data: 9-3-2018
Project: 20171107EWI
'''

import numpy as np
from numpy import sin, pi, arcsin
from scipy import interpolate

class AngleRangeTransfrom(object):

    '''
    Method to tansform between map (sine, range) and map (theta, range)

    sine_to_degree: (sine, range) -> (theta, range)
    degree_to_sine: (theta, range) -> (sine, range)

    :return interpolated data

    '''

    def __init__(self, data, rmin, rmax, angle_min=-pi/2, angle_max=pi/2):

        self.data = data
        self.p, self.q = self.data.shape        # (range, angle)
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.range = np.linspace(rmin, rmax, self.p)

    def sine_to_degree(self, kind='linear'):

        angle = np.arcsin( np.linspace(sin(self.angle_min), sin(self.angle_max), self.q) )
        new_angle = np.linspace(self.angle_min, self.angle_max, self.q)
        interp_data = interpolate.interp2d(angle, self.range, self.data, kind=kind)(new_angle, self.range)

        return interp_data

    def degree_to_sine(self, kind='linear'):

        sine = np.sin(np.linspace(self.angle_min, self.angle_max, self.q))
        new_sine = np.linspace(arcsin(self.angle_min), arcsin(self.angle_max), self.q)
        inter_data = interpolate.interp2d(sine, self.range, self.data, kind=kind)(new_sine, self.range)

        return inter_data

if __name__ == '__main__':
    pass
