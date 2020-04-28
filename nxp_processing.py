#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
Author: Shengzhi Xu
Email: sz.xu@hotmail.com
Data: 13-3-2018
Project: 20171107EWI
'''

import time
import tkinter as tk
from tkinter import filedialog
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import loadmat
import h5py
from scipy.constants import speed_of_light as C
from numpy import sin, pi, e, log10
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import hamming, hann
from tqdm import tqdm

from polarplot import RangeAnglePolarPlot
from angle_range_transfrom import AngleRangeTransfrom
from spatial_smoothing import SpatialSmoothing


#%%
## utils
def normalizer(mat):
    max_value = np.max(mat)
    return mat - max_value

def eigsort(eigresult):
    """
    Sort the output of scipy.linalg.eig() in terms of
    eignevalue magnitude
    """
    ix = sp.argsort(abs(eigresult[0]))
    return ( eigresult[0][ix], eigresult[1][:,ix] )

def db(data):
    data = abs(data)
    data[data==0] = np.nan
    data = 10*log10(data)
    data = data - min(data)
    data[data==np.nan] = 0
    return data

def clim(data, clim=20):
    max_value = np.max(data)
    threshold = max_value - clim
    data[data<=threshold]=threshold
    return data

def num_tar_det(eigenvalue, threshold_cof):
    num_tar = 0
    eigenvalue = abs(eigenvalue)
    while 1:
        mean_val = np.mean(eigenvalue)
        thresh = threshold_cof * mean_val
        new_tar = sum(eigenvalue > thresh)
        if new_tar == 0:
            break
        else:
            num_tar += new_tar
            eigenvalue = eigenvalue[new_tar::]
            
    return num_tar

params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

D = 1.889e-3
J = 2j*np.pi
E = e

#%%

class NXPReal(object):
    '''
    NXP data processing object.
    Please make sure the Calibration.mat and the RadarSettings.mat are in the current fold!!
    
    '''

#%%
    def __init__(self,
                 slice_number=None,
                 direct_wave_range=3,
                 end_wave_range=None,
                 verbose=False):
        
        # Defaut system setting
        
        self.slice = slice_number
        self.direct_wave_range = direct_wave_range
        self.angle_music_number = 180
        self.range_fft_number = 2048
        self.doppler_fft_number = 1024
        self.angle_fft_number = 128
        self.end_wave_range = end_wave_range
        self.verbose = verbose

        self._load_setting()

#%%
    def load_data(self, path):
        
        # Load Bin Data
        
        # Choose file
        if self.slice is not None:
            fold = ((self.slice-1) // 200) * 200
            
            path = '{}/{}_{}/'.format(path, str(fold + 1).zfill(4), str(fold + 200).zfill(4))

            self.file = '{}th_slice'.format(self.slice)  # 835
            print('\nLoading Data and Parameters of slice {}...\n'.format(self.slice))
            try:
                f = loadmat(path + self.file)
            except:
                print("Cannot find the data, please check the path: {}".format(path + self.file))
        
        else:
            root = tk.Tk()
            root.update()
            path = filedialog.askopenfilename()
            root.destroy()
            try:
                f = loadmat(path)
                print('\nLoading Data and Parameters of slice {}...\n'.format(path.split('/')[-1][0:4]))
            except:
                print("Cannot open the file!")
    
        DATA = f['Dat']
        self._extended_parameter()

        return DATA

#%%
    def _load_setting(self):
        
        # System SETTINGS
        
        try:
            f = sp.io.loadmat('Calibration.mat')
            self.CAL = f['CAL']
        except FileNotFoundError:
            print('Cannot find the Calibration.mat!')
            self.CAL = np.ones((1, 12))
            
        self.SETTINGS = {}
        if self.verbose:
            print('\n\nLoading Radar Settings...\n')
        try:
            with h5py.File('RadarSettings.mat') as f:
                for i in list(f['settings'].keys()):
                    self.SETTINGS[i] = f['settings'][i].value[0, 0]
                    self.SETTINGS['MIMO_coding_matrix'] = f['settings']['MIMO_coding_matrix'].value
                    exec('self.{} = f[{}][{}].value[0,0]'.format(i, "'settings'", "i"))
                    if self.verbose:
                        print(i, ':  ', f['settings'][i].value[0, 0])
        except:
            print('\nCannot find the RadarSettings.mat!!!!!!!!!!!!\n\n\n')
#            exit()

        self.LAMBDA = C/self.SETTINGS['Fc']
        self.FD = D/self.LAMBDA

#%%
    def _extended_parameter(self):
        
        # Deriviated from system SETTINGS
        
        self.range_resolution = C / 2 / self.SETTINGS['BW']
        self.mu = self.SETTINGS['BW'] / \
                  (self.SETTINGS['Chirp_time'] - self.SETTINGS['DwellTime'] - self.SETTINGS['Reset_time'])
        self.range_max = 0.5 * (C * self.SETTINGS['Fs'] / 2 / self.mu) * 0.8
        if self.end_wave_range is None:
            self.end_wave_range = self.range_max
        self.velocity_max = 0.5 * C / 2 / self.SETTINGS['Chirp_time'] / self.SETTINGS['Fc']
        self.unmigrate_velocity = C / 2 / self.SETTINGS['BW'] / self.SETTINGS['Chirp_time'] / self.SETTINGS['NChirps']
        self.range_velocity_coupling_cof = self.mu / 2 / self.SETTINGS['Fc'] / self.SETTINGS['Fs']
        self.range_angle_coupling_cof = D * self.mu / self.SETTINGS['Fs'] / C
        if self.verbose:
            print('Unambiguous Range is: {:.2f} meter.'.format(self.range_max))
            print('Unambiguous velocity is {:.2f} m/s.'.format(self.velocity_max))
            print('Migration velocity is {:.2f} m/s.'.format(self.unmigrate_velocity))
            print('Range Velocity coupling coefficient is {:.2g}.'.format(self.range_velocity_coupling_cof))
            print('Range Angle coupling coefficient is {:.2g}.'.format(self.range_angle_coupling_cof))
    
            print('Maximum Range Velocity coupling: {:.2g}.'.format(self.range_velocity_coupling_cof*512*256))
            print('Maximum Range Angle coupling: {:.2g}.'.format(self.range_angle_coupling_cof * 12 * 512))

#%%
    def pre_processing(self, DATA, settle_time=4e-6, window_fun=None):
        
        '''
        Preprocessing the data from ttt -> ttk 
        '''

        # remove dwell time, reset time, and settle time
        start_point = int(self.SETTINGS['NSamples']*
                          (self.SETTINGS['DwellTime']+settle_time)/self.SETTINGS['Chirp_time']) + 1
        end_point = np.int(
            self.SETTINGS['NSamples'] - self.SETTINGS['NSamples'] *
            self.SETTINGS['Reset_time'] / self.SETTINGS['Chirp_time']) - 1
        data_beat_freq = DATA[:, :, :, start_point:end_point]

        # remove direct wave
        direct_wave_fft_grids = int(self.direct_wave_range * self.range_fft_number / 2 / self.range_max)
        end_wave_fft_grids = int(self.end_wave_range*self.range_fft_number /2/ self.range_max)
        
        # window function
        if window_fun == 'hamming':
            win_data = np.tile(hamming(data_beat_freq.shape[-1]),[data_beat_freq.shape[0], \
                               data_beat_freq.shape[1], data_beat_freq.shape[2], 1]) * data_beat_freq
        elif window_fun == 'hann':
            win_data = np.tile(hann(data_beat_freq.shape[-1]),[data_beat_freq.shape[0], \
                               data_beat_freq.shape[1], data_beat_freq.shape[2], 1]) * data_beat_freq
        else:
            win_data = data_beat_freq
        
        # ttt -> ttk
        range_fft_data = fft(win_data, n=self.range_fft_number, axis=-1)[:, :, :, 0:self.range_fft_number // 2]
        cal_data_full_fft = self._calibration(fft(win_data, axis=-1)) # MIMO virtual array calibration
        
        range_fft_data_cutoff = range_fft_data[:, :, :, direct_wave_fft_grids:end_wave_fft_grids]
        
        # MIMO virtual array calibration
        cal_data =  self._calibration(range_fft_data_cutoff=range_fft_data_cutoff)
        print("Data shape: ", cal_data.shape)
            
        return cal_data, cal_data_full_fft

#%%
    def _calibration(self, range_fft_data_cutoff):
        
        # calibration
        *num_virtue_array, num_slow_time, self.num_fft_fast_time = range_fft_data_cutoff.shape
        
        # MIMO data -> virtual array data
        virtual_array_data = range_fft_data_cutoff.reshape(12, num_slow_time, self.num_fft_fast_time)  # data:ttk
        
        # calibration with CALIBRATION vector
        correction = np.transpose(np.tile(self.CAL, [num_slow_time, self.num_fft_fast_time, 1]), [2, 0, 1])
        calibrated_data = virtual_array_data * correction

        return calibrated_data  #ttk

#%%
    def range_angle_fft(self, calibrated_data, slow_time_cutoff=16, cmap='jet', 
                        clim_am=15, save_fig=False, title=None, window_fun=None):
        
        tic = time.time()
        # window function for angle
        if window_fun == 'hamming':
            win_data = np.transpose(np.tile(hamming(calibrated_data.shape[0]), [calibrated_data.shape[1], \
                               calibrated_data.shape[2], 1]), [2, 0, 1] )* calibrated_data
        elif window_fun == 'hann':
            win_data = np.transpose(np.tile(hann(calibrated_data.shape[0]), [calibrated_data.shape[1], \
                               calibrated_data.shape[2], 1]), [2, 0, 1] )* calibrated_data
        else:
            win_data = calibrated_data
            
        fad = fftshift(fft(win_data[:, 0:slow_time_cutoff, :],
                           n=self.angle_fft_number, axis=0), axes=0)

        # plot
        absolute_fad = 10 * log10(abs(fad)+1e-30)
        pic = absolute_fad[:, 0, :]   # (angle, range)
        
        new_pic = AngleRangeTransfrom(pic.T, rmin=self.direct_wave_range, rmax=self.end_wave_range,
                                        angle_min=-pi / 2, angle_max=pi / 2).sine_to_degree()
        print('Time for FFT: {:.2f} seconds!'.format(time.time() - tic))
        fig = plt.figure(figsize=[14.5, 7])
        RangeAnglePolarPlot(fig=fig, data=(normalizer(clim(new_pic.T, clim_am))), rmax=self.end_wave_range,
                              rmin=self.direct_wave_range, rinc=10).polar_plot(cmap=cmap,
                                                                      title=title,
                                                                      levels=list(np.linspace(-clim_am, 0, 41)),
                                                                      normalizer=False,
                                                                      ylabel='(dB)')
        if save_fig:
            plt.savefig('FFT{}.jpg'.format(self.file), dpi=400)

        return new_pic.T

#%%
    def range_doppler_fft(self, calibrated_data, cmap='jet', save_fig=False):
        tic = time.time()
        frd = fftshift(fft(calibrated_data, n=self.doppler_fft_number, axis=-2), axes=-2)
        absolute_frd = 10 * log10(abs(frd))
        print('Time for FFT: {:.2f} seconds!'.format(time.time() - tic))
        # plot
        plt.figure()
        plt.imshow(np.fliplr(absolute_frd[0, :, :]).T, interpolation='nearest', cmap=cmap,
                   aspect='auto', clim=([np.max(absolute_frd) - 30, np.max(absolute_frd) - 8]),
                   extent=[-self.velocity_max, self.velocity_max, self.direct_wave_range, self.end_wave_range])
        plt.colorbar()

        if save_fig:
            plt.savefig('rd_fft {}.jpg'.format(self.file), dpi=400)

        return (absolute_frd)

#%%
    def angle_doppler_2d_music(self, calibrated_data, slow_time_cutoff=256):
        
        slow_time_cutoff = slow_time_cutoff
        dm = calibrated_data[:, 0:slow_time_cutoff, :].reshape(
            (12 * slow_time_cutoff, self.num_fft_fast_time))  # data: (tt)k

        R = dm.dot(dm.T.conj())
        eigval, eigvec = eigsort(np.linalg.eig(R))

        threshold = np.mean(abs(eigval)) * 20
        num_targets = sum(eigval >= threshold)
        print(num_targets)
        noise_subspace = eigvec[:, 0:12 * slow_time_cutoff - num_targets]
        noise_subspace_sqaure = noise_subspace.dot(noise_subspace.T.conj())

        freq_bins_number = 200
        angle_scan = np.linspace(-pi / 2, pi / 2, self.angle_music_number)
        freq_domain = np.linspace(-0.5, 0.5, freq_bins_number)
        Pmusic = np.zeros((self.angle_music_number, freq_bins_number))

        steer_func = lambda a,b: (E ** (-J * self.FD * sin(a) * np.arange(12)).reshape((12, 1))).dot(
                (E ** (-J * b * np.arange(slow_time_cutoff)).T).reshape((1, slow_time_cutoff)))
        # 2D MUSIC
        print('\n2D MUSIC processing for doppler-angle....')
        for i, a in tqdm(enumerate(angle_scan)):
            for j, b in enumerate(freq_domain):
                steering = steer_func(a, b)
                steering = steering.reshape((1, 12 * slow_time_cutoff))
                p = abs(steering.dot(noise_subspace_sqaure).dot(steering.T.conj()))
                Pmusic[i, j] = 20 * log10(1 / p[0, 0])

        plt.figure()
        plt.imshow(Pmusic.T, extent=[-90, 90, -20, 20], aspect='auto')
        plt.colorbar()

        return Pmusic

#%%
    def angle_range_2d_music(self, calibrated_data, slow_time_cutoff=256, title=None, save_fig=False, cut=20, tmp=61):
        slow_time_cutoff = slow_time_cutoff
        _, _, dz = calibrated_data.shape
        cut_data = np.delete( np.delete(calibrated_data, np.s_[0:cut], -1), np.s_[dz-cut::], -1)
        # cut_data = np.delete(calibrated_data, np.s_[104:204], -1)
        calibrated_data_ttt = ifft(cut_data, axis=-1) # ttk -> ttt
        dx, dy, dz = calibrated_data_ttt[:, 0:slow_time_cutoff, :].shape
        dm = np.transpose(calibrated_data_ttt[:, 0:slow_time_cutoff, :], [0, 2, 1]).reshape(dx*dz, dy)

        R = dm.dot(dm.T.conj())
        # spatial smoothing???
#        R = SpatialSmoothing(dm).spatial_smoothing(derank_q=1, derank_p=20)
        eigval, eigvec = eigsort(np.linalg.eig(R))

        threshold = np.mean(abs(eigval)) * 5
        num_targets = sum(eigval >= threshold)
        print(num_targets)
        noise_subspace = eigvec[:, 0: (dx* dz - num_targets)]
        noise_subspace_sqaure = noise_subspace.dot(noise_subspace.T.conj())

        freq_bins_number = 200
        angle_scan = np.linspace(-pi / 2, pi / 2, self.angle_music_number)
        freq_domain = np.linspace(0, 0.5, freq_bins_number, endpoint=True)
        Pmusic = np.zeros((self.angle_music_number, freq_bins_number))

        steer_func = lambda a,b: (E ** (-J * self.FD * sin(a) * np.arange(12)).reshape((12, 1))).dot(
                (E ** (-J * b * np.arange(dz)).T).reshape((1, dz)))
        # 2D MUSIC
        print('\n2D MUSIC processing for doppler-angle....')
        for i, a in tqdm(enumerate(angle_scan)):
            for j, b in enumerate(freq_domain):
                steering = steer_func(a, b)
                steering = steering.reshape((1, 12 * dz))
                p = abs(steering.dot(noise_subspace_sqaure).dot(steering.T.conj()))
                Pmusic[i, j] = 20 * log10(1 / p[0, 0])  # (angle, doppler)

        # plt.figure()
        # plt.imshow(Pmusic.T, extent=[-90, 90, self.direct_wave_range, self.end_wave_range], aspect='auto', cmap='jet')
        # plt.colorbar()

        fig = plt.figure(figsize=[14.5, 7])
        RangeAnglePolarPlot(fig=fig, data=(normalizer(clim(Pmusic[:, 0:69], 20))),
                            rmax=20,
                            rmin=3, rinc=10).polar_plot(cmap=cmap,
                                  title=title,
                                  levels=list(np.linspace(-20, 0, 41)),
                                  normalizer=False,
                                  ylabel='dB')
        if save_fig:
            plt.savefig('angle_range_2dmusic {}.jpg'.format(self.file), dpi=400)
        return Pmusic

#%%
    def range_angle_1d_music(self, fig, calibrated_data, derank_q=0, derank_p=0,
                             slow_time_cutoff=16, cmap='jet', save_fig=False, name=None, 
                             threshold_cof=4, clim_am=10, dynamic_thresh=True, 
                             title=None, norm_with_cov=True):
        
        angle_scan = np.linspace(-pi / 2, pi / 2, self.angle_music_number)
        Pmusic = np.zeros((len(angle_scan), self.num_fft_fast_time))
        spectral_norm = np.zeros(self.num_fft_fast_time, dtype=complex)
        amnorm = np.zeros(self.num_fft_fast_time, dtype=complex)  ###############################################
        print('\n1D MUSIC processing for range-angle....')
        tic = time.time()
        null_range = 0

        music_1d_data = calibrated_data[:, 0:slow_time_cutoff, :]  # ttk
        # amplitude_extraction = fft(music_1d_data, axis=1, n=doppler_fft_number)
        for i in range(self.num_fft_fast_time):
            #    print(i)
            tmp = music_1d_data[:, :, i]  ## tt
            
            ## SPATIAL SMOOTHING
            if derank_p|derank_q:
                R = SpatialSmoothing(tmp).spatial_smoothing(derank_q=derank_q, derank_p=derank_p)
            else:
                R = tmp.dot(tmp.T.conj())
            amnorm = np.linalg.norm(tmp, 2)
            eigval, eigvec = eigsort(np.linalg.eig(R))
            
            ## DYNAMIC THRESHOLD
            if dynamic_thresh == True:
                num_targets = num_tar_det(eigval, threshold_cof)
            else:
                threshold = np.mean(abs(eigval)) * threshold_cof
#                threshold = 0.001
                num_targets = sum(abs(eigval) >= abs(threshold))
#                print(threshold)
            
            
            ## JUMP ZERO TAR RANGE CELL
            if num_targets == 0:
                null_range = null_range + 1
                continue       
            elif num_targets >= 2:
                if self.verbose:
                    print('target number for range{}: {}'.format(i, num_targets))
                else:
                    pass
            else:
                pass
            
            ## MUSIC ALGORITHM
            noise_subspace = eigvec[:, 0:R.shape[0] - num_targets]
            spectral_norm[i] = max(abs(eigval))

            for j, a in enumerate(angle_scan):
                steering = E**(-J * self.FD * sin(a) * np.arange(R.shape[0])).reshape((1, R.shape[0] ))
                p = (abs(steering.dot(noise_subspace).dot(noise_subspace.T.conj()).
                         dot(steering.T.conj())))
                Pmusic[j, i] = log10(1 / p[0, 0])      #(angle, range)
            [ta, ti] = [np.max(Pmusic[:, i]), np.min(Pmusic[:, i])]
            Pmusic[:, i] = ((Pmusic[:, i] -ti + 1e-8) / (ta-ti))
            
            if norm_with_cov is True:
                Pmusic[:, i] = Pmusic[:, i]*10*np.log10(amnorm*np.sqrt(tmp.size**2)) #* abs(spectral_norm[i])**(1/8)
#                            * np.log10(spectral_norm[i].real) # *(self.SETTINGS['NRx'] + self.SETTINGS['NTx']))
        
        print('Time for 1D MUSIC: {:.2f} seconds!'.format(time.time() - tic))
        # plot
#        fig = plt.figure(figsize=[14.5, 7])
        pic = RangeAnglePolarPlot(fig=fig, data=(normalizer(clim(Pmusic, clim_am))), rmax=self.end_wave_range,
                                    rmin=self.direct_wave_range, rinc=10).polar_plot(levels=list(
                                        np.linspace(-clim_am, 0, 41)),
                                            title=title,
                                            cmap=cmap,
                                            normalizer=False,
                                            ylabel='(dB)')
        if save_fig:
            if name is None:
                plt.savefig('MUSIC{}.jpg'.format(self.file), dpi=400)
            else:
                plt.savefig(name, dpi=400)

        return pic

#%%
if __name__ == '__main__':
    plt.close('all')
    model = NXPReal(slice_number=1062, # 1062 and 1462
                    direct_wave_range=3, 
                    end_wave_range=20, 
                    verbose=True)
    # 2140 2066 2678 2149 1062 1475
    # slice derank thresh, cutoff
    # 1462, 2, 5, 32, nodyn, nowind
    
    # 1062, 2, 5, 32, nodyn, nowind
    # 1469-1473, 3, 5, 256      missing bike
    
    # 1062, 2, 6.5, 32, nodyn, 'hann'    # 很好， 哈哈
    # 1500, 0, 8, 32, dy, 'hamming'
    
    # For windows
    if sys.platform == 'win32':
        path = 'C:/Users/shengzhixu/Google Drive/MPaper/CAMA 2018/NXP processing MATLAB/BeatSignals'
        bad_data = model.load_data(path=path)
    # For mac
    else:
        path = '/Volumes/Personal/Google Drive/MPaper/CAMA 2018/NXP processing MATLAB/BeatSignals'
        bad_data = model.load_data(path=path)
    good_data, cal_data_full_fft = model.pre_processing(DATA=bad_data, window_fun=None)

    save_fig = True
    name = None     #'nonnormalize.png'
    derank = 2
    threshold_cof = 5
    cmap = 'jet'
    norm_with_cov = True
    clim_am = 20
    fig = plt.figure(figsize=[14.5, 7])
    pmusic = model.range_angle_1d_music(fig=fig, 
                                        calibrated_data=good_data, 
                                        derank_q=derank, 
                                        derank_p=derank,
                                        save_fig=save_fig, 
                                        name=name, 
                                        slow_time_cutoff=32, 
                                        cmap=cmap,
                                        threshold_cof=threshold_cof, 
                                        clim_am=clim_am, 
                                        dynamic_thresh=False,
                                        norm_with_cov=norm_with_cov)

    pfft = model.range_angle_fft(calibrated_data=good_data,
                                 save_fig=save_fig,
                                 cmap=cmap,
                                 clim_am=20,
                                 window_fun='hamming')
    # model.range_doppler_fft(calibrated_data=good_data)
#    model.angle_doppler_2d_music(calibrated_data=good_data)
#     Pmusic1 = model.angle_range_2d_music(calibrated_data=cal_data_full_fft, cut=10, tmp=70)
    # Pmusic2 = model.angle_range_2d_music(calibrated_data=cal_data_full_fft, cut=60)
    # Pmusic3 = model.angle_range_2d_music(calibrated_data=cal_data_full_fft, cut=100)

    plt.show()
