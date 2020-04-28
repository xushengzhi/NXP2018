import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pylab import *
from scipy.io import loadmat
from scipy.signal import hann, hamming
from numpy.fft import fftshift, fft2, fft, ifft
from scipy.constants import speed_of_light as C
import h5py
from polarplot import range_angle_polarplot
from angle_range_transfrom import angle_range_transfrom
import time


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

params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)



'''
Load data and pre-processing!
'''
## Load data and system parameters
slice = 911
fold = (slice//200)*200
path = 'BeatSignals/{}_{}/'.format(str(fold+1).zfill(4), str(fold+200).zfill(4))
file = '{}th_slice'.format(slice)        # 835

print('\nLoading Data and Parameters:')
f = loadmat(path+file)
DATA = f['Dat']
# DATA = np.squeeze(np.transpose(DATA, [4,3,2,1,0]))  # NTx, NRx, NSweep, NSamples

f = loadmat('Calibration.mat')
CAL = f['CAL']

SETTINGS = {}
with h5py.File('RadarSettings.mat') as f:
    for i in list(f['settings'].keys()):
        SETTINGS[i] = f['settings'][i].value[0, 0]
        SETTINGS['MIMO_coding_matrix'] = f['settings']['MIMO_coding_matrix'].value
        exec('{} = f[{}][{}].value[0,0]'.format(i, "'settings'","i"))
        print(i, ':  ', f['settings'][i].value[0, 0])


D = 1.889e-3
J = 2j*np.pi
E = np.e
LAMBDA = C/SETTINGS['Fc']
FD = D/LAMBDA

# processing setting
save_fig = False
slow_time_cutoff = 16
cmap = plt.cm.jet

print('Data shape: {}'.format(DATA.shape))



## Extended Parameters

range_resolution = C/2/SETTINGS['Fc']
mu = SETTINGS['BW']/(SETTINGS['Chirp_time'] - SETTINGS['DwellTime'] - SETTINGS['Reset_time'])
range_max = 0.5*(C * SETTINGS['Fs'] / 2 / mu)
velocity_max = 0.5 * C / 2 / SETTINGS['Chirp_time'] / SETTINGS['Fc']
unmigrate_velocity = C/2/SETTINGS['BW'] / SETTINGS['Chirp_time'] / SETTINGS['NChirps']
range_velocity_coupling_cof = mu/2/SETTINGS['Fc']/SETTINGS['Fs']
range_angle_coupling_cof = D*mu/SETTINGS['Fs']/C

print('Unambiguous Range is: {:.2f} meter.'.format(range_max))
print('Unambiguous velocity is {:.2f} m/s.'.format(velocity_max))
print('Migration velocity is {:.2f} m/s.'.format(unmigrate_velocity))
print('Range Velocity coupling coefficient is {:.2g}.'.format(range_velocity_coupling_cof))
print('Range Angle coupling coefficient is {:.2g}.'.format(range_angle_coupling_cof))



## Preprocessing

range_fft_number = 2048
doppler_fft_number = 1024
angle_fft_number = 128

# remove dwell time, reset time, and settle time
settle_time = 4e-6
start_point = int(SETTINGS['NSamples']*(SETTINGS['DwellTime'] + settle_time)/SETTINGS['Chirp_time'])+1
end_point = np.int(SETTINGS['NSamples']-SETTINGS['NSamples']*SETTINGS['Reset_time']/SETTINGS['Chirp_time'])-1
data_beat_freq = DATA[:, :, :, start_point:end_point]

# remove direct wave
direct_wave_range = 3  # meter
direct_wave_fft_grids = int(direct_wave_range * range_fft_number/2 / range_max)
range_fft_data = fft(data_beat_freq, n=range_fft_number, axis=-1)[:,:,:, 0:range_fft_number//2]
range_fft_data_cutoff = range_fft_data[:,:,:, direct_wave_fft_grids::]
del range_fft_data




'''
Start to processing
'''

## RD FFT
print('\nFFT processing for range-doppler and range angle....')

frd = fftshift(fft(range_fft_data_cutoff, n=doppler_fft_number, axis=-2), axes=-2)
absolute_frd = 10*log10(abs(frd))
del frd

# plot
plt.figure()
plt.imshow(fliplr(absolute_frd[0,0,:,:]).T, interpolation='nearest', cmap='jet',
           aspect='auto', clim=([np.max(absolute_frd)-30, np.max(absolute_frd)-8]),
           extent=[-velocity_max, velocity_max, direct_wave_range, range_max])
plt.colorbar()

if save_fig:
    plt.savefig('rd_fft {}.jpg'.format(file), dpi=600)
# del absolute_frd


## Calibration
*num_virtue_array, num_slow_time, num_fft_fast_time = range_fft_data_cutoff.shape
virtual_array_data = range_fft_data_cutoff.reshape(12, num_slow_time, num_fft_fast_time)     # data:ttk
correction = np.transpose(np.tile(CAL, [num_slow_time, num_fft_fast_time, 1]), [2,0,1])
calibrated_data = virtual_array_data*correction                            # data:ttk   (12*256*978)


## AR FFT
# reshape and calibration
fad = fftshift(fft(calibrated_data, n=angle_fft_number, axis=0), axes=0)
del virtual_array_data, correction

# plot
absolute_fad = 10*log10(abs(fad))
pic = absolute_fad[:, 0, :]
new_pic = angle_range_transfrom(pic.T, rmin=direct_wave_range, rmax=range_max,
                                angle_min=-pi/2, angle_max=pi/2).sine_to_degree()
fig = plt.figure(figsize=[14.5, 9])
range_angle_polarplot(fig=fig, data=new_pic.T, rmax=range_max, rmin=direct_wave_range).polar_plot(cmap=cmap, title=file)

if save_fig:
    plt.savefig('ar_fft {}.jpg'.format(file), dpi=600)

del pic, absolute_fad, fad


## 2D MUSIC for (angle-Dopple)
slow_time_cutoff = slow_time_cutoff
dm = calibrated_data[:,0:slow_time_cutoff,:].reshape((12*slow_time_cutoff,num_fft_fast_time)) # data: (tt)k

R = dm.dot(dm.T.conj())
eigval, eigvec = eigsort(np.linalg.eig(R))

threshold = mean(eigval)*3
num_targets = sum( eigval >= threshold )
noise_subspace = eigvec[:, 0:12*slow_time_cutoff - num_targets]

angle_music_number = 180
freq_bins_number = 200
angle_scan = linspace(-pi/2, pi/2, angle_music_number)
freq_domain = linspace(-0.5, 0.5, freq_bins_number)
Pmusic = np.zeros((angle_music_number, freq_bins_number))

# 2D MUSIC
print('\n2D MUSIC processing for doppler-angle....')
for i, a in enumerate(angle_scan):
    for j, b in enumerate(freq_domain):
        steering = (E**(-J*FD*sin(a)*np.arange(12)).reshape((12, 1))). \
            dot( (E**(-J*b*np.arange(slow_time_cutoff)).T).reshape((1, slow_time_cutoff)) )
        steering = steering.reshape((1, 12*slow_time_cutoff))
        p = 10*log10(abs( steering.dot(noise_subspace).dot(noise_subspace.T.conj()).dot(steering.T.conj())))
        Pmusic[i,j] = 1/p[0,0]

plt.figure()
plt.imshow(Pmusic.T, extent=[-90, 90, -25, 25], aspect='auto')
plt.colorbar()

# compare
plt.figure()
fav = fftshift(fftn(calibrated_data[:, 0:slow_time_cutoff, 0],
                    s=[angle_fft_number, doppler_fft_number], axes=[0, 1]), axes=[0, 1])
plt.imshow(10*log10(abs(fav)).T, extent=[-90, 90, -25, 25], aspect='auto')
plt.colorbar()




## 1D MUSIC   (searching each range do 1D MUSIC using Doppler as reference dimension)
Pmusic2 = np.zeros((len(angle_scan), num_fft_fast_time))
spectral_norm = np.zeros(num_fft_fast_time, dtype=complex)
print('\n1D MUSIC processing for range-angle....')
tic = time.time()
null_range = 0

music_1d_data = calibrated_data[:, 0:slow_time_cutoff, :]   # ttk
# amplitude_extraction = fft(music_1d_data, axis=1, n=doppler_fft_number)
for i in range(num_fft_fast_time):
#    print(i)
    tmp = music_1d_data[:, :, i]    ## tt
    R = tmp.dot(tmp.T.conj())
    eigval, eigvec = eigsort(np.linalg.eig(R))

    threshold = mean(eigval) * 5
    num_targets = sum(eigval >= threshold)
    if num_targets == 0:
        null_range = null_range+1
        continue
    elif num_targets >= 2:
        print('target number for range{}: {}'.format(i, num_targets))
    else:
        pass
    noise_subspace = eigvec[:, 0:12-num_targets]
    spectral_norm[i] = max(eigval)

    for j, a in enumerate(angle_scan):
        steering =  E**(-J*FD*sin(a)*np.arange(12)).reshape((1, 12))
        p = (abs(steering.dot(noise_subspace).dot(noise_subspace.T.conj()).dot(steering.T.conj())))
        Pmusic2[j, i] = 1/p[0, 0]
    [ta, ti] = [np.max(Pmusic2[:, i]), np.min(Pmusic2[:, i])]
    Pmusic2[:,i] = ((Pmusic2[:, i] - ti + 0.00001)/(ta-ti)) * (spectral_norm[i].real)**(1/2)

print('Time for 1D MUSIC: {:.2f} seconds!'.format(time.time() - tic))
# plot
fig = plt.figure(figsize=[14.5, 9])
levels_polarplot = [0, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
pic = range_angle_polarplot(fig=fig, data=(Pmusic2[:, 0::]), rmax=range_max,
                      rmin=direct_wave_range).polar_plot(levels=None, cmap=cmap, title=file)
if save_fig:
    plt.savefig('1D MUSIC {}.jpg'.format(file), dpi=600)


# ## 2D MUSIC (using doppler beat frequency as reference) ALMOST IMPOSSIBLE
# range_music_number = range_fft_number
# range_scan = linspace(direct_wave_range, range_max, range_music_number)
# Pmusic3 = np.zeros((angle_music_number, range_music_number))


plt.show()