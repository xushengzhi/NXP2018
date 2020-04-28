import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft, fft2, fftn
from scipy.io import loadmat
import h5py
from scipy.constants import speed_of_light as C
import scipy as sp

d = 1.889e-3
J = 2j*np.pi
E = np.e


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

## Load parameters

settings = {}
with h5py.File('RadarSettings.mat') as f:
    for i in list(f['settings'].keys()):

        settings[i] = f['settings'][i].value[0, 0]
        settings['MIMO_coding_matrix'] = f['settings']['MIMO_coding_matrix'].value
        exec('{} = f[{}][{}].value[0,0]'.format(i, "'settings'","i"))
        print(i, ':  ', f['settings'][i].value[0, 0])

sampling_frequency = settings['Fs']/settings['DecimationRate']
start_time_point = np.ceil(settings['DwellTime']*settings['NSamples']/ settings['Chirp_time']) + 3
end_time_point = np.ceil(settings['Reset_time']*settings['NSamples'] / settings['Chirp_time']) + 3
settle_time_point = np.ceil(settings['NSamples'] * 0.1)
start_point = int(start_time_point + settle_time_point)
end_point = int(settings['NSamples'] - end_time_point)

Lam = C / settings['Fc']


## Load data
path = '1050_1150.mat'
data = loadmat(path)['data']
newd = np.transpose(data, (4, 3, 2, 1, 0))
del data
Dat = newd.reshape((12, 256, 512, 100))
del newd
uD = Dat[:, 0:64, start_point:end_point, 50:51]
del Dat

# 0 Ele     1 Slow      2 Fast      3 Frame



## Doa-range FFT
ax = [0, 2]
s = [128, 1024]
fd = fftshift(fftn(uD, axes=ax, s=s), axes=ax)
plt.figure()
nftd = normalizer(10*np.log10(np.squeeze(abs(fd[:, 1, :]))))[:, 0:512]
plt.imshow(nftd.T, interpolation='nearest',
           cmap='jet', aspect='auto', clim=[-20, -0])
plt.colorbar()

## range-Doppler FFT
ax = [1, 2]
s = [256, 1024]

fd = fftshift(fftn(uD, axes=ax, s=s), axes=ax)

plt.figure()
nftd = normalizer(10*np.log10(np.squeeze(abs(fd[1, :, :]))))[:, 100:512]
plt.imshow(nftd.T, interpolation=None,
           cmap='jet', aspect='auto', clim=[-30, -0])
plt.colorbar()


## range FFT
rd = fftshift(fft(uD, axis=2, n=2048), axes=2)
plt.figure()
plt.plot(abs(rd[0, 0, 0:1024, 0]))


## (Range FFT ->) Doppler-DOA FFT
pic = rd[:, :, 784, 0]
fp = fftshift(fft2(np.squeeze(pic), s=[128, 256]))
plt.figure()
plt.imshow(normalizer(np.log10(abs(fp))))


## (Range FFT ->) MUSIC -> DOA
angle_scan = np.linspace(-np.pi/4, np.pi/4, 120)
l = np.arange(12)

R = pic.dot(pic.T.conj())
eigval, eigvec = eigsort(np.linalg.eig(R))
noise_subspace = eigvec[:, 0:11]
pmusic = np.zeros(len(angle_scan))

for i, a in enumerate(angle_scan):
    steer = E**(-J * l * d / Lam * np.sin(a)).reshape((1,12))
    p = steer.dot(noise_subspace).dot(noise_subspace.T.conj()).dot(steer.T.conj())
    pmusic[i] = 10*np.log10(1/abs(p[0, 0]))


plt.figure()
plt.plot(np.linspace(-45, 45, len(angle_scan)), pmusic)


## Range-DOA map (Doppler as reference)

Prd = np.zeros((1024, len(angle_scan)))
for i in range(1024):
    print(i)
    da = rd[:, :, i, 0]
    if abs(da[0,0]) >=0.1:

        R = da.dot(da.T.conj())
        eigval, eigvec = eigsort(np.linalg.eig(R))
        noise_subspace = eigvec[:, 0:11]
        for j, a in enumerate(angle_scan):
            steer = E ** (-J * l * d / Lam * np.sin(a)).reshape((1, 12))
            p = steer.dot(noise_subspace).dot(noise_subspace.T.conj()).dot(steer.T.conj())
            Prd[i, j] = 10 * np.log10(1 / abs(p[0, 0]))
    else:
        Prd[i, :] = -10.7

plt.figure()
plt.imshow(Prd, interpolation='nearest', aspect='auto')


plt.show()