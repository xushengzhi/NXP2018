import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import h5py
from pylab import *
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat, savemat


def normalizer(mat):
    max_value = np.max(mat)
    return mat - max_value

    
# Extract data
path = '1001_1200.mat'
arrays = {}
f = h5py.File(path)
for k, v in f.items():
    arrays[k] = np.array(v)

D = arrays['BeatSignals_1001_1200'][50:150,:,:,:,:]
del arrays

# savemat('1050_1150.mat', {'data':D})


# Calibration = np.array([   0.0128 + 0.1034j,
#    0.1178 + 0.1354j,
#   -0.0567 - 0.1022j,
#   -0.0923 - 0.1740j,
#   -0.0012 + 0.0976j,
#    0.0821 + 0.1158j,
#   -0.0213 - 0.1079j,
#   -0.0374 - 0.1577j,
#   -0.0312 - 0.0864j,
#   -0.0753 - 0.1046j,
#    0.0983 + 0.1323j,
#    0.0627 + 0.1936j])



# Animation
fig = plt.figure()
ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

frames = []
for i in range(200):
   data = np.squeeze(D[i, :, :])
   ftd = fftshift(fft2(data, [2048, 1024]))
   nftd = normalizer(10*log10(abs(ftd)))[0:1024, :]
   frames.append(nftd)


cv0 = frames[0]
cf = ax.imshow(cv0)
cb = fig.colorbar(cf, cax=cax)
tx = ax.set_title('Frame 0')

def animate(i):
   arr = frames[i]
   cf = ax.imshow(arr, cmap='jet', interpolation='nearest', clim=[-40, 0])
   cax.cla()
   fig.colorbar(cf, cax=cax, )
   tx.set_text('Frame {0}'.format(i))

ani = animation.FuncAnimation(fig, animate, frames=200,
                             interval=10, repeat_delay=100)

# ani.save('1001_1200.mp4', dpi=100)