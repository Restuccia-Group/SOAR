import csiread
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

csifile = "../Samples/output43455c0_80MHz.pcap"
csidata = csiread.Nexmon(csifile, chip='43455c0', bw=80)
csidata.read()
csi= np.fft.ifftshift(csidata.csi, axes=None)

non_zero= np.concatenate((np.arange(6,127,1),np.arange(130,251,1)),axis=None)
csi=csi[:, non_zero]

csi_mag = np.abs(csi)
csi_angle = np.unwrap(np.angle(csidata.csi))

fig, ax = plt.subplots(2, sharex=True)
for i in range(100):
    ax[0].plot(csi_mag[i])
    ax[1].plot(csi_angle[i])

plt.show()
np.save('data.npy', csi)