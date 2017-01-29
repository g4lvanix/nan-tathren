#!/usr/bin/env python3

import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

# physical data
c = 330 # propagation speed in m/s
# some configuration
fs = 48000
Ts = 1/fs
# chirp parameters
tstart = 0
tstop = 0.2
fstart = 0
fstop = 20e3
maxsamp = 3*fs
# echo parameters
echo1_delay = int(0.5*fs)
echo2_delay = int(0.52*fs)
echo1_amp = 0.1
echo2_amp = 0.1

# generate frequency chirp signal
tchp = np.linspace(tstart,tstop,tstop*fs)
chp = scipy.signal.chirp(tchp,fstart,tstop,fstop)

# generate TX and RX signals
txsig = np.concatenate((chp,np.zeros(int(maxsamp-chp.size))))
t = np.arange(txsig.size)*Ts

rxsig = echo1_amp*np.roll(txsig,echo1_delay)
rxsig += echo2_amp*np.roll(txsig,echo2_delay)

# apply AWGN to RX signal
rxsig = rxsig #+ np.random.randn(rxsig.size)/np.sqrt(2)

# cross correlation of chirp with RX signal yields the echo location in the
# time series data even in noisy conditions
# this is the application of a matched filter
txsig_fft = np.fft.rfft(txsig)
rxsig_fft = np.fft.rfft(rxsig)
corr_fft = rxsig_fft/txsig_fft
corr = np.fft.irfft(corr_fft)

#corr = np.correlate(rxsig,chp,mode='same')
tcorr = np.arange(corr.size)*Ts

# analyze cross correlation data
peaks = np.array(scipy.signal.argrelmax(np.abs(corr),order=int(chp.size/4)))
times = (peaks-int(chp.size/2))*Ts / 2
distances = c*times

print(peaks,times,distances)

# write the resulting signals to wave files for demonstration
#scipy.io.wavfile.write('chp.wav',fs,np.array(chp,dtype='float32'))
#scipy.io.wavfile.write('rxsig.wav',fs,np.array(rxsig,dtype='float32'))

plt.figure()
plt.subplot(211)
plt.hold(True)
plt.plot(np.abs(rxsig_fft))
plt.plot(np.abs(txsig_fft))
plt.grid()
plt.subplot(212)
plt.plot(np.abs(corr_fft))
plt.grid()

# and some plotting
plt.figure()
plt.subplot(211)
plt.plot(t,rxsig,t,txsig)
plt.grid()
plt.subplot(212)
plt.plot(tcorr,np.abs(corr))
plt.grid()
plt.show()
