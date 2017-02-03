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

rxsig_clean = echo1_amp*np.roll(txsig,echo1_delay)
rxsig_clean += echo2_amp*np.roll(txsig,echo2_delay)

# apply AWGN to RX signal
rxsig = rxsig_clean + np.random.randn(rxsig_clean.size)/np.sqrt(2)

# cross correlation of chirp with RX signal yields the echo location in the
# time series data even in noisy conditions
# this is the application of a matched filter
# in this case an FFT is used as it is much easier to make use of the convolution
# property of the DFT for a large number of samples
# as it turns out scipy already has a function that does this
# also take the absolute value as we're only interested in finding peaks
corr = np.abs(scipy.signal.fftconvolve(rxsig,chp[::-1],mode="valid"))
#corr = np.abs(np.convolve(rxsig,chp[::-1],mode='valid'))

# this idea came from an ESO website: http://www.eso.org/projects/dfs/papers/jitter99/node7.html
# first we calculate the mean and (estimated) standard deviation and then clip
# the cross correlations signal so that only signals with a deviation greater
# than K times the standard deviation remain, i.e.:
# for sample in corr: if sample < m+K*s: sample = 0
m = np.average(corr)
s = np.std(corr,ddof=1)
K = 6
peaks = (corr - (m+K*s)).clip(min=0).nonzero()[0]

# analyze cross correlation data
distances = c*Ts / 2 * peaks
print(distances)

highlight = np.zeros(corr.size)
highlight[peaks] = corr[peaks]

# write the resulting signals to wave files for demonstration
#scipy.io.wavfile.write('chp.wav',fs,np.array(chp,dtype='float32'))
#scipy.io.wavfile.write('rxsig.wav',fs,np.array(rxsig,dtype='float32'))

# this is only needed for plotting the cross correlation on the same time scale
# as the TX & RX signals
tcorr = np.arange(corr.size)*Ts
# and some plotting
plt.figure()
plt.subplot(211)
plt.hold(True)
plt.title("Time domain signals")
plt.plot(t,rxsig,label="RX signal + noise")
plt.plot(t,txsig,label="TX signal")
plt.plot(t,rxsig_clean,label="RX signal, clean")
plt.xlabel("Time [s]")
plt.legend()
plt.grid()
plt.subplot(212)
plt.title("Cross correlation signal")
plt.hold(True)
plt.plot(tcorr,corr)
plt.plot(tcorr,highlight,'ro')
plt.xlabel("Time [s]")
plt.grid()
plt.show()
