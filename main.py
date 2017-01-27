#!/usr/bin/env python3

import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

# some configuration
fs = 48000
Ts = 1/fs

tstart = 0
tstop = 0.2
fstart = 0
fstop = 20e3

tchp = np.linspace(tstart,tstop,tstop*fs)
chp = scipy.signal.chirp(tchp,fstart,tstop,fstop)

txsig = np.concatenate((chp,np.zeros(2.8*fs)))
rxsig = np.concatenate((np.zeros(0.4*fs),chp))
rxsig.resize(txsig.shape)

# apply AWGN to rx signal
rxsig = rxsig + 5*np.random.randn(rxsig.size)

t = np.linspace(0,3,3*fs)

corr = np.correlate(rxsig,chp,mode='same')
tcorr = np.arange(corr.size)*Ts

plt.subplot(211)
plt.plot(t,rxsig,t,txsig)
plt.grid()
plt.subplot(212)
plt.plot(tcorr,corr)
plt.grid()
plt.show()
