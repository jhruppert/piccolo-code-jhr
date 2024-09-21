# 
# Time series analysis. Various functions for computing power spectra and filtering.
# 
# With some nice beta from https://kls2177.github.io/Climate-and-Geophysical-Data-Analysis
# Github page of Karen Smith.
# 
# James Ruppert  
# jruppert@ou.edu  
# 8/19/24

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Autocorrelation function
def autocorr(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

########################################################
# Calculate normalized power spectrum
########################################################

def compute_power_spec(data_in, time_step=1):

    # Standardize
    data = (data_in - np.nanmean(data_in))/np.nanstd(data_in)

    # Interpolate across NaNs
    def interp_nans(data):
        return np.isnan(data), lambda z: z.nonzero()[0]
    nans, x= interp_nans(data)
    data_interp = np.copy(data)
    data_interp[nans] = np.interp(x(nans), x(~nans), data[~nans])

    # Add diurnal signal as test
    # signal = np.sin(np.arange(len(data_interp)) * 2*np.pi*time_step)
    # data_interp += signal

    # Power spectrum
    ps = np.abs(np.fft.fft(data_interp))**2

    freqs = np.fft.fftfreq(data.size, time_step)
    # period = 1/freqs
    idx = np.argsort(freqs)

    # contstruct expected red noise spectrum 
    rspec = []
    T=len(data_interp)
    T2=T/2
    alpha = autocorr(data_interp)[1]
    rspec = (1 - alpha**2) / (1 - 2*alpha*np.cos(np.pi*np.arange(0,T2)/T2) + alpha**2)

    freqs = freqs[idx][len(ps)//2:]
    ps = ps[idx][len(ps)//2:]

    # Calculate significance using F-test
    dof = 2
    fstat = stats.f.ppf(.95,dof,1000)
    spec95 = [fstat*m for m in rspec]
    fstat = stats.f.ppf(.99,dof,1000)
    spec99 = [fstat*m for m in rspec]

    # Normalize spectra
    ps/=np.sum(ps)
    total_rspec = np.sum(rspec)
    rspec/=total_rspec
    spec95/=total_rspec
    spec99/=total_rspec

    pspec = {
        'freqs': freqs,
        'ps': ps,
        'rspec': rspec,
        'spec95': spec95,
        'spec99': spec99,
    }

    return pspec

########################################################
# Plot power spectrum
########################################################

# Time step in units of 

def plot_power_spec(data_in, time_step=1, title=''):

    pspec = compute_power_spec(data_in, time_step=time_step)

    plt.plot(pspec['freqs'], pspec['ps'], '-k', label='Spectrum')
    plt.plot(pspec['freqs'], pspec['rspec'], '-r', label='Red-noise fit')
    plt.plot(pspec['freqs'], pspec['spec95'], '--g', label='95% confidence')
    plt.plot(pspec['freqs'], pspec['spec99'], '--b', label='99% confidence')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel("$f$ [cycles/day]")
    plt.ylabel('Normalized Power')
    # plt.xlim(0,3)#freqs[-1])
    plt.xlim(1e-2,1e2)#freqs[-1])
    plt.ylim(1e-6,1e0)#freqs[-1])
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()

    return