# 
# Time series analysis. Various functions for computing power spectra, filtering,
# computing diurnal composites, etc.
# 
# With some nice beta on power spectra from
# https://kls2177.github.io/Climate-and-Geophysical-Data-Analysis, the Github
# page of Karen Smith.
# 
# James Ruppert  
# jruppert@ou.edu  
# 8/19/24

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


# Autocorrelation function
def autocorr(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])


########################################################
# Compute normalized power spectrum
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

# Time step should be in units of 1/(time step) converted to /day

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


########################################################
# Simplify a given (potentially irregular) time series
########################################################

def simplify_time(invar_0, times_invar_0, dt):

    # Puts input variable onto a regular time grid, with NaNs
    # for potential gaps at start and end of array.
    # 
    # CAUTION: if there are gaps in the middle of the time series,
    # they will be filled using this function.
    # 
    # times_invar_0 should be a datetime64 array
    # 
    # dt is time step size as a timedelta64

    invar = np.array(invar_0)
    times_invar = np.array(times_invar_0, dtype='datetime64[ns]')

    # Use timedeltas to determine N steps per day (npd)
    # ...being sure to use consistent timedelta units, i.e., [ns]
    h24 = np.timedelta64(24, 'h').astype('timedelta64[ns]')

    t0 = times_invar[0].astype('datetime64[D]')
    t1 = times_invar[-1].astype('datetime64[D]')
    times_new = np.arange(t0, t1+h24, dt, dtype='datetime64[ns]')

    invar_interp = np.interp(times_new.astype(np.float64), times_invar.astype(np.float64), invar)

    # Set any extrapolated values at the ends of the series to NaN
    invar_interp[np.where(times_new < times_invar[0])] = np.nan
    invar_interp[np.where(times_new > times_invar[-1])] = np.nan

    return invar_interp, times_new


########################################################
# Calculate daily running mean
########################################################

def daily_running_mean(invar_0, times_invar_0):

    invar = np.array(invar_0)
    times_invar = np.array(times_invar_0)
 
    # Use timedeltas to determine N steps per day (npd)
    # ...being sure to use consistent timedelta units, i.e., [ns]
    h24 = np.timedelta64(24, 'h').astype('timedelta64[ns]')
    dt = (times_invar[1] - times_invar[0]).astype('timedelta64[ns]')
    npd = int(np.round(h24 / dt))

    # Create new daily time array with values at day midpoint
    # h12 = np.timedelta64(12, 'h')
    # times_daily_12h = np.arange(times_invar[0]+h12, times_invar[-1]+h12, h24)
    # times_daily_00h = np.arange(times_invar[0], times_invar[-1], h24)
    
    # nd = int(times_invar.size/npd)

    # Simply interpolation to mid-point of day
    # invar_daily2 = np.interp(times_daily_12h.astype(np.float64), times_invar.astype(np.float64), invar)

    # Daily average
    # invar_daily1 = np.full(nd, np.nan)
    # for iday in range(nd):
    #     ind_avg = np.where((times_invar >= times_daily_00h[iday]) & (times_invar < times_daily_00h[iday]+h24))
    #     invar_daily1[iday] = np.nanmean(invar[ind_avg])

    # ROLLING AVERAGE WITH A COSINE WINDOW
    invar_df = pd.DataFrame(invar)
    # invar_daily3 = invar_df.rolling(window=npd, center=True, closed='both', min_periods=int(npd*0.2)).mean()
    # Interpolate over NaNs
    invar_interp = np.squeeze(invar_df.interpolate())
    invar_daily = invar_interp.rolling(window=npd, center=True, closed='both', min_periods=npd,
                                    win_type='cosine').mean()

    # return invar_daily1, invar_daily2, invar_daily3, invar_daily4, times_daily_12h
    return np.squeeze(invar_daily)


########################################################
# Calculate diurnal composite
########################################################

def diurnal_composite(invar_0, times_invar_0, dt, anom=False, days=None, standardize=False):

    # times_invar_0 should be a datetime64 array
    # 
    # dt is time step size as a timedelta64
    # 
    # 'days' should be an array of days to include, at their respective 00:00 H
    # in dtype datetime64[ns].

    invar_1 = np.copy(invar_0)
    # times_invar = np.copy(times_invar_0)

    # Standardize?
    if standardize:
        invar_1 = (invar_1 - np.nanmean(invar_1)) / np.nanstd(invar_1)

    # Simplify time series
    invar_2, times_invar = simplify_time(invar_1, times_invar_0, dt)

    # Remove daily-mean trend using running mean
    # BUT! Save the trend so that you can add back the period-specific mean, if desired
    dm_trend = daily_running_mean(invar_2, times_invar_0)
    invar_2 = invar_2 - dm_trend

    # Use timedeltas to determine N steps per day (npd)
    # ...being sure to use consistent timedelta units, i.e., [ns]
    h24 = np.timedelta64(24, 'h').astype('timedelta64[ns]')
    dt = (times_invar[1] - times_invar[0]).astype('timedelta64[ns]')
    npd = int(np.round(h24 / dt))
    nd = int(invar_2.shape[0]/npd)

    t0 = times_invar[0].astype('datetime64[D]')
    t1 = times_invar[-1].astype('datetime64[D]')
    days_check = np.arange(t0, t1+h24, h24, dtype='datetime64[ns]')
    if days is None:
        days = days_check
    else:
        # Ensure that the provided days don't exceed actual data bounds
        days = days[np.where((days >= t0) & (days <= t1))]
    nd=days.size

    # Add the mean back, if desired
    if not anom:
        ind_select = np.where((times_invar >= days[0]) & (times_invar < days[-1]+h24))[0]
        mean_sav = np.mean(dm_trend[ind_select])
        invar_2+=mean_sav

    invar_select = np.zeros((nd, npd))
    for iday in range(nd):
        ind_select = np.where((times_invar >= days[iday]) & (times_invar < days[iday]+h24))[0]
        invar_select[iday,:] = invar_2[ind_select]

    # Average across days
    var_dc = np.nanmean(invar_select, axis=0)

    # Standardize?
    # Re-standardize
    if standardize:
        var_dc = (var_dc - np.nanmean(var_dc)) / np.nanstd(var_dc)

    # Repeat diurnal cycle in time
    var_dc = np.concatenate((var_dc, var_dc, var_dc[0:1]))

    return var_dc