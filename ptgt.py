#+
# Name:
#		Potential Temperature Gradient Tropopause Python Module
# Purpose:
#		This module contains two functions: one to compute the potential temperature gradient
#		tropopause (PTGT) from coincident arrays of temperature, altitude, and pressure; and
#		another to plot the temperature profile with PTGT altitudes superimposed.
# Author and history:
#		Cameron R. Homeyer  2022-06-27.
# Licensing:
#		This code is free to use and redistribute. Please send questions to chomeyer@ou.edu.
#-

# function to calculate tropopause altitude.
# T, z, and p are numpy arrays of temperature (in C or K), z (in km), and p (in hPa), respectively.
def calc_trop(T, z, p, ntrops=1):

    # import Python libraries
    import math
    import numpy as np
    from scipy.interpolate import CubicSpline

    # create array to store tropopause altitude(s)
    trop    = np.zeros(ntrops)
    trop[:] = float('nan')

    # sort profile arrays (in case they are provided unsorted)
    isort = np.argsort(z)
    zsort = z[isort]
    Tsort = T[isort]
    psort = p[isort]

    # search for finite values
    ifin = np.where(np.isfinite(Tsort))
    nfin = len(ifin[0])

    # keep finite points only
    if (nfin > 0):
        zsort = zsort[ifin]
        Tsort = Tsort[ifin]
        psort = psort[ifin]
    else:
        return -1

    # define 100-m altitude array for interpolation and search for altitude layer in data column
    zint = np.arange(0.1,40.1,0.1)
    k0   = ((np.where(zint > zsort[ 0]))[0])[ 0]
    k1   = ((np.where(zint < zsort[-1]))[0])[-1]

    # interpolate temperature (using cubic splines) and pressure to 100-m grid
    index = np.arange(0.0,float(zsort.size))
    iint  = np.interp(zint[k0:k1],zsort,index)
    cs    = CubicSpline(zsort,Tsort)
    Tint  = cs(zint[k0:k1])
    pint  = 10.0**(np.interp(iint,index,np.log10(psort)))
    zint  = zint[k0:k1]

    # get number of altitudes in final profile array
    nz = zint.size

    theta  = ((1000.0/pint)**(287.0/1004.0))*Tint                                        # compute potential temperature

    trop_found = 0   # set flag for tropopause
    itrop      = 0   # set write index for tropopause
    for i in range(0,nz-1):
        if (pint[i] <= 500.0):
            ptg = (theta[i+1] - theta[i]) / (zint[i+1] - zint[i])                        # calculate potential temperature gradient for layer

            if (itrop == 0):
                ptg_thresh = 10.0
            else:
                ptg_thresh = 15.0

            if ((ptg >= ptg_thresh) and (trop_found == 0)):
                k   = np.where((zint > zint[i]) & (zint <= (zint[i] + 2.0)))             # find points within 2 km of current altitude
                k2  = np.where((zint > (zint[i] + 2.0)))                                 # find points more than 2 km above current altitude
                nk  = len(k[0])
                nk2 = len(k2[0])

                if (nk > 0) and (nk2 > 0):
                    ptg2 = (theta[k] - theta[i]) / (zint[k] - zint[i])                   # calculate potential temperature gradient for all altitudes within 2 km of point

                    invalid = np.where((ptg2 < ptg_thresh))                              # search for layers violating potential temperature gradient threshold within 2 km of point
                    if (len(invalid[0]) == 0):
                        trop[itrop] = zint[i]                                            # store tropopause altitude
                        trop_found  = 1                                                  # set flag that a tropopause has been found so the criteria for multiple tropopauses can be enforced
                        itrop      += 1                                                  # increment write index

                        if (itrop == ntrops): return trop                                # if max number of tropopauses reached, return heights
            elif (trop_found == 1):
                k   = np.where((zint > zint[i]) & (zint <= (zint[i] + 1.0)))             # find points within 1 km of current altitude
                k2  = np.where((zint > (zint[i] + 1.0)))                                 # find points more than 1 km above current altitude
                nk  = len(k[0])
                nk2 = len(k2[0])

                if (nk > 0) and (nk2 > 0):
                    ptg2 = (theta[k] - theta[i]) / (zint[k] - zint[i])                   # calculate potential temperature gradients for all altitudes within 1 km of point

                    ipass = np.where((ptg2 < 10.0))                                      # search for potential temperature gradients less than 10 K/km
                    if (len(ipass[0]) == nk): trop_found = 0                             # if all potential temperature gradient layers within 1 km fall below 10 K/km, then reset flag so additional trops can be identified

    return trop


# function to plot temperature profile and tropopause altitude.
# T, z, and p are numpy arrays of temperature (in K), z (in km), and p (in hPa), respectively.
def plot(T,z,ptgt):

    # import Python libraries
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()                                                           # initialize plot
    ax.plot(T,z)                                                                       # draw temperature profile

    for i in range(0,ptgt.size):
        ax.plot([180,320],[ptgt[i],ptgt[i]])                                           # superimpose tropopause altitude(s)

    # modify plot axes
    ax.set(xlim=(180, 320), xticks=np.arange(180, 330, 10), xlabel='Temperature (K)',
           ylim=(0, 20), yticks=np.arange(0, 25, 5), ylabel='Altitude (km)',
           title='Potential Temperature Gradient Tropopause')
    plt.show()                                                                         # reveal plot
