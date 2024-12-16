# Read functions for BOWTIE various datasets.
# 
# Soundings - full time series 
# 
# DSHIP ship data
# 
# Radiometer
# 
# Sun photometer
# 
# 
# James Ruppert
# 18 Sept 2024

import numpy as np
import subprocess
import xarray as xr
import pandas as pd


data_main = "/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/macsyrett/dynamo-interp/"

#############################################
### Sounding data
#############################################

def read_dynamo_soundings(search_string = '43599'):

    #### File and time list

    def get_sounding_filelist(search_string):

        # main = "/Volumes/wiss/M203/Radiosondes/level2/"
        main = data_main

        process = subprocess.Popen(['ls --color=none '+main + '*' +search_string+'*nc'],shell=True,
            stdout=subprocess.PIPE,universal_newlines=True)
        snd_files = process.stdout.readlines()
        nsnd=len(snd_files)
        times=[]
        for ifile in range(nsnd):
                snd_files[ifile] = snd_files[ifile].strip()
                time_str = snd_files[ifile].split('/')[-1].split('.')[0]
                if (int(time_str[-12:-10]) >= 10 or int(time_str[-12:-10]) <= 2):
                    # print(time_str)
                    yy = time_str[-16:-12]
                    mm = time_str[-12:-10]
                    dd = time_str[-10:-8]
                    hh = time_str[-7:-5]
                    nn = time_str[-5:-3]
                    sounding_time = np.datetime64(yy+'-'+mm+'-'+dd+'T'+hh+':'+nn)
                    #if search_string == 'ascen':
                    #    # Add 1:10 to all times (assumed time to reach 100 hPa)
                    #    sounding_time += np.timedelta64(70, 'm')
                    times.append(sounding_time)

        return snd_files, np.array(times)

    #### Add dummy time steps for big jumps in time

    #### Main variable read loop

    def read_soundings(files, times):

        # Arrays to save variables
        nz=3100
        nt = len(times)
        dims = (nt, nz)
        p    = np.full(dims, np.nan)
        tmpk = np.full(dims, np.nan)
        rh   = np.full(dims, np.nan)
        #mr   = np.full(dims, np.nan)
        wdir = np.full(dims, np.nan)
        u    = np.full(dims, np.nan)
        v    = np.full(dims, np.nan)
        hght_0c = np.full(nt, np.nan)

        # Get height

        sndfile = xr.open_dataset(files[0], engine='netcdf4')
        hght = np.squeeze(sndfile['hght'].data) # m
        sndfile.close()

        for ifile in range(nt):

            try:
                sndfile = xr.open_dataset(files[ifile])
                p[ifile,:]    = np.squeeze(sndfile['P'].data)    # Pa
                tmpk[ifile,:] = np.squeeze(sndfile['T'].data)   # K
                rh[ifile,:]   = np.squeeze(sndfile['RH'].data)*1e2 # 0-1 --> %
                #mr[ifile,:]   = np.squeeze(sndfile['mr'].data)   # kg/kg
                wdir[ifile,:] = np.squeeze(sndfile['WDIR'].data) # deg
                u[ifile,:]    = np.squeeze(sndfile['U'].data)    # m/s
                v[ifile,:]    = np.squeeze(sndfile['V'].data)    # m/s
                sndfile.close()
                hght_0c[ifile]= hght[ np.where(tmpk[ifile,:] <= 273.15)[0][0] ]
            except:
                print("Failed to read ",files[ifile].split('/')[-1])
                # Will leave failed read time steps as NaN
                continue
        sounding = {
            'hght':hght,
            'hght_0c':hght_0c,
            'p': p,
            'tmpk': tmpk,
            'rh': rh,
            #'mr': mr,
            'u': u,
            'v': v,
            'wdir': wdir,
        }
        
        return sounding

    #### Call the functions

    # Read list of sounding files
    snd_files, times = get_sounding_filelist(search_string=search_string)

    # Adds NaN columns into time and sounding arrays where there are gaps > 3 h
    # so that time-height plots properly show gaps

    # Read soundings into "fixed" time array
    # Provides sounding dataset as a dictionary
    soundings = read_soundings(snd_files, times)

    return soundings, snd_files, times

#############################################