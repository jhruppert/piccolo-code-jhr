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


#############################################
### Sounding data
#############################################

def read_bowtie_soundings(search_string = 'ascen'):

    #### File and time list

    def get_sounding_filelist(search_string):

        main = "/Volumes/wiss/M203/Radiosondes/level2/"

        process = subprocess.Popen(['ls --color=none '+main+'*'+search_string+'*nc'],shell=True,
            stdout=subprocess.PIPE,universal_newlines=True)
        snd_files = process.stdout.readlines()
        nsnd=len(snd_files)
        times=[]
        for ifile in range(nsnd):
            snd_files[ifile] = snd_files[ifile].strip()
            time_str = snd_files[ifile].split('/')[-1].split('.')[0]
            # print(time_str)
            mm = time_str[-12:-10]
            dd = time_str[-10:-8]
            hh = time_str[-7:-5]
            nn = time_str[-5:-3]
            sounding_time = np.datetime64('2024-'+mm+'-'+dd+'T'+hh+':'+nn)
            if search_string == 'ascen':
                # Add 1:10 to all times (assumed time to reach 100 hPa)
                sounding_time += np.timedelta64(70, 'm')
            times.append(sounding_time)

        return snd_files, np.array(times)

    #### Add dummy time steps for big jumps in time

    def fix_time_3hrly(times_in, files_in):

        time_0 = np.datetime64('2024-08-14T21:00')
        time_1 = np.datetime64('2024-10-05T00:00')
        delta_3h = np.timedelta64(180, 'm')
        times_new = np.arange(time_0, time_1, delta_3h, dtype='datetime64[m]')
        ntime = len(times_new)
        # Assumes the sounding time is within one hour of 3h time stamp
        delta_check = np.timedelta64(60, 'm')

        files_new = []
        ntime_new=0
        for itime in range(ntime):
            tdiff = np.abs(times_in - times_new[itime])
            tdiff_min = tdiff.min()
            if tdiff_min <= delta_check:
                file_itime = np.where(tdiff == tdiff.min())[0][0]
                files_new.append(files_in[file_itime])
                ntime_new=itime+1
            else:
                files_new.append('null')
        times_new=times_new[:ntime_new]
        files_new=files_new[:ntime_new]

        return files_new, times_new

    #### Main variable read loop

    def read_soundings(files, times):

        # Arrays to save variables
        nz=3100
        nt = len(times)
        dims = (nt, nz)
        p    = np.full(dims, np.nan)
        tmpk = np.full(dims, np.nan)
        rh   = np.full(dims, np.nan)
        mr   = np.full(dims, np.nan)
        wdir = np.full(dims, np.nan)
        u    = np.full(dims, np.nan)
        v    = np.full(dims, np.nan)
        hght_0c = np.full(nt, np.nan)

        # Get height
        sndfile = xr.open_dataset(files[0])
        hght = np.squeeze(sndfile['alt'].data) # m
        sndfile.close()

        for ifile in range(nt):

            try:
                sndfile = xr.open_dataset(files[ifile])
                p[ifile,:]    = np.squeeze(sndfile['p'].data)    # Pa
                tmpk[ifile,:] = np.squeeze(sndfile['ta'].data)   # K
                rh[ifile,:]   = np.squeeze(sndfile['rh'].data)*1e2 # 0-1 --> %
                mr[ifile,:]   = np.squeeze(sndfile['mr'].data)   # kg/kg
                wdir[ifile,:] = np.squeeze(sndfile['wdir'].data) # deg
                u[ifile,:]    = np.squeeze(sndfile['u'].data)    # m/s
                v[ifile,:]    = np.squeeze(sndfile['v'].data)    # m/s
                sndfile.close()
                hght_0c[ifile]= hght[ np.where(tmpk[ifile,:] <= 273.15)[0][0] ]
            except:
                # print("Failed to read ",files[ifile].split('/')[-1])
                # Will leave failed read time steps as NaN
                continue
        sounding = {
            'hght':hght,
            'hght_0c':hght_0c,
            'p': p,
            'tmpk': tmpk,
            'rh': rh,
            'mr': mr,
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
    snd_files, times = fix_time_3hrly(times, snd_files)

    # Read soundings into "fixed" time array
    # Provides sounding dataset as a dictionary
    soundings = read_soundings(snd_files, times)

    return soundings, snd_files, times



#############################################
### Radiometer data
#############################################

def read_bowtie_radiometer():

    main_radiometer = "/Volumes/wiss/M203/Radiometer_MWR-HatPro-Uni-Leipzig/Data/"

    process = subprocess.Popen(['ls --color=none '+main_radiometer+'*/*singl*nc'],shell=True,
        stdout=subprocess.PIPE,universal_newlines=True)
    rdm_files = process.stdout.readlines()
    nfiles=len(rdm_files)
    for ifile in range(nfiles):
        rdm_files[ifile] = rdm_files[ifile].strip()
        rdmfile = xr.open_dataset(rdm_files[ifile])
        rdm_time = rdmfile['time'].data
        cwv = rdmfile['iwv'].data
        flag = rdmfile['iwv_quality_flag'].data
        rdmfile.close()
        rdm_time = np.array(rdm_time, dtype='datetime64[s]')
        cwv = np.array(cwv)
        cwv[np.where(flag != 0)] = np.nan
        if ifile == 0:
            times=rdm_time
            cwv_rdm=cwv
        else:
            times=np.concatenate((times,rdm_time))
            cwv_rdm=np.concatenate((cwv_rdm,cwv))

    return cwv_rdm, times



#############################################
### Sun photometer data
#############################################

# Downloading this data from https://aeronet.gsfc.nasa.gov/new_web/cruises_v3/Meteor_24_0.html

def read_bowtie_sunphotometer():

    main_photometer = "/Volumes/wiss/M203/microtops/downloaded/Meteor_24_0/AOD/Meteor_24_0_all_points.lev15"

    photom = pd.read_csv(main_photometer, sep=',', on_bad_lines='skip', skiprows=4)

    # Get Datetimes from time stamps
    df_datetime = pd.DataFrame({'year': photom['Date(dd:mm:yyyy)'].str[-4:],
                                'month': photom['Date(dd:mm:yyyy)'].str[3:5],
                                'day': photom['Date(dd:mm:yyyy)'].str[0:2],
                                'hour': photom['Time(hh:mm:ss)'].str[0:2],
                                'minute': photom['Time(hh:mm:ss)'].str[3:5],
                                'second': photom['Time(hh:mm:ss)'].str[6:8]})

    photom['Date(dd:mm:yyyy)'] = pd.to_datetime(df_datetime)

    # Sort dataframe
    photom = photom.sort_values('Date(dd:mm:yyyy)')

    # Convert IWV column to float
    photom['Water Vapor(cm)'] = pd.to_numeric(photom['Water Vapor(cm)'], errors='coerce')*10 # cm --> mm

    # photom = pd.read_csv(main_photometer, sep=',', on_bad_lines='skip', skiprows=2)

    # # Get Datetimes from time stamps
    # df_datetime = pd.DataFrame({'year': photom['DATE'].str[-4:],
    #                             'month': photom['DATE'].str[0:2],
    #                             'day': photom['DATE'].str[3:5],
    #                             'hour': photom['TIME'].str[0:2],
    #                             'minute': photom['TIME'].str[3:5],
    #                             'second': photom['TIME'].str[6:8]})
    # photom['DATE'] = pd.to_datetime(df_datetime)
    # # for icol in range(32):
    # #     print(photom.iloc[0:3, icol])

    # # Sort dataframe
    # photom = photom.sort_values('DATE')

    # # Convert IWV column to float
    # photom['WATER'] = pd.to_numeric(photom['WATER'], errors='coerce')*10 # cm --> mm

    return photom



#############################################
### SeaSnake data
#############################################

# Get snake data
def read_snake_files():
    main = "/Volumes/wiss/M203/SeaSnake/seaSnakeData/"
    process = subprocess.Popen(['ls --color=none '+main+'*/*dat'],shell=True,
        stdout=subprocess.PIPE,universal_newlines=True)
    dat_files = process.stdout.readlines()
    # Skip first day
    dat_files=dat_files[3:]
    nfile=len(dat_files)
    frames = []
    for ifile in range(nfile):
        idatfile = dat_files[ifile].strip()
        df = pd.read_csv(idatfile, header=None, sep=',', on_bad_lines='skip')
        frames.append(df)
    # Concatenate
    frames = pd.concat(frames)
    # Convert first row to datetime64
    frames[0] = pd.to_datetime(frames[0])
    # Return concatenated dataframe sorted by time
    return frames.sort_values(0)



#############################################
### DSHIP data
#############################################

def read_dship():

    # file = '/Volumes/wiss/M203/Dship_data/data/meteor_meteo_dship.nc'
    file = '/Volumes/wiss/M203/Dship_data/data/meteor_meteo_dship_20240916.nc'
    # dset=xr.open_dataset(file,engine='h5netcdf',chunks='auto')

    dset=xr.open_dataset(file)
    time = dset.time.data
    sst1 = dset.sst_port.data # Weatherstation
    sst2 = dset.sst_extern_port.data # Thermosalinigraph SBE38(DShip)
    sst3 = dset.sst_intern_port.data # Thermosalinigraph SBE38(DShip)
    wspd = dset.wspd.data # m/s
    shortwave = dset.swr.data
    dset.close()

    dship = {
        'time':time,
        'sst1':sst1,
        'sst2':sst2,
        'sst3':sst3,
        'wspd':wspd,
        'shortwave':shortwave,
        }

    return dship