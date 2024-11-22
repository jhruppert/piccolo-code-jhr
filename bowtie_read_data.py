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

# data_main = "/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/piccolo-data/data/"
data_main = "./data/"

#############################################
### Sounding data
#############################################

# Mask soundings that don't reach 100 hPa (or other threshold setting)
# Not using actual masking, just setting to NaN
def mask_soundings(soundings, p_threshold=100): # p_threshold should be in hPa
    import copy
    # First save minimum pressure
    nt = soundings['p'].shape[0]
    min_pres = np.zeros(nt)
    for isnd in range(nt):
        min_pres[isnd] = np.nanmin(soundings['p'][isnd,:]*1e-2) # Pa --> hPa
    # Mask out soundings that don't reach 100 hPa
    idx_masked = (min_pres > p_threshold)
    idx_masked = np.repeat(idx_masked[:,np.newaxis], soundings['p'].shape[1], axis=1)
    soundings_masked = copy.deepcopy(soundings)
    for key in soundings_masked.keys():
        if key == 'hght': continue
        elif soundings_masked[key].ndim == 2:
            # soundings[key] = np.ma.masked_where(idx_masked, soundings[key], copy=False)
            soundings_masked[key][np.where(idx_masked)] = np.nan
        else:
            # soundings[key] = np.ma.masked_where(idx_masked[:,0], soundings[key], copy=False)
            soundings_masked[key][np.where(idx_masked[:,0])] = np.nan
    return soundings_masked

#############################################
### Main nested read function
#############################################

def read_bowtie_soundings(search_string = 'ascen'):

    #### File and time list

    def get_sounding_filelist(search_string):

        # main = "/Volumes/wiss/M203/Radiosondes/level2/"
        main = data_main+'soundings/level2/'

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
                # Add 1:10 to all times
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

    # main = "/Volumes/wiss/M203/Radiometer_MWR-HatPro-Uni-Leipzig/Data/"
    main = data_main+'radiometer/'

    process = subprocess.Popen(['ls --color=none '+main+'*/*singl*nc'],shell=True,
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

    # main_photometer = "/Volumes/wiss/M203/microtops/downloaded/Meteor_24_0/AOD/Meteor_24_0_all_points.lev15"
    main = data_main+"microtops/Meteor_24_0/AOD/Meteor_24_0_all_points.lev15"

    photom = pd.read_csv(main, sep=',', on_bad_lines='skip', skiprows=4)

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
def read_bowtie_seasnake():

    # def read_seasnake_raw():
    # main = "/Volumes/wiss/M203/SeaSnake/seaSnakeData/"
    main = data_main+"SeaSnake/seaSnakeData/"

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

    # Convert columns to numeric
    ncolumns = frames.shape[1]
    for icol in range(1,ncolumns):
        frames[icol] = pd.to_numeric(frames[icol], errors='coerce')
    # Take care of some funky lines
    frames.loc[frames[2] > 200,2] = np.nan
    frames.loc[frames[4] > 200,4] = np.nan
    frames.loc[frames[4] < 10,4] = np.nan

    # Return concatenated dataframe sorted by time
    return frames.sort_values(0)

    # def tenmin_avg(snakedat_in):

    #     times_invar = np.array(snakedat_in[0], dtype='datetime64[ns]')
    #     tenmin = np.timedelta64(10, 'm')
    #     t_start_new = np.array(times_invar[0], dtype='datetime64[m]')
    #     top_ind = np.max(np.where(np.isfinite(times_invar)))
    #     t_end_new = np.array(times_invar[top_ind], dtype='datetime64[m]')
    #     times_new = np.arange(t_start_new, t_end_new, tenmin, dtype='datetime64[ns]')

    #     # Interpolate to new time array
    #     data_interp1 = np.interp(times_new.astype(np.float64), times_invar.astype(np.float64), snakedat_in[1])
    #     data_interp2 = np.interp(times_new.astype(np.float64), times_invar.astype(np.float64), snakedat_in[2])
    #     data_interp3 = np.interp(times_new.astype(np.float64), times_invar.astype(np.float64), snakedat_in[3])
    #     data_interp4 = np.interp(times_new.astype(np.float64), times_invar.astype(np.float64), snakedat_in[4])

    #     # Put into dictionary to create new DataFrame
    #     # (this approach is needed to support different dtypes)
    #     frame = {0:times_new,
    #              1:data_interp1,
    #              2:data_interp2,
    #              3:data_interp3,
    #              4:data_interp4,
    #              }

    #     return pd.DataFrame(data=frame)

    # Read raw data
    # snakedat = read_seasnake_raw()
    # Interpolate from one- to ten-minute data
    # snakedat_new = tenmin_avg(snakedat)

    # return snakedat



#############################################
### DSHIP data
#############################################

def read_bowtie_dship():

    file = data_main+'DSHIP/meteor_meteo_dship_20240923.nc'
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



#############################################
### ISAR SeaSkinTemp data
#############################################

# Get snake data
def read_bowtie_ISAR_sst():

    main = data_main+"ISAR_seaskintemp/"

    process = subprocess.Popen(['ls --color=none '+main+'*ISAR*nc'],shell=True,
        stdout=subprocess.PIPE,universal_newlines=True)
    dat_files = process.stdout.readlines()
    nfile=len(dat_files)
    for ifile in range(nfile):
        dat_files[ifile] = dat_files[ifile].strip()
        sstfile = xr.open_dataset(dat_files[ifile])
        sst_time = sstfile['time'].data
        isst = sstfile['sea_surface_temperature'].data # K
        # flag = sstfile['iwv_quality_flag'].data
        sstfile.close()
        sst_time = np.array(sst_time, dtype='datetime64[s]')
        isst = np.array(isst)
        # cwv[np.where(flag != 0)] = np.nan
        if ifile == 0:
            times=sst_time
            sst=isst
        else:
            times=np.concatenate((times,sst_time))
            sst=np.concatenate((sst,isst))

    return sst, times

