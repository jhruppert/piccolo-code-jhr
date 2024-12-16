import sys
sys.path.append('/home/macsyrett/bowtie-soundings/')
import numpy as np
import metpy.calc as mpcalc
import subprocess, cmocean
import xarray as xr
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from time_series_diag import *
from bowtie_read_data import *
from thermo_functions import *
import math

# Soundings
snd_asc, snd_files, times_asc = read_bowtie_soundings(search_string = 'ascen')
# soundings, snd_files, times_snd = read_bowtie_soundings(search_string = 'descen')
hght = snd_asc['hght']

filename = '/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/dynamo/DYNAMO_43599_L4_5hPa_Sounding_Data.nc'
ds = xr.open_dataset(filename)
alt = ds.alt.data
#print(alt[i])
for i in range(len(ds.sounding)):
#for i in range(0, 1):
    rh = mpcalc.relative_humidity_from_dewpoint(ds.T.data[i]*units.degC, ds.Td.data[i]*units.degC)
    rh = np.interp(hght, alt[i], rh.magnitude)
    ta = np.interp(hght, alt[i], ds['T'][i]+273.15)
    p = np.interp(hght, alt[i], ds['p'][i])
    wdir = np.interp(hght, alt[i], ds['wind_dir'][i])
    u,v = mpcalc.wind_components(ds.wind_spd.data[i]*units('m/s'), ds.wind_dir.data[i]*units.deg)
    u = np.interp(hght, alt[i], u.magnitude)
    v = np.interp(hght, alt[i], v.magnitude)
    release_time = ds['release_time_enc'][i]
    release_date = ds['release_date_enc'][i]

    

    year = math.floor(release_date.data / 10000)
    month = math.floor((release_date.data - year*10000) / 100)
    day = release_date.data - year*10000 - month*100

    hour = math.floor(release_time.data / 10000)
    minute = math.floor((release_time.data - hour*10000) / 100)

    release_time_formatted = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)

    year = str(year)

    if len(str(month)) == 1:
        month = '0' + str(month)
    else: 
        month = str(month)

    if len(str(day)) == 1:
        day = '0' + str(day)
    else:    
        day = str(day)
    
    if len(str(hour)) == 1:
        hour = '0' + str(hour)
    else:
        hour = str(hour)

    if len(str(minute)) == 1:
        minute = '0' + str(minute)
    else:
        minute = str(minute)

    release_time_str = year + month + day + 'T' + hour + minute

    output_ds = xr.Dataset(
        {
            'RH': (['hght'], rh), # Ratio
            'T': (['hght'], ta), # Kelvin
            'P': (['hght'], p), # hPa
            'WDIR': (['hght'], wdir), # Degrees
            'U': (['hght'], u), # m/s
            'V': (['hght'], v), # m/s
            'release_time': release_time_formatted,
        },
        coords = {
            'hght': hght
        }
    )

    output_ds.to_netcdf('/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/macsyrett/dynamo-interp/43599_L4_5hPa' + release_time_str + '_v1.0.1'  + '.nc', mode = 'w', format = 'NETCDF4')