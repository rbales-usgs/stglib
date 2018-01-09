from __future__ import division, print_function
import pandas as pd
import xarray as xr


def read_par(filnam, spb=False, skiprows=None, skipfooter=0):

    names = ['date', 'time', 'counts']

    par = read_eco_csv(filnam, names, skiprows=skiprows, skipfooter=skipfooter)

    par = eco_pd_to_xr(par, spb=spb)

    return par


def read_ntu(filnam, spb=False, skiprows=None, skipfooter=0):

    names = ['date', 'time', 'a', 'counts', 'b']

    ntu = read_eco_csv(filnam, names, skiprows=skiprows, skipfooter=skipfooter)

    ntu = eco_pd_to_xr(ntu, spb=spb)

    return ntu


def read_eco_csv(filnam, names, skiprows=None, skipfooter=0):

    return pd.read_csv(filnam,
                      sep='\t',
                      names=names,
                      parse_dates=[['date', 'time']],
                      infer_datetime_format=True,
                      engine='python',
                      skiprows=skiprows,
                      skipfooter=skipfooter)


def eco_pd_to_xr(df, spb=False):

    if spb:
        times = df['date_time'].values.reshape((-1, spb))[:, int(spb/2)] # get middle time
        counts = df['counts'].values.reshape((-1, spb))
        sample = range(spb)

        ds = xr.Dataset({'time': ('time', times),
                         'counts': (['time', 'sample'], counts),
                         'sample': ('sample', sample)})
    else:
        times = df['date_time']
        counts = df['counts']

        ds = xr.Dataset({'time': ('time', times),
                         'counts': ('time', counts)})

    return ds