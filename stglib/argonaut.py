import numpy as np
import pandas as pd
import xarray as xr

# def read_argonaut(filbase):
#     return pd.read_csv(
#         filnam,
#         skiprows=skiprows,
#         infer_datetime_format=True,
#         parse_dates=["Date and Time"],
#         encoding=encoding,
#     )


def read_dat_raw(filnam):
    df = pd.read_csv(
        filnam,
        sep=r"\s+",
        parse_dates={"time": ["Year", "Month", "Day", "Hour", "Minute", "Second"]},
        date_format="%Y %m %d %H %M %S",
    )
    df.set_index("time", inplace=True)
    return df


def read_dat(filnam):
    return read_dat_raw(filnam)


def rename_columns(df):
    df.rename(columns="_".join, inplace=True)
    df.columns = df.columns.str.replace("'", "")
    df.rename(
        columns={
            "(__Y__,_ __(_)__)___(__M__,_ __(_)__)___(__D__,_ __(_)__)___(__H__,_ __(_)__)___(__M__,_ __(_)_._1__)___(__S__,_ __(_)__)": "time"
        },
        inplace=True,
    )
    df.columns = df.columns.str.replace(r"\(.*\)", "")

    return df


def read_vel_snr_std(filbase):
    df = pd.read_csv(
        filbase + ".vel",
        sep=r"\s+",
        header=[0, 1],
        parse_dates=[[1, 2, 3, 4, 5, 6]],
        date_format="%Y %m %d %H %M %S",
    )

    df = rename_columns(df)
    df.set_index("time", inplace=True)
    ds = xr.Dataset(df)

    numbins = len([k for k in ds.data_vars if "_Vy" in k])

    # Per argonaut manual, blanking distance is distance to start of first cell, then cell size from then on
    print(filbase + ".ctl")
    with open(filbase + ".ctl") as f:
        for row in f:
            if "BlankDistance" in row:
                blankdistance = float(row[29:])
            elif "CellSize" in row:
                cellsize = float(row[29:])

    bincenters = blankdistance + np.arange(numbins) * cellsize + cellsize / 2

    ds["bindist"] = xr.DataArray(bincenters, dims="bindist")
    # ds['bin'] = xr.DataArray(range(numbins), dims='bindist')

    ds["vx"] = xr.DataArray(
        np.vstack(tuple(ds["Cell%02d_Vx" % (k + 1)] for k in range(numbins))).T,
        dims=("time", "bindist"),
    )
    ds["vy"] = xr.DataArray(
        np.vstack(tuple(ds["Cell%02d_Vy" % (k + 1)] for k in range(numbins))).T,
        dims=("time", "bindist"),
    )
    ds["spd"] = xr.DataArray(
        np.vstack(tuple(ds["Cell%02d_Spd" % (k + 1)] for k in range(numbins))).T,
        dims=("time", "bindist"),
    )
    ds["dir"] = xr.DataArray(
        np.vstack(tuple(ds["Cell%02d_Dir" % (k + 1)] for k in range(numbins))).T,
        dims=("time", "bindist"),
    )
    for v in ds.data_vars:
        if "_Vx" in v or "_Vy" in v or "_Spd" in v or "_Dir" in v:
            ds = ds.drop(v)

    snr = pd.read_csv(
        filbase + ".snr",
        sep=r"\s+",
        header=[0, 1],
        parse_dates=[[1, 2, 3, 4, 5, 6]],
        date_format="%Y %m %d %H %M %S",
    )

    snr = rename_columns(snr)

    for var in ["SNR1", "SNR2"]:
        ds[var.lower()] = xr.DataArray(
            np.vstack(
                tuple(snr["Cell%02d_%s" % (k + 1, var)] for k in range(numbins))
            ).T,
            dims=("time", "bindist"),
        )

    std = pd.read_csv(
        filbase + ".std",
        sep=r"\s+",
        header=[0, 1],
        parse_dates=[[1, 2, 3, 4, 5, 6]],
        date_format="%Y %m %d %H %M %S",
    )

    std = rename_columns(std)

    for var in ["Errx", "Erry"]:
        ds[var.lower()] = xr.DataArray(
            np.vstack(
                tuple(std["Cell%02d_%s" % (k + 1, var)] for k in range(numbins))
            ).T,
            dims=("time", "bindist"),
        )

    dat = read_dat(filbase + ".dat")
    ds["level"] = xr.DataArray(dat["Level"], dims="time")

    return ds


def read_ctl(filnam):
    with open(filnam) as f:
        row = ""
        while "Argonaut ASCII data file Long format is as follows" not in row:
            row = f.readline().rstrip()

        while "Flow data file format is as follows" not in row:
            row = f.readline().rstrip()

        f.close()
