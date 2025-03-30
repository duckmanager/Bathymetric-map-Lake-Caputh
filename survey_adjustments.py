#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import numpy as np



# turn depths negativ - maybe add correction for water fluctuation later
def adjust_depths(com_gdf: gpd.GeoDataFrame):

    """
    Convert all depth values to negative for bathymetric representation.

    Ensures that all depth measurements are expressed as negative values, reflecting their vertical offset below the water surface.
    (more details in Readme)
    args:
        com_gdf: GeoDataFrame - combined bathymetric data from multibeam and boundary points

    returns:
        com_gdf: GeoDataFrame - same as input but with [Depth (m)] as negative values
    """

    # turn depths into negative values
    com_gdf["Depth (m)"] = pd.to_numeric(
        com_gdf["Depth (m)"], errors="coerce"
    ).abs() * (-1)

    return com_gdf


#########################################################################################################################################################
#########################################################################################################################################################


def correct_waterlevel(gdf, data_dir, reference_day=""):
    
    """
    Correct depth values based on daily water level fluctuations.

    Interpolates water levels for measurement dates using a CSV file of waterlevel measurments, and adjusts all depth values relative to the water level on that day.
    The reference day can be chosen manual, otherwise will be chosen automatically.
    (more details in Readme)
    args:
        gdf: GeoDataFrame - depth data (sonar and edge point) with measurement date and depth
        data_dir: Path - data directory containing "waterlevel/waterlevel.csv" with measured daily water levels (format: DD/MM/YYYY)
        reference_day: str (optional) - reference day in MM/DD/YYYY format; if empty, the function selects the best match automatically

    returns:
        gdf: GeoDataFrame - same as input with corrected depth values in [Depth (m)] and original values in [Depth_uncorrected (m)]
    """


    # Extract missing Date-rows from Date/Time
    mask = gdf["Date"].isna()
    if mask.any():
        gdf.loc[mask, "Date"] = pd.to_datetime(
            gdf.loc[mask, "Date/Time"], format="%m/%d/%Y %I:%M:%S %p"
        ).dt.strftime("%m/%d/%Y")

    # transformation of all Date rows to datetime format
    gdf["Date_dt"] = pd.to_datetime(gdf["Date"], format="%m/%d/%Y")

    # 3.load CSV with measured waterlevels (CSV-Date Format: DD/MM/YYYY) and Datetransformation
    wl = pd.read_csv(data_dir / "waterlevel" / "waterlevel.csv", sep=";")
    wl["date_dt"] = pd.to_datetime(wl["date"], dayfirst=True, errors="raise")
    wl = wl.sort_values("date_dt")

    # for interpoaltion: transform dates to numerical values
    wl_ord = wl["date_dt"].map(lambda d: d.toordinal()).values
    wl_vals = wl["waterlevel"].values
    meas_ord = gdf["Date_dt"].map(lambda d: d.toordinal()).values

    # Lineare Interpolation of waterlevels between measuring days - keeping exact matches
    gdf["waterlevel"] = np.interp(meas_ord, wl_ord, wl_vals)

    # determining the reference day
    user_input_reference = False  # flag to save if reference day from user input exists
    if reference_day:
        ref_dt = pd.to_datetime(reference_day, format="%m/%d/%Y")
        user_input_reference = True
    else:
        # Search for first measurment-date with match in waterlevel data
        exact = gdf[gdf["Date_dt"].isin(wl["date_dt"])]
        if not exact.empty:
            ref_dt = exact["Date_dt"].min()
        else:
            # If no exact match found, day closest to measurement date of waterlevels
            uniq = gdf["Date_dt"].unique()
            ref_dt = min(uniq, key=lambda d: np.min(np.abs(wl["date_dt"] - d)))

    # determine waterlevel on reference day by interpolation
    ref_wl = np.interp(ref_dt.toordinal(), wl_ord, wl_vals)

    # save original depth data
    gdf["Depth_uncorrected (m)"] = gdf["Depth (m)"]

    # correct depth data if !=0
    # Formel:corrected = original - (level_measured - level_reference)
    corr = gdf["Depth (m)"].where(
        gdf["Depth (m)"] == 0, gdf["Depth (m)"] - (gdf["waterlevel"] - ref_wl)
    )
    gdf["Depth (m)"] = corr

    # remove temporary columns
    gdf.drop(columns=["Date_dt", "waterlevel"], inplace=True)

    # make sure "Date" gets saved in MM/DD/YYYY format
    gdf["Date"] = pd.to_datetime(gdf["Date"], format="%m/%d/%Y").dt.strftime("%m/%d/%Y")

    # print the reference day and way to determine it
    ref_day_str = ref_dt.strftime("%m/%d/%Y")
    if user_input_reference:
        print(f"Reference day from user input: {ref_day_str}")
    else:
        print(f"Reference day automatically set to: {ref_day_str}")

    return gdf