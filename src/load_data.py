#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from pymatreader import read_mat
from datetime import datetime, timedelta
from tqdm import tqdm


# create dataframe
def create_dataframe(data_dir: Path):
    """
    Establish DataFrame for later data storage

    Creates pandas Dataframe with column count of the longest .sum file. (Feature espacially important for .vel files.)

    args:
        data_dir: Path - path to folder with stored sonar data

    returns:
        sum_dataframe: DataFrame- empty Dataframe with prepared columns
        sum_header: list- list with all sum headers
    """
    # recognize file with the longest header
    longest_file = max(
        data_dir.glob("*.sum"),
        key=lambda f: len(f.read_text(encoding="latin1").splitlines()[0]),
        default=None,
    )

    # create geodataframe with longest header variables + extra columns("file_id", "Utc", "Lat", "Long")
    # XXX: REMOVE ME header = None
    sum_header = longest_file.read_text(encoding="latin1").splitlines()[0].strip().split(",")

    # XXX: REMOVE ME
    # with longest_file.open("r", encoding="latin1") as file:
    #     sum_header = file.readline().strip().split(",")

    # assign additional columns not present in sum files
    additional_columns = ["file_id", "Utc", "Lat", "Long"]

    # create dataframe - maybe assign geometry columns and crs later when converting to geodataframe - or smarter to do now?

    sum_dataframe = pd.DataFrame(columns=additional_columns + sum_header)

    # specify data type - add more if necessary
    sum_dataframe = sum_dataframe.astype({"Depth (m)": float})

    return (sum_dataframe, sum_header)


#########################################################################################################################################################
#########################################################################################################################################################


def assign_data_to_dataframe(
    data_dir: Path, sum_dataframe: pd.DataFrame, sum_header: list
):
    
    """
    Assign sonar and sonar-GPS data from .sum and .mat files to a unified DataFrame.

    Reads all .sum files in the directory, extracts metadata and sample values, corrects UTC time, 
    by using the correct utc-function and using data from corresponding .mat files, and compiles everything into a structured DataFrame.
    
    args:
        data_dir: Path - path to sonar-data
        sum_dataframe: DataFrame - empty DataFrame with predefined columns for sonar and metadata
        sum_header: list - header fields of the .sum files used for column assignment

    returns:
        sum_dataframe: DataFrame - combined DataFrame with all extracted values, corrected UTC time, and file identifiers
    """

    data_list = []

    # Iterate through .sum-files
    for file in data_dir.glob("*.sum"):
        file_id = file.stem  # filename as file_id

        # read same .mat file as .sum file
        mat_file = data_dir / f"{file_id}.mat"
        if mat_file.exists():
            mat_data = read_mat(str(mat_file))
            raw_utc = mat_data.get("GPS", {}).get("Utc", [])
            gps_quality = mat_data.get("GPS", {}).get("GPS_Quality", [])

            # transform to array, if data is no list
            if not isinstance(raw_utc, list):
                raw_utc = raw_utc.flatten()
            if not isinstance(gps_quality, list):
                gps_quality = gps_quality.flatten()

            corrected_utc = correct_utc(raw_utc, gps_quality)
        else:
            print(f"Warning: Mat-file -{file_id}.mat- not found")
            corrected_utc = []

        lines = file.read_text(encoding="latin1").splitlines()
        # XXX: REMOVE ME
        # with file.open("r") as f:
        #     lines = f.readlines()

        # extract data
        for i, line in enumerate(lines[1:]):  # skip header
            values = line.strip().split(",")
            row_dict = dict(zip(sum_header, values))
            row_dict.update(
                {
                    "file_id": file_id,
                    "Utc": corrected_utc[i] if i < len(corrected_utc) else None,
                    "Lat": None,
                    "Long": None,  # gets assigned later
                }
            )
            data_list.append(row_dict)

    # put data into dataframe
    sum_dataframe = pd.DataFrame(data_list, columns=sum_dataframe.columns)

    return sum_dataframe


#########################################################################################################################################################
#########################################################################################################################################################


def correct_utc(utc_list, gps_quality_list):

    """
    Correct faulty UTC timestamps based on GPS quality and temporal continuity.

    Inserted function in assign_data_to_dataframe.
    Builds a continuous 1-second interval timeline starting from the first valid UTC value and replaces unreliable timestamps (UTC = 0 or GPS_Quality = 0) with corresponding values from this idealized timeline.
    (more details in Readme)

    args:
        utc_list: list - raw UTC timestamps extracted from .mat file (format: HHMMSS.x)
        gps_quality_list: list - GPS quality for each sample (0 = invalid, 1–5 = increasing accuracy)

    returns:
        utc_array: list - corrected list of UTC timestamps with unreliable values replaced
    """

    # determine first valid utc-timestamp - should be the first timestamp
    first_valid_utc = next(
        (utc for utc in utc_list if isinstance(utc, (int, float)) and utc != 0), None
    )
    if first_valid_utc is None:
        print("Invalid UTC list!")
        return utc_list

    # Seperate first valid UTC into hours, minutes, seconds and decimals
    utc_str = f"{first_valid_utc:08.1f}"  # Format: HHMMSS.x
    base_time_str = utc_str[:6]  # HHMMSS
    decimal_part = utc_str[7]  # Subseconds

    try:
        start_time = datetime.strptime(base_time_str, "%H%M%S")
    except ValueError as e:
        print(f"Error parsing time: {e}")
        return utc_list

    # Create optimal timeline (list of corrected timeststamps)
    corrected_timestamps = [
        start_time + timedelta(seconds=i) for i in range(len(utc_list))
    ]
    corrected_utc_full = [
        t.strftime("%H%M%S") + f".{decimal_part}" for t in corrected_timestamps
    ]

    # Transform list to numpy array for vectorised matching
    utc_array = np.array(utc_list, dtype=object)
    gps_array = np.array(gps_quality_list, dtype=object)

    # Determine faulty indices with UTC==0 or GPS_Quality==0
    mask = (utc_array == 0) | (gps_array == 0)

    # Substitute UTC of indices with ideal time stamps
    corrected_utc_array = np.array(corrected_utc_full)
    utc_array[mask] = corrected_utc_array[mask]

    return utc_array.tolist()


#########################################################################################################################################################
#########################################################################################################################################################


# load external GPS data and transform to UTM33N
def get_gps_dataframe(data_dir: Path):
   
    """
    Extract and project external GPS data from .txt files into UTM33N.

    Reads consecutive BESTPOSA and GPZDA entries from external GPS logs, pairs positional data with timestamps, and returns a projected GeoDataFrame in UTM zone 33N (EPSG:25833).
    (more details in Readme)
    
    args:
        data_dir: Path - path to folder containing external GPS-data (one file per day) and no other .txt

    returns:
        gps_geodf_projected: GeoDataFrame - projected GPS data with position, timestamp, and accuracy information
    """ 
    
    # prepare empty list
    gps_data_list = []

    # iterate through the GPS files
    for gps_file in tqdm(data_dir.glob("*.txt")):
        gps_data = gps_file.read_text().splitlines()
        bestposa = (None, None)  # Variable to safe lat, long tuple

        for line in gps_data:
            # update when new bestposa is reached
            if line.startswith("#BESTPOSA"):
                bestposa_raw = line.split(",")
                if bestposa_raw[10].replace(".", "").isdigit():
                    bestposa = (
                        bestposa_raw[10],
                        bestposa_raw[11],
                        bestposa_raw[12],
                        bestposa_raw[15],
                        bestposa_raw[16],
                        bestposa_raw[17],
                    )  
                else:
                    bestposa = (
                        bestposa_raw[11],
                        bestposa_raw[12],
                        bestposa_raw[13],
                        bestposa_raw[16],
                        bestposa_raw[17],
                        bestposa_raw[18],
                    )

            # couple Bestposa with next GPZDA line
            elif line.startswith("$GPZDA"):
                try:
                    _, time, day, month, year, *_ = line.split(",")
                    date = (int(day), int(month), int(year))

                    # save current position with time stamp
                    gps_data_list.append(
                        {
                            "date": date,
                            "utc": int(float(time)),
                            "lat": bestposa[0],
                            "long": bestposa[1],
                            "hgt": bestposa[2],
                            "DOP (lat)": float(bestposa[3]),
                            "DOP (lon)": float(bestposa[4]),
                            "VDOP": float(bestposa[5]),
                        }
                    )
                except ValueError as e:
                    print(f"error at {gps_file.name}: {e}")

    # turn into dataframe
    gps_df = pd.DataFrame(gps_data_list)

    # turn into geodataframe to project to UTM33N
    gps_geodf = gpd.GeoDataFrame(
        gps_df, crs="EPSG:4326", geometry=gpd.points_from_xy(gps_df.long, gps_df.lat)
    )

    # project geodataframe to local UTM 33N (epsg:25833)
    gps_geodf_projected = gps_geodf.to_crs(epsg=25833)
    return gps_geodf_projected


# Future improments: allow reading multible files of the same day by sorting by date in advance.

#########################################################################################################################################################
#########################################################################################################################################################


def create_interpolated_coords(sum_df, gps_gdf):
    
    """
    Interpolate high-precision coordinates for sonar data using external GPS positions.

    Matches external-GPS timestamps to sonar timestamps with different decimalseconds by interpolating the coordinates. 
    Applies quality filters, and performs linear interpolation between valid 1-second GPS intervals to estimate sonar positions at sub-second resolution from full second external-gps data.
    (more details in Readme)

    args:
        sum_df: DataFrame - sonar data including corrected UTC timestamps
        gps_gdf: GeoDataFrame - external GPS data with positions (UTM32N), UTC, and DOP quality metrics

    returns:
        sum_df: DataFrame - original sonar data with added interpolated UTM coordinates
        used_gps_gdf: GeoDataFrame - filtered GPS points used for successful interpolation
    """

    # threshold of DOP (Dilution of Precision) in external GPS data, above which GPS points get discarded
    DOP_threshold = float(0.1)  # STD of lat/ lon position deviation in m

    # save original columns of sum_df
    original_columns_sum_df = sum_df.columns.tolist()

    # transform date in sum_df to tuples (format: dd.mm.yy)
    sum_df["date"] = pd.to_datetime(
        sum_df["Date/Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    sum_df["date"] = sum_df["date"].apply(
        lambda x: (x.day, x.month, x.year) if pd.notnull(x) else None
    )

    # check gps_gdf if date is tuple
    gps_gdf["date"] = gps_gdf["date"].apply(lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # transform time in sum_df
    sum_df["utc_float"] = pd.to_numeric(sum_df["Utc"], errors="coerce")
    sum_df["utc_int"] = sum_df["utc_float"].apply(np.floor).astype("Int64")
    sum_df["frac"] = sum_df["utc_float"] - sum_df["utc_int"]  # get decimal seconds

    # save original indices
    gps_gdf = gps_gdf.reset_index().rename(columns={"index": "orig_index"})

    # sort gps_gdf by date and utc
    gps_gdf = gps_gdf.sort_values(["date", "utc"]).reset_index(drop=True)

    # find next point in gps data and unites them in one column in seperate columns
    gps_gdf["geom_x"] = gps_gdf["geometry"].x
    gps_gdf["geom_y"] = gps_gdf["geometry"].y
    gps_gdf["next_geom_x"] = gps_gdf.groupby("date")["geom_x"].shift(
        -1
    )  # go up one second inside of 'date' group
    gps_gdf["next_geom_y"] = gps_gdf.groupby("date")["geom_y"].shift(-1)
    gps_gdf["next_utc"] = gps_gdf.groupby("date")["utc"].shift(-1)
    gps_gdf["next_DOP_lat"] = gps_gdf.groupby("date")["DOP (lat)"].shift(-1)
    gps_gdf["next_DOP_lon"] = gps_gdf.groupby("date")["DOP (lon)"].shift(-1)

    # Merge sum_df with gps_gdf by date and utc
    print("merge GPS and sonar data")
    merged = pd.merge(
        sum_df,
        gps_gdf,
        left_on=["date", "utc_int"],
        right_on=["date", "utc"],
        how="left",
        suffixes=("", "_gps"),
    )

    # output error message for sonar points wihtout gps point
    no_match = merged["utc"].isna()
    if no_match.any():
        print(
            f"Error: {no_match.sum()} entrys have no matching GPS data and got removed."
        )

    # Check if GPS points to interpolate between are exactly one second apart
    correct_timing = merged["next_utc"] == merged["utc"] + 1

    # Check if both points for interpolation meet DOP requirements
    valid = (
        (merged["DOP (lat)"] <= DOP_threshold)
        & (merged["DOP (lon)"] <= DOP_threshold)
        & (merged["next_DOP_lat"] <= DOP_threshold)
        & (merged["next_DOP_lon"] <= DOP_threshold)
        & correct_timing
    )

    # Calculation of interpolated coordinates (in UTM 32N) only using valid points
    # Interpolation by vector between 1sec apart points * decimal second of sonar-point
    # could add skipping of decimal-second=0 but this would prevent efficent vectorisation
    merged["Interpolated_Long"] = np.where(
        valid,
        merged["geom_x"] + merged["frac"] * (merged["next_geom_x"] - merged["geom_x"]),
        None,
    )

    merged["Interpolated_Lat"] = np.where(
        valid,
        merged["geom_y"] + merged["frac"] * (merged["next_geom_y"] - merged["geom_y"]),
        None,
    )

    # Count and print amount of points without interpolated coordinates (not two consecutive seconds or high DOP)
    removed_points = merged["Interpolated_Long"].isna().sum()
    print(f"reomoved points: {removed_points} (not two consecutive samples or low associated GPS-precision).")

    # Drop all rows without interpolated coordinates
    merged = merged.dropna(subset=["Interpolated_Long", "Interpolated_Lat"])

    # only keep original and interpolation columns in sum_df
    sum_df = merged[
        original_columns_sum_df + ["Interpolated_Long", "Interpolated_Lat"]
    ].copy()

    # save all gps points used for interpolation
    used_idx = merged["orig_index"].dropna().unique()
    used_gps_gdf = gps_gdf[gps_gdf["orig_index"].isin(used_idx)].copy()

    # only keep original clumns in gps_gdf
    used_gps_gdf = used_gps_gdf[
        [
            "date",
            "utc",
            "lat",
            "long",
            "hgt",
            "DOP (lat)",
            "DOP (lon)",
            "VDOP",
            "geometry",
        ]
    ]

    return sum_df, used_gps_gdf
