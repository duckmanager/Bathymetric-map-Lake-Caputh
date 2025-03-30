#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import shapely.geometry
import numpy as np
from tqdm import tqdm

def create_multibeam_points(sum_df: gpd.GeoDataFrame):

    """
    Create georeferenced points for each sonar beam based on boat heading and beam geometry.

    Computes the horizontal offset for each beam (Beam1–4) from the central sonar location (VB) using fixed beam angles and depth data, incorporating boat heading to correctly position all beams in UTM32N coordinates.

    args:
        sum_df: GeoDataFrame - sonar data with corrected UTC and interpolated positions in UTM zone 33N (EPSG:25833)

    returns:
        transformed_gdf: GeoDataFrame - point geometry and metadata for each beam (VB, Beam1–4), including position, depth, and timestamp
    """

    # convert boat direction (deg) to numeric values
    sum_df["Boat Direction (deg)"] = pd.to_numeric(
        sum_df["Boat Direction (deg)"], errors="coerce"
    )

    # convert Boat direction in Azimuth to radians - (why was that exactly neessary?)
    sum_df["Boat Direction (rad)"] = np.radians(sum_df["Boat Direction (deg)"])

    # Define beam types and ceam correction angles - (insert source for beam angles)
    beams = [
        {"type": "VB", "depth_col": "VB Depth (m)", "angle": np.radians(0)},
        {"type": "Beam1", "depth_col": "BT Beam4 Depth (m)", "angle": np.radians(45)},
        {"type": "Beam2", "depth_col": "BT Beam3 Depth (m)", "angle": np.radians(135)},
        {"type": "Beam3", "depth_col": "BT Beam2 Depth (m)", "angle": np.radians(225)},
        {"type": "Beam4", "depth_col": "BT Beam1 Depth (m)", "angle": np.radians(315)},
    ]

    # empty list for new dataframe
    transformed_data = []

    # Iterate over former depth
    for _, row in tqdm(
        sum_df.iterrows(), total=sum_df.shape[0], desc="Transforming sonar data"
    ):
        base_x, base_y = row["Interpolated_Long"], row["Interpolated_Lat"]
        boat_dir = row["Boat Direction (rad)"]
        file_id, utc, date_time = row["file_id"], row["Utc"], row["Date/Time"]

        for beam in beams:
            beam_type = beam["type"]
            depth = row[beam["depth_col"]]

            if pd.notna(depth):  # only if valid depth exists
                if beam_type == "VB":
                    new_x, new_y = (
                        base_x,
                        base_y,
                    )  # Vertical Beam (VB) stays at original position
                else:  # calculate position of new beams
                    distance = np.tan(np.radians(25)) * float(
                        depth
                    )  # calculate distance to VB
                    azimuth_corrected = boat_dir + beam["angle"]  # absolute angle to VB
                    new_x = base_x + distance * np.sin(
                        azimuth_corrected
                    )  # new beam positions
                    new_y = base_y + distance * np.cos(azimuth_corrected)

                transformed_data.append(
                    {
                        "file_id": file_id,
                        "Utc": utc,
                        "Date/Time": date_time,
                        "Beam_type": beam_type,
                        "Depth (m)": depth,
                        "Longitude": new_x,
                        "Latitude": new_y,
                        "geometry": shapely.geometry.Point(new_x, new_y),
                    }
                )

    # create new dataframe form collected data
    transformed_gdf = gpd.GeoDataFrame(
        transformed_data, geometry="geometry", crs="EPSG:25833"
    )
    return transformed_gdf