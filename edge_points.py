#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np


def generate_boundary_points(shp_data_dir:Path, point_data_dir:Path, edge_points_zero:bool=False):
    
    """
    Generate depth points around a lake shoreline based on manual measurements or assign zero depth to all edge points.

    Creates evenly spaced (1m) artificial edge points along a lake's polygon boundary and assigns depths by matching to nearby measured edge points.
    Points between measurements (within 150m) receive interpolated depths.
    If no second measurement is close enough, the measured value is extrapolated over 15m.
    If edge_points_zero is True, all boundary points are assigned a depth of 0 without using any measurement or interpolation.
    (See README for more details.)

    Args:
        shp_data_dir (Path): Path to the folder containing the shapefile of the lake (polygon).
        point_data_dir (Path): Path to the folder containing the CSV file with measured edge depths.
        edge_points_zero (bool): If True, skip measurements and assign zero depth to all boundary points.

    Returns:
        boundary_gdf (GeoDataFrame): Boundary points with geometry, UTM coordinates, depth, date, and file identifier ("artificial_boundary_points").
    """


    spacing = 1  # distance between artifical edge points in m (CRS EPSG:25833)
    interpolation_distance = 200  # distance to cover between measured points
    extrapolation_distance = 15  # distance to extrapolate measured depth to the side without measured point within interpolation_distance

    # load lake outline
    shp_files = list(shp_data_dir.glob("*.shp"))
    if len(shp_files) != 1:
        raise ValueError(f"Expected exactly one shapefile in {shp_data_dir}, found {len(shp_files)}")

    lake_boundary = gpd.read_file(shp_files[0]).to_crs("EPSG:25833")
    boundary = lake_boundary.unary_union.exterior

    # load measured edge points - only one csv should be present in the data dir
    csv_files = list(point_data_dir.glob("*.csv"))
    if len(csv_files) != 1:
        raise ValueError(f"Expected exactly one CSV file in {point_data_dir}, but found {len(csv_files)}.")
    edge_points = pd.read_csv(csv_files[0])    
    edge_gdf = gpd.GeoDataFrame(
            edge_points,
            geometry=gpd.points_from_xy(edge_points.Longitude, edge_points.Latitude), # change to Long Lat - always results in errors E N
            crs="EPSG:25833",
        )

    # check for one unique survey-date - necessary to correct the interpolated points for waterlevel changes
    unique_dates = edge_gdf["Date"].unique()
    if len(unique_dates) == 1:
        common_date = unique_dates[0]
    else:
        raise ValueError(f"Error: All edge point measurements must be from the same date to allow for later correction of water level fluctuations. Please fix manually. Dates found: {unique_dates}")
    

    # create equally spaced edge points
    distances = np.arange(0, boundary.length, spacing)
    boundary_points = [boundary.interpolate(d) for d in distances]
    boundary_gdf = gpd.GeoDataFrame(geometry=boundary_points, crs="EPSG:25833")
    boundary_gdf["depth"] = np.nan 

    # set all edge points to 0m if edge_poins_zero = True
    if edge_points_zero:
        boundary_gdf["depth"] = 0  # assign 0 to all
        common_date = "05/17/2009"  # placeholder as no real date is present - wont be needed

        # Rename columns for later merging with sonar data
        boundary_gdf["Longitude"] = boundary_gdf.geometry.x
        boundary_gdf["Latitude"] = boundary_gdf.geometry.y
        boundary_gdf["Depth (m)"] = boundary_gdf["depth"]
        boundary_gdf.drop(columns=["depth"], inplace=True)
        boundary_gdf["file_id"] = "artificial_boundary_points"
        boundary_gdf["Date"] = common_date

        return boundary_gdf

    # assign measured point to closest edge point
    for idx, row in edge_gdf.iterrows():
        nearest_idx = boundary_gdf.geometry.distance(row.geometry).idxmin()
        boundary_gdf.loc[nearest_idx, "depth"] = row["Depth (m)"]

    # Safe indices of edge points with depth assigned from measured points
    known_idxs = boundary_gdf[~boundary_gdf["depth"].isna()].index.to_list()
    n = len(boundary_gdf)

    # Interpolation between known points (cyclic)
    for i in range(len(known_idxs)):
        idx1 = known_idxs[i]
        idx2 = known_idxs[(i + 1) % len(known_idxs)]
        dist = min(abs(idx2 - idx1), n - abs(idx2 - idx1)) * spacing
        # only intperolate if points are within defined interpolation distance
        if dist <= interpolation_distance:
            if idx2 > idx1:
                interp_range = range(idx1 + 1, idx2)
            else:
                interp_range = list(range(idx1 + 1, n)) + list(range(0, idx2))

            for j, interp_idx in enumerate(interp_range, 1):
                frac = j / (len(interp_range) + 1)
                interpolated_depth = (
                    (1 - frac) * boundary_gdf.loc[idx1, "depth"]
                    + frac * boundary_gdf.loc[idx2, "depth"]
                )
                boundary_gdf.loc[interp_idx, "depth"] = interpolated_depth

    # Extrapolation for points without neighbor point in interpoaltion distance
    for idx in known_idxs:
        # Extrapolate "backwards"
        for i in range(1, extrapolation_distance + 1):
            extrap_idx = (idx + i) % n
            if np.isnan(boundary_gdf.loc[extrap_idx, "depth"]):
                boundary_gdf.loc[extrap_idx, "depth"] = boundary_gdf.loc[idx, "depth"]
            else:
                break
        # extrapolate "forwards"
        for i in range(1, extrapolation_distance + 1):
            extrap_idx = (idx - i) % n
            if np.isnan(boundary_gdf.loc[extrap_idx, "depth"]):
                boundary_gdf.loc[extrap_idx, "depth"] = boundary_gdf.loc[idx, "depth"]
            else:
                break

    # Rename columns for later merging with sonar data
    boundary_gdf["Longitude"] = boundary_gdf.geometry.x
    boundary_gdf["Latitude"] = boundary_gdf.geometry.y
    boundary_gdf["Depth (m)"] = boundary_gdf["depth"]
    boundary_gdf.drop(columns=["depth"], inplace=True)
    boundary_gdf["file_id"] = "artificial_boundary_points"
    boundary_gdf["Date"] = common_date

    # remove points without depths assigned
    boundary_gdf = boundary_gdf.dropna(subset=["Depth (m)"]).reset_index(drop=True)

    return boundary_gdf

#########################################################################################################################################################
#########################################################################################################################################################

def combine_multibeam_edge(geodf_projected, boundary_gdf):

    """
    Merge sonar points with artificial lake boundary points.

    Combines georeferenced depth data from sonar beam measurements and interpolated shoreline points into a single GeoDataFrame for unified spatial analysis or export.

    args:
        geodf_projected: GeoDataFrame - multibeam data with position and depth (from create_multibeam_points)
        boundary_gdf: GeoDataFrame - interpolated edge points with depth (from generate_boundary_points or error-filtered version)

    returns:
        gdf_combined: GeoDataFrame - merged point data with geometry, depth, and associated metadata (EPSG:25833)
    """
    
    gdf_combined = gpd.GeoDataFrame(
        pd.concat([geodf_projected, boundary_gdf], ignore_index=True)
    )

    return gdf_combined