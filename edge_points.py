#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

def generate_boundary_points(data_dir):
    
    """
    Generate depth points around a lake shoreline based on manual measurments.

    Creates evenly spaced (1m) artificial edge points along a lake's polygon boundary and assigns depths by matching to nearby measured edge points.
    Points between measuremnts (within 150m) get depths assigned by interpoaltion. 
    If no interpoaltion is within reach the same depth gets extrapolated for a short distance (15m).
    (more details in Readme)
    
    args:
        data_dir: Path - path to the folder including the shapefile of the lake and CSV file with measured edge depths

    returns:
        boundary_gdf: GeoDataFrame - boundary points with geometry, UTM coordinates, depth, date, and file identifier ("artificial_boundary_points")
    """

    spacing = 1  # distance between artifical edge points in m (CRS EPSG:25833)
    interpolation_distance = 150  # distance to cover between measured points
    extrapolation_distance = 15  # distance to extrapolate measured depth to the side without measured point within interpolation_distance

    # load lake edge and transform to line-geometry
    lake_boundary = gpd.read_file(data_dir / "shp_files" / "waterbody.shp").to_crs(
        "EPSG:25833"
    )
    boundary = lake_boundary.unary_union.exterior

    # load measured points and transform into gdf
    edge_points = pd.read_csv(
        data_dir / "outline" / "measured_edgepoints.csv" # oder "measured_edgepoints.csv"
    )  # change name and N E - change in readme
    edge_gdf = gpd.GeoDataFrame(
        edge_points,
        geometry=gpd.points_from_xy(edge_points.E, edge_points.N),
        crs="EPSG:25833",
    )

    # check for uniform dates
    unique_dates = edge_gdf["Date"].unique()
    if len(unique_dates) == 1:
        common_date = unique_dates[0]  # save the date
    else:
        print(
            "Error: All edge point measurements must be from the same date to allow for later correction of water level fluctuations. Please fix manually"
        )
        print("Found dates:", unique_dates)

    # create artifical edge points with equal distances
    distances = np.arange(0, boundary.length, spacing)
    boundary_points = [boundary.interpolate(d) for d in distances]
    boundary_gdf = gpd.GeoDataFrame(geometry=boundary_points, crs="EPSG:25833")
    boundary_gdf["depth"] = (
        np.nan
    )  # depth column for better differentiation of interpolation

    # Assigning nearest neighbor points and find nearest edge point to measurments
    # transforming into an array for faster access
    boundary_coords = np.column_stack(
        (boundary_gdf.geometry.x, boundary_gdf.geometry.y)
    )
    edge_coords = np.column_stack((edge_gdf.geometry.x, edge_gdf.geometry.y))
    # find nearest neighbors of each edge point to assign measured depth to nearest edge point
    boundary_tree = cKDTree(boundary_coords)
    _, edge_gdf["nearest_boundary_idx"] = boundary_tree.query(edge_coords)

    # assigning measured depths to nearest artifical edge points
    boundary_gdf.loc[edge_gdf["nearest_boundary_idx"], "depth"] = edge_gdf[
        "Depth (m)"
    ].values

    # sort edge points along the edge
    edge_gdf = edge_gdf.sort_values("nearest_boundary_idx").reset_index(drop=True)
    edge_gdf["next_point"] = edge_gdf["geometry"].shift(-1)
    edge_gdf["next_depth"] = edge_gdf["Depth (m)"].shift(-1)
    edge_gdf["distance_to_next"] = edge_gdf.geometry.distance(edge_gdf["next_point"])

    # Interpolation between measurment points if <interpoaltion_distance m distance
    for _, row in edge_gdf.iterrows():
        if (
            pd.notna(row["next_depth"])
            and row["distance_to_next"] < interpolation_distance
        ):
            idx1 = row["nearest_boundary_idx"]
            idx2 = boundary_tree.query([row["next_point"].x, row["next_point"].y])[1]
            idx_start, idx_end = min(idx1, idx2), max(idx1, idx2)
            range_idx = range(idx_start, idx_end + 1)
            depth_diff = row["next_depth"] - row["Depth (m)"]
            num_points = len(range_idx)
            depth_step = depth_diff / (num_points - 1) if num_points > 1 else 0
            for i, idx in enumerate(range_idx):
                boundary_gdf.at[idx, "depth"] = row["Depth (m)"] + i * depth_step

    # Berechne den Abstand zum vorherigen Messpunkt (cyclic)
    edge_gdf["prev_point"] = edge_gdf["geometry"].shift(1)
    edge_gdf["prev_depth"] = edge_gdf["Depth (m)"].shift(1)
    edge_gdf["distance_to_prev"] = edge_gdf.geometry.distance(edge_gdf["prev_point"])

    # Extrapolation entlang der Seeumrisslinie (zyklisch) in beide Richtungen

    num_extrap_points = int(extrapolation_distance / spacing)

    for _, row in edge_gdf.iterrows():
        if pd.isna(row["nearest_boundary_idx"]):
            continue
        idx = int(row["nearest_boundary_idx"])
        depth_value = row["Depth (m)"]

        # Extrapolation in forwards dircetion:
        # Condition: no measured points within >=interpolation_distance m
        if (
            pd.isna(row["next_depth"])
            or row["distance_to_next"] >= interpolation_distance
        ):
            for i in range(1, num_extrap_points + 1):
                forward_idx = (idx + i) % len(boundary_gdf)
                # Fülle nur, wenn noch kein Wert gesetzt wurde
                if pd.isna(boundary_gdf.at[forward_idx, "depth"]):
                    boundary_gdf.at[forward_idx, "depth"] = depth_value
                else:
                    break  # Stoppe, wenn bereits ein Wert existiert

        # Extrapolate in backwards direction:
        # Condition: no measured points within >=interpolation_distance m
        if (
            pd.isna(row["prev_depth"])
            or row["distance_to_prev"] >= interpolation_distance
        ):
            for i in range(1, num_extrap_points + 1):
                backward_idx = (idx - i) % len(boundary_gdf)
                if pd.isna(boundary_gdf.at[backward_idx, "depth"]):
                    boundary_gdf.at[backward_idx, "depth"] = depth_value
                else:
                    break

    # Delete all artifical edge points without assigned depth
    boundary_gdf = boundary_gdf.dropna(subset=["depth"]).copy()

    # safe to universal columns for merging with sonar measurments
    boundary_gdf["Longitude"] = boundary_gdf.geometry.x
    boundary_gdf["Latitude"] = boundary_gdf.geometry.y
    boundary_gdf["Depth (m)"] = boundary_gdf["depth"]
    boundary_gdf.drop(columns=["depth"], inplace=True)
    boundary_gdf["file_id"] = "artificial_boundary_points"
    boundary_gdf["Date"] = (
        common_date  # if fails, the measumrent points used multiple different dates
    )

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