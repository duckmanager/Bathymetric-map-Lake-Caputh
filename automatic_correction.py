#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def detect_and_remove_faulty_depths(
    geodf_projected: gpd.GeoDataFrame,
    faulty_points_dir:Path,
    max_distance: int = 5,
    threshold: float = 0.5,
    automatic_detection=False     # set True / False if autoamtic correction should be applied - if True existing csv with faulty depths will be overwritten
):

    """
    Identify and remove faulty depth points based on deviation from local neighborhood mean.

    Compares each point's depth to the average depth (not including examined point) of nearby points within a defined radius; points exceeding the threshold difference are flagged and removed, unless marked as artificial boundary points.
    (more details in README)
    args:
        geodf_projected: GeoDataFrame - depth data including geometry and [Depth (m)] (from adjust_depths or create_multibeam_points)
        faulty_points_dir: Path - directory where removed (faulty) points will be saved as "faulty_points.csv"
        max_distance: int - radius (in meters) to consider neighboring points for depth comparison (default: 5)
        threshold: float - depth difference threshold (in meters) beyond which a point is considered faulty (default: 0.5)
        automatic_detection: bool - if True: perform automatic filtering and overwrite faulty_points.csv, if existing; if False: skip automatic detection and return input unchanged

    returns:
        filtered_gdf: GeoDataFrame - data with faulty points removed (same structure as input)
        removed_gdf: GeoDataFrame - all removed points, including original index for traceability; empty if no points removed or detection is disabled
    """

    if not isinstance(geodf_projected, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf must be gdf!")

        # if automatic_detection = False - return gpd unchanged
    if not automatic_detection:
        print("Skipping automatic detection of faulty depths")
        return geodf_projected, gpd.GeoDataFrame(
            columns=geodf_projected.columns
        )  # empty GeoDataFrame

    # save original index in column
    geodf_projected = geodf_projected.copy()
    geodf_projected["orig_index"] = geodf_projected.index

    # Extract relevant data as numpy array
    coords = np.vstack([geodf_projected.geometry.x, geodf_projected.geometry.y]).T
    depths = geodf_projected["Depth (m)"].values

    # create cKDTree for efficient neighbor search
    tree = cKDTree(coords)

    # look on tree for neighbours in set distance - safe indices in list
    indices = tree.query_ball_tree(tree, max_distance)

    # Lists for filtered indices
    valid_indices = []
    removed_indices = []

    for i, neighbors in tqdm(
        enumerate(indices), total=len(indices), desc="Filtering points"
    ):
        if len(neighbors) > 1:  # It must be neighbors
            neighbor_depths = [
                depths[j] for j in neighbors if j != i
            ]  # dont use middlepoint
            if neighbor_depths:  # be sure they are not empty
                mean_depth = np.mean(neighbor_depths)  # calculate mean
                if abs(depths[i] - mean_depth) > threshold:  # check against threshold
                    removed_indices.append(i)
                else:
                    valid_indices.append(i)
        else:
            valid_indices.append(i)

    # check for borderpoints marked as faulty - should be only used as reference, not filtering them
    boundary_indices = [
        i
        for i in removed_indices
        if geodf_projected.iloc[i]["file_id"] == "artificial_boundary_points"
    ]
    removed_indices = [i for i in removed_indices if i not in boundary_indices]
    valid_indices.extend(boundary_indices)

    # Create new gdf with filtered and faulty points
    filtered_gdf = (geodf_projected.iloc[valid_indices].copy()).drop(
        columns="orig_index"
    )
    removed_gdf = geodf_projected.iloc[removed_indices].copy() # change to save as csv

    # save removed points to faulty_points_dir, overwriting existing ones
    if (not removed_gdf.empty):
        removed_gdf.to_csv(faulty_points_dir/"faulty_points.csv", index=False)


    return filtered_gdf, removed_gdf

