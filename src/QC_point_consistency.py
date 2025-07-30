#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Secondary script to validate the measuring consistency of the processed data from the main skript.
"""
# librarys:
import argparse
import logging
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict



def get_args():
    arg_par = argparse.ArgumentParser()

    #################
    # Options
    #################
    arg_par.add_argument(
        "--matching_radius",
        default=0.2,
        type=float,
        help="Max distance to match points for depth comparison.",
    )

    arg_par.add_argument(
        "--min_time_diff",
        default=pd.Timedelta(minutes=5),
        type=pd.Timedelta,
        help="Min time difference to match points for depth comparison.",
    )

    #################
    # Paths
    #################
    arg_par.add_argument(
        "--data_dir",
        "-dd",
        default=Path.cwd().parent.joinpath("output", "processed_data"),
        type=Path,
        help="Path to folder with the processed dataset.",
    )

    arg_par.add_argument(
        "--output_data_dir",
        "-odd",
        default=Path.cwd().parent.joinpath("output", "QC"),
        type=Path,
        help="Path to folder to store the results of the quality controle.",
    )

    return arg_par.parse_args()
    
    ######################################
    ######################################


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s : %(asctime)s : %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)

    # logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    args.output_data_dir.mkdir(parents=True, exist_ok=True)


#########################################################################################################################################
# processing functions

def calculate_depth_differences_intersections(transformed_gdf, 
    max_distance: float = 0.2,  # meters
    min_time_diff = pd.Timedelta(minutes=5)):  # min time difference between points):
    """
    Calculate depth differences between spatially close and temporally distinct points.

    Identifies all neighboring sonar measurement points within a given spatial range, filters them by a minimal time difference, and calculates depth differences between valid pairs.
    The depth difference is calculated as:  **earlier depth − later depth**. Artificial boundary points (`file_id == "artificial_boundary_points"`) are excluded entirely from the comparison.
    Designed to provide a basis for evaluating survey consistency and detecting potential vertical offsets between repeated measurements in overlapping areas.
    All results are grouped by the corresponding date pairs.

    args:
        transformed_gdf: GeoDataFrame - filtered sonar dataset including geometry, depth, and internal 'Date/Time' field (EPSG:25833)
        max_distance: str - max distance between points (in meters) to be compared
        min_time_diff: pandas.Timedelat - min time difference between points (in min) for them to be compared

    returns:
        depth_diff_df: DataFrame - depth differences grouped by survey date combinations (columns named as 'YYYY-MM-DD-YYYY-MM-DD')
        used_points_gdf: GeoDataFrame - all points used in the pairwise comparisons
    """
    
    # Check for GeoDataFrame
    if not isinstance(transformed_gdf, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf must be GeoDataFrame!")

    # remove all "artifical edge points"
    transformed_gdf = transformed_gdf[transformed_gdf['file_id'] != "artificial_boundary_points"].copy()

    # Check if all data was removed
    if transformed_gdf.empty:
        raise ValueError("No data left after removing artifical edge points!")

    # Convert Date/Time to pdandas datetime - sonar internal Date and time being used, not backed up by GPS time!
    transformed_gdf['DateTime'] = pd.to_datetime(transformed_gdf['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Check for faulty timestamps
    if transformed_gdf['DateTime'].isna().any():
        raise ValueError("Error with Date/Time timestamps! Date/Time is empty or conversion to pd.datetime failed")

    transformed_gdf['Date'] = transformed_gdf['DateTime'].dt.date  # Ectract date for comparison

    # Extract data to numpy array for vectorised calculations
    coords = np.vstack([transformed_gdf.geometry.x, transformed_gdf.geometry.y]).T
    depths = transformed_gdf['Depth (m)'].values
    datetimes = transformed_gdf['DateTime'].values
    dates = transformed_gdf['Date'].values

    # cKDTree for efficent neighbor search
    tree = cKDTree(coords)
    
    # find neighbors in defined distance 
    indices = tree.query_ball_tree(tree, max_distance)

    depth_diff_dict = defaultdict(list)
    used_indices = set()

    # Iterate over all points and calculate depth difference
    for idx, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Calculating depth difference"):
        point_time = datetimes[idx]
        point_date = dates[idx]
        point_depth = depths[idx]
        
        # Don't take difference with itself
        neighbors = [n for n in neighbors if n != idx]

        if not neighbors:
            continue  # skip if no neighbors exist

        # convert neighbors list into numpy array
        neighbor_times = datetimes[neighbors]
        neighbor_dates = dates[neighbors]
        neighbor_depths = depths[neighbors]
        
        # Check which neighbor points were measured later, than center point (date and time), so depth differences wont be doubled (e.g: a-b & b-a) 
        valid_mask = (neighbor_dates > point_date) | ((neighbor_dates == point_date) & (neighbor_times > point_time))
        valid_time_mask = (neighbor_times - point_time) > min_time_diff
        
        final_valid_mask = valid_mask & valid_time_mask
        valid_neighbors = np.array(neighbors)[final_valid_mask]
        valid_depths = neighbor_depths[final_valid_mask]
        valid_times = neighbor_times[final_valid_mask]
        valid_dates = neighbor_dates[final_valid_mask]

        # Save differences with valid order
        for neighbor_idx, match_depth, match_time, match_date in zip(valid_neighbors, valid_depths, valid_times, valid_dates):
            # Calculate in correct order: erlier point - later point
            if match_time > point_time:
                depth_diff = point_depth - match_depth  # later - earlier point
                earlier_date, later_date = point_date, match_date
            else:
                depth_diff = match_depth - point_depth  # later - earlier point
                earlier_date, later_date = match_date, point_date

            
            date_pair = tuple(sorted((earlier_date, later_date)))
            depth_diff_dict[date_pair].append(depth_diff)

            used_indices.add(idx)
            used_indices.add(neighbor_idx)

    # Create Dataframe grouped by measuremnt days
    depth_diff_df = pd.DataFrame(dict([(f"{k[0]}-{k[1]}", pd.Series(v)) for k, v in depth_diff_dict.items()]))

    # create geodataframe with all used points for validation
    used_points_gdf = transformed_gdf.loc[list(used_indices)].copy()

    return depth_diff_df, used_points_gdf


def calculate_depth_differences_close_points(transformed_gdf, max_distance:float=0.2):
    """
    Calculate depth differences between spatially close points, regardless of time difference.

    Identifies neighboring sonar measurement points within a specified spatial distance and computes pairwise depth differences. Unlike intersection-only methods, this includes both consecutive points from the same survey (within spacial distance) and overlapping points from different days. 
    All results are grouped by the corresponding date pairs. The depth difference is always calculated as:  **earlier depth − later depth**.
    Artificial boundary points (`file_id == "artificial_boundary_points"`) are excluded from both the calculations and the output.

    This method is particuallry designed for evaluating local noise or inconsistencies within and between survey passes.
    
    args:
        transformed_gdf: GeoDataFrame - sonar depth data containing coordinates, timestamps, and [Depth (m)] column
        max_distance: float (default: 0.2) - maximum spatial distance (in meters) between two points to be considered neighbors

    returns:
        depth_diff_df: DataFrame - depth differences grouped by date combinations, one column per date pair in format 'YYYY-MM-DD–YYYY-MM-DD'
        used_points_gdf: GeoDataFrame - all points that were used in the pairwise depth comparisons
    """

    
    # Check for GeoDataFrame
    if not isinstance(transformed_gdf, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf must be GeoDataFrame!")


    # remove all "artificial_boundary_points"
    transformed_gdf = transformed_gdf[transformed_gdf['file_id'] != "artificial_boundary_points"].copy()

    # check if data exists after filtering
    if transformed_gdf.empty:
        raise ValueError("Error: No data after removing artifical boundary points!")

    # extract date
    transformed_gdf['DateTime'] = pd.to_datetime(transformed_gdf['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # check for error in timestamps
    if transformed_gdf['DateTime'].isna().any():
        raise ValueError("Error with Date/Time timestamps! Date/Time is empty or conversion to pd.datetime failed")

    transformed_gdf['Date'] = transformed_gdf['DateTime'].dt.date

    # Initialise dicts
    depth_diff_dict = defaultdict(list)
    used_indices = set()

    # extract data to Numpy array for faster processing
    coords = np.vstack([transformed_gdf.geometry.x, transformed_gdf.geometry.y]).T
    depths = transformed_gdf['Depth (m)'].values
    datetimes = transformed_gdf['DateTime'].values
    dates = transformed_gdf['Date'].values

    # cKDTree for efficent neighbor identification
    tree = cKDTree(coords)

    # Look for neighbors in set distance
    indices = tree.query_ball_tree(tree, max_distance)

    # Iterate over all neighbor points to calculate depth difference
    for idx, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Calculating depth difference"):
        point_depth = depths[idx]
        point_date = dates[idx]
        point_time = datetimes[idx]

        # Dont take difference with itself
        neighbors = [n for n in neighbors if n != idx]

        if not neighbors:
            continue  # skip if no neighbor exists

        # Convert list of neighbor points to numpy-array
        neighbor_depths = depths[neighbors]
        neighbor_dates = dates[neighbors]
        neighbor_times = datetimes[neighbors]

        # Calculate difference and save it grouped by survey days
        for neighbor_idx, match_depth, match_time, match_date in zip(neighbors, neighbor_depths, neighbor_times, neighbor_dates):
            # Only calculate earlier point - later point, switch for opposit
            if match_time > point_time:
                depth_diff = point_depth - match_depth  # switch for later - earlier point!
                earlier_date, later_date = point_date, match_date
            else:
                depth_diff = match_depth - point_depth  # switch for later - earlier point!
                earlier_date, later_date = match_date, point_date


            date_pair = tuple(sorted((earlier_date, later_date)))

            # Save by date gorup
            depth_diff_dict[date_pair].append(depth_diff)
            used_indices.add(idx)
            used_indices.add(neighbor_idx)

    # Create dataframe grouped by dates
    depth_diff_df = pd.DataFrame(dict([(f"{k[0]}-{k[1]}", pd.Series(v)) for k, v in depth_diff_dict.items()]))

    # Geodataframe with used points for validation
    used_points_gdf = transformed_gdf.loc[list(used_indices)].copy()

    return depth_diff_df, used_points_gdf



###############################################

def compute_statistics_intersections(depth_diff_df:pd.DataFrame):

    """
    Compute summary statistics and visualize depth differences from intersecting survey lines.

    Calculates the mean and standard deviation of depth differences for each date pair in the input DataFrame and displays the distributions using a boxplot, including point counts above each box.

    args:
        depth_diff_df: DataFrame - depth differences grouped by date combinations, typically from calculate_depth_differences_intersections

    returns:
        stats_df: DataFrame - statistical summary with 'Mean' and 'StdDev' for each date combination
        box: matplotlib object - the generated boxplot object
    """

    # Compute basic statistics
    stats_df = pd.DataFrame({
        'Mean': depth_diff_df.mean(),
        'StdDev': depth_diff_df.std()
    })

    # print(stats_df)

    # Create boxplot figure
    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot([depth_diff_df[col].dropna() for col in depth_diff_df.columns],
                     tick_labels=depth_diff_df.columns, patch_artist=True)

    # Format x-axis labels: from YYYY-MM-DD-YYYY-MM-DD to dd.mm-dd.mm.yy
    def format_label(label):
        parts = label.split('-')
        if len(parts) == 6:
            return f"{parts[2]}.{parts[1]}-{parts[5]}.{parts[4]}.{parts[3][2:]}"
        return label
    new_labels = [format_label(col) for col in depth_diff_df.columns]
    ax.set_xticklabels(new_labels, rotation=45, ha='right')

    # Add horizontal gridlines at every y-tick
    ax.yaxis.grid(True, linestyle='-', color='lightgray', linewidth=0.8)

    # Determine vertical position for annotation text
    y_pos = max(depth_diff_df.max(skipna=True)) * 1.2 if not depth_diff_df.isna().all().all() else 1

    # Annotate each box with n (number of values) and mean
    for i, col in enumerate(depth_diff_df.columns, start=1):
        n_points = depth_diff_df[col].count()
        mean_val = depth_diff_df[col].mean()
        ax.text(i, y_pos, f"n={n_points}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i, y_pos * 0.97, f"Ø={mean_val:.2f}", ha='center', va='top', fontsize=9, color='black')

    # axis label
    ax.set_xlabel("Daten überschneidener Messreihen")  # Overlapping measurement dates
    ax.set_ylabel("Tiefenunterschied (m)")  # Depth difference
    ax.set_title("Tiefenunterschiede überschneidener Messreihen")  # Depth differences of intersecting surveys
    ax.set_ylim(None, y_pos * 1.2)

    plt.tight_layout()
    plt.show()

    return stats_df, fig


def compute_statistics_closepoints(depth_diff_df:pd.DataFrame):
   
    """
    Compute summary statistics and visualize depth differences between spatially close points.

    Calculates the mean and standard deviation of depth differences for each date pair in the input DataFrame and visualizes the distributions using a boxplot, including count labels for the number of point pairs per group.

    args:
        depth_diff_df: DataFrame - depth differences grouped by date combinations, typically from calculate_depth_differences_close_points

    returns:
        stats_df: DataFrame - statistical summary with 'Mean' and 'StdDev' for each date combination
        box: matplotlib object - the generated boxplot object
    """

    # Compute basic statistics
    stats_df = pd.DataFrame({
        'Mean': depth_diff_df.mean(),
        'StdDev': depth_diff_df.std()
    })

    # print(stats_df)

    # Create boxplot figure
    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot([depth_diff_df[col].dropna() for col in depth_diff_df.columns],
                     tick_labels=depth_diff_df.columns, patch_artist=True)

    # Format x-axis labels: from YYYY-MM-DD-YYYY-MM-DD to dd.mm-dd.mm.yy
    def format_label(label):
        parts = label.split('-')
        if len(parts) == 6:
            return f"{parts[2]}.{parts[1]}-{parts[5]}.{parts[4]}.{parts[3][2:]}"
        return label
    new_labels = [format_label(col) for col in depth_diff_df.columns]
    ax.set_xticklabels(new_labels, rotation=45, ha='right')

    # Add horizontal gridlines at every y-tick
    ax.yaxis.grid(True, linestyle='-', color='lightgray', linewidth=0.8)

    # Determine vertical position for annotation text
    y_pos = max(depth_diff_df.max(skipna=True)) * 1.2 if not depth_diff_df.isna().all().all() else 1

    # Annotate each box with n (number of values) and mean
    for i, col in enumerate(depth_diff_df.columns, start=1):
        n_points = depth_diff_df[col].count()
        mean_val = depth_diff_df[col].mean()
        ax.text(i, y_pos, f"n={n_points}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i, y_pos * 0.97, f"Ø={mean_val:.2f}", ha='center', va='top', fontsize=9, color='black')

    # axis titles
    ax.set_xlabel("Daten verglichener Punkte")  # Compared point dates
    ax.set_ylabel("Tiefenunterschied (m)")  # Depth difference
    ax.set_title("Tiefenunterschiede naheliegender Punkte")  # Depth differences of close points
    ax.set_ylim(None, y_pos * 1.2)

    plt.tight_layout()
    plt.show()

    return stats_df, fig



def main():
    data_dir = args.data_dir
    logging.info("reading data")
    filtered_data_df = pd.read_csv(data_dir/"filtered_data.csv")
    filtered_data_df["geometry"] = filtered_data_df["geometry"].apply(wkt.loads)
    filtered_data= gpd.GeoDataFrame(filtered_data_df, crs='EPSG:25833', geometry="geometry")


    logging.info("Calculating depth difference on intersections")
    depth_difference_intersections, used_points_intersections_gdf= calculate_depth_differences_intersections(filtered_data, max_distance=args.matching_radius, min_time_diff= args.min_time_diff)

    logging.info("Calculating depth difference of close points")
    depth_difference_closep, used_points_closep_gdf = calculate_depth_differences_close_points(filtered_data, max_distance=args.matching_radius)

    logging.info("calculating statics")
    logging.info("statics depth difference intersections")
    stats_intersec_df, intersec_plot = compute_statistics_intersections(depth_difference_intersections)
    logging.info("statistics depth diffference close points")
    stats_closep_df, closep_plot = compute_statistics_closepoints(depth_difference_closep)

    logging.info("saving data")
    data_output_dir = args.output_data_dir
    used_points_intersections_gdf.to_csv(data_output_dir/"QC_time_n_space_diff_used_points.csv", index=False)
    used_points_closep_gdf.to_csv(data_output_dir/"QC_space_diff_used_points.csv", index=False)
    intersec_plot.savefig(data_output_dir/ "boxplot_QC_time_n_space_diff.png", dpi=300, bbox_inches="tight")
    closep_plot.savefig(data_output_dir/ "boxplot_QC_space_diff.png", dpi=300, bbox_inches="tight")

    logging.info("The quality assessment is done!")


main()

