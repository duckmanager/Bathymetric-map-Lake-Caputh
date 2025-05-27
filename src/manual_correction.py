#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm
from shapely.geometry import Point, LineString

def interactive_error_correction(
        faulty_points_dir: Path,
        filtered_gdf: gpd.GeoDataFrame,
        manual_overwrite: bool = True,
        vb_window_size: int = 3  # Number of VB points before and after sidebeam-point to specify window for local projection on route
    ):
    """
    Manually inspect and remove faulty sonar depth points using interactive plotting.

    Opens an interactive Matplotlib window for each survey run (file_id) to visually inspect and mark erroneous depth points. 
    Faulty points can be selected via single clicks or rectangular selection. Selected points are saved to a CSV and removed 
    from the returned dataset. Previously saved selections are either applied automatically or preloaded into the plots for further editing.

    VB are used to determine the cumulative travel distance (x-axis). Other beams get projected on the nearest route part recorded at roughly the same time.
    If only one beam type is present, or if the 'Beam_type' column is missing, all points are treated as vertical beams (VB), 
    and their position is calculated using cumulative distance along the survey track. No projection is performed in this case.

    args:
        faulty_points_dir: Path - directory containing or receiving "faulty_points.csv", which stores manually selected error points including their original index
        filtered_gdf: GeoDataFrame - sonar depth data already filtered by automated methods (e.g., detect_and_remove_faulty_depths)
        manual_overwrite: bool - determines behavior when a CSV with faulty points already exists:
            True  → interactive plot always opens, existing faulty points are preloaded and editable
            False → if a CSV exists, no plot is shown; points are removed directly based on saved indices

    returns:
        df_corrected: GeoDataFrame - same as input but with all manually selected faulty points removed; includes previously separated artificial boundary points
    """
    
    # Define CSV to store faulty points and copy the input dataframe
    FILTER_CSV = faulty_points_dir / "faulty_points.csv"
    # load main data
    df = filtered_gdf
    # check for column with original index
    if 'orig_index' not in df.columns:
        df['orig_index'] = df.index

    if Path(FILTER_CSV).exists():
        if not manual_overwrite: # if manual overwrite = False -> use faulty indices list for filtering
            removed_points_df = pd.read_csv(FILTER_CSV)
            bad = removed_points_df['orig_index'].tolist()  # get original indices of faulty points
            boundary_points = df[df['file_id'] == "artificial_boundary_points"] # dont filter edge points
            df_corrected = pd.concat([df.drop(bad), boundary_points], ignore_index=True) # filter faulty points 
            print(f"({len(bad)}) marked error points got loaded and removed from data")
            
            return df_corrected
        else:
            # manual overwrite = True -> manual error correction is started with or without existing faulty points
            removed_points_df = pd.read_csv(FILTER_CSV)
            loaded_bad = removed_points_df['orig_index'].tolist()
            print(f"({len(loaded_bad)}) marked error points loaded for manual check")
    else:
        loaded_bad = []

    # Separate edge points from main data
    boundary_points = df[df['file_id'] == "artificial_boundary_points"]
    df = df[df['file_id'] != "artificial_boundary_points"]
    all_bad_indices = []

    # Process each survey (file_id) separately
    for survey_id in tqdm(df['file_id'].unique(), desc="Processing surveys"):
        sub_df = df[df['file_id'] == survey_id].copy()

        # Determine if there's only one beam type or if Beam_type is missing:
        if ('Beam_type' not in sub_df.columns) or (sub_df['Beam_type'].nunique() == 1):
            is_single_beam = True
        else:
            is_single_beam = False

        # In single-beam scenario, treat all points as VB; otherwise, select only VB points
        if is_single_beam:
            vb_df = sub_df.copy()
        else:
            vb_df = sub_df[sub_df['Beam_type'] == "VB"].copy()

        if vb_df.empty:
            print(f"Survey {survey_id}: No points found. Skipping.")
            continue

        # Compute cumulative distances for VB points
        vb_coords = np.column_stack((vb_df['Longitude'], vb_df['Latitude'])) # create Array from long, lat data
        diff = np.diff(vb_coords, axis=0)
        cum_dist = np.r_[0, np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))] # using pythagoras to calculate each distance
        vb_df['cum_dist'] = cum_dist

        pts_all = []  # List to collect tuples: (projected x, depth, original index, color)
        markers = {}  # Dictionary for active markers

        # Prepare plot; use original colors for different beam types
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#000000", "#f25b60", "#59c347", "#4c62f6", "#f0cb49"] # prepare colors

        # Get unique beam types; in one-beam scenario, all points are treated as VB
        beams = sub_df['Beam_type'].unique() if 'Beam_type' in sub_df.columns else ["VB"]

        # Process each beam type (if only one type, projection is just cumulative distance)
        for color, beam in zip(colors, beams):
            beam_df = sub_df if is_single_beam else sub_df[sub_df['Beam_type'] == beam].copy()
            proj_list = [] # list for each calculated x-axis value
            for idx, row in beam_df.iterrows(): # iterating over all points with beam type
                x, y = row['Longitude'], row['Latitude']
                current_pt = Point(x, y) # create shapely point to project on a line later
                # In single-beam case or if all beams are VB use cumulative distance directly.
                if is_single_beam or (('Beam_type' in sub_df.columns) and beam == "VB"):
                    proj_val = vb_df.loc[idx, 'cum_dist']
                else: # For non-VB beams in multi-beam scenario, project onto local VB segment.
                    vb_indices = np.array(vb_df.index) # find position on VB line
                    vb_pos = np.searchsorted(vb_indices, idx)
                    if vb_pos > 0 and vb_indices[vb_pos - 1] == idx:
                        vb_pos -= 1
                    # determine local window of VB-line to project the beams on
                    start = max(0, vb_pos - vb_window_size)
                    end = min(len(vb_df), vb_pos + vb_window_size + 1)
                    seg = vb_df.iloc[start:end]
                    if len(seg) >= 2:
                        local_line = LineString(zip(seg['Longitude'], seg['Latitude'])) # create local linestring 
                        proj_val = seg['cum_dist'].iloc[0] + local_line.project(current_pt) # project point on local linestring
                    else:
                        proj_val = 0
                proj_list.append(proj_val) # save the x-axis location of the projected point
                pts_all.append((proj_val, row['Depth (m)'], idx, color))
            ax.scatter(proj_list, beam_df['Depth (m)'], label=beam, color=color, s=15) # show point in the plot

        # visual plot settings
        ax.set_xlabel("Fahrstrecke (m)")
        ax.set_ylabel("Tiefe (m)")
        if pts_all:
            ax.set_ylim(min(p[1] for p in pts_all) - 0.5, 0)
        ax.legend()
        plt.title(f"Messfahrt: {survey_id}")

        # Pre-mark loaded faulty points
        for p in pts_all:
            if p[2] in loaded_bad: # p[0],p[1] are x- y coords in the plot, p[2] is the original Index
                marker, = ax.plot(p[0], p[1], 'ro', markersize=8)
                markers[p[2]] = marker
        survey_bad_indices = [p[2] for p in pts_all if p[2] in loaded_bad] # create list of faulty points
        points_coords = np.array([[p[0], p[1]] for p in pts_all]) # create array fpr diance caluclation in case of click for selection

        # Single click event: toggle marker
        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None: # return if click is out of the plot area
                return
            click_disp = np.array(ax.transData.transform((event.xdata, event.ydata))) # transform click coordinates into pixel coordinates
            pts_disp = ax.transData.transform(points_coords)
            distances = np.linalg.norm(pts_disp - click_disp, axis=1) # calculate distance of click and plottet points
            i = int(np.argmin(distances)) # smallest distance to next point
            if distances[i] < 10: # if point wihtin 10 pixels
                proj, depth, idx, _ = pts_all[i] 
                if idx in markers: # if point was already selected - delete from markers dict an list of bad indices
                    markers[idx].remove()
                    del markers[idx]
                    if idx in survey_bad_indices:
                        survey_bad_indices.remove(idx)
                else: # if points was not slected
                    marker, = ax.plot(proj, depth, 'ro', markersize=8) # turn point red
                    markers[idx] = marker # add to marked dict
                    survey_bad_indices.append(idx) # save in list of faulty indices
                fig.canvas.draw() # refresh the plot

        # Rectangle selector event: toggle markers for selected points
        def on_select(eclick, erelease): # ecklick contains coordinate of click start - erelease coordinates of cklick release
            x_min, x_max = sorted([eclick.xdata, erelease.xdata]) # extract coordinates of click and release
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])
            for p in pts_all: # contains all points of current survey in (distance, depth, index, originalcolor)
                if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max: # checks if point is withign rectangle
                    if p[2] in markers: # if index of point is already in markers remove selection and remove from list of bad indices
                        markers[p[2]].remove()
                        del markers[p[2]]
                        if p[2] in survey_bad_indices:
                            survey_bad_indices.remove(p[2])
                    else:
                        marker, = ax.plot(p[0], p[1], 'ro', markersize=8) # if not already selected - turn red and
                        markers[p[2]] = marker
                        survey_bad_indices.append(p[2]) # add to faulty indices
            fig.canvas.draw()

        # Connect the click event and create a persistent RectangleSelector
        fig.canvas.mpl_connect('button_press_event', on_click)
        rs = RectangleSelector(ax, on_select, useblit=True, button=[1],
                               minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        plt.show()
        all_bad_indices.extend(survey_bad_indices)

    # Remove marked bad points and re-add boundary points
    df_corrected = pd.concat([df.drop(all_bad_indices), boundary_points], ignore_index=True)
    df.loc[all_bad_indices].to_csv(FILTER_CSV, index=False)
    print(f"{len(all_bad_indices)} points were removed.")
    return df_corrected


#########################################################################################################################################################
#########################################################################################################################################################



def filter_validation_points(com_gdf: gpd.GeoDataFrame, sample_rate: int= 9, create_validation_data:bool=True):
   
    """
    Split dataset into interpolation and validation point-datasets by regular sampling.

    Selects every Xth non-boundary point from the dataset as a validation point to ensure a spatially uniform distribution, while excluding artificial boundary points from both sampling and validation.
    Sample_rate changes the distance between points.

    args:
        com_gdf: GeoDataFrame - full bathymetric dataset including sonar points and artificial boundary points
        sample_rate : int - count of points sorted into validation dataset
        create_validation_data: bool - if set FALSE, sampling of validation data will be skipped, and emtpy dataframes will be returned

    returns:
        gdf_interpol_points: GeoDataFrame - points used for interpolation (excluding boundary and validation points)
        gdf_validation_points: GeoDataFrame - regularly sampled validation points (excluding boundary points)
    """
    
    if create_validation_data==False:
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

    # calculate without boundary points
    mask_boundary = com_gdf["file_id"] == "artificial_boundary_points"
    indices = np.arange(len(com_gdf))
    mask_filter = ~mask_boundary & (indices % sample_rate == 0)

    gdf_validation_points = com_gdf[mask_filter].copy()
    gdf_interpol_points = com_gdf[~mask_filter & ~mask_boundary].copy()

    return gdf_interpol_points, gdf_validation_points