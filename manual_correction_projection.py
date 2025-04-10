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
    vb_window_size: int = 3  # Anzahl VB-Punkte nach vorne und hinten für lokale Projektion
):
    FILTER_CSV = faulty_points_dir / "faulty_points.csv"
    df = filtered_gdf.copy()

    if 'orig_index' not in df.columns:
        df['orig_index'] = df.index

    if FILTER_CSV.exists():
        if not manual_overwrite:
            removed_points_df = pd.read_csv(FILTER_CSV)
            bad = removed_points_df['orig_index'].tolist()
            boundary_points = df[df['file_id'] == "artificial_boundary_points"]
            df_corrected = pd.concat([df.drop(bad), boundary_points], ignore_index=True)
            print(f"({len(bad)}) marked error points got loaded and removed from data")
            return df_corrected
        else:
            removed_points_df = pd.read_csv(FILTER_CSV)
            loaded_bad = removed_points_df['orig_index'].tolist()
            print(f"({len(loaded_bad)}) marked error points loaded for manual check")
    else:
        loaded_bad = []

    boundary_points = df[df['file_id'] == "artificial_boundary_points"]
    df = df[df['file_id'] != "artificial_boundary_points"]
    bad_indices = []

    for fid in tqdm(df['file_id'].unique(), desc="Messfahrten"):
        subdf = df[df['file_id'] == fid].copy()
        vb_df = subdf[subdf['Beam_type'] == "VB"].copy()
        if vb_df.empty:
            print(f"Messfahrt {fid}: Keine VB-Punkte gefunden. Überspringe diese Fahrfahrt.")
            continue

        vb_coords = np.column_stack((vb_df['Longitude'], vb_df['Latitude']))
        vb_dist = np.r_[0, np.cumsum(np.sqrt(np.sum(np.diff(vb_coords, axis=0)**2, axis=1)))]
        vb_df['cum_dist'] = vb_dist

        pts_all = []
        fig, ax = plt.subplots(figsize=(10, 6))
        markers = {}
        beams = subdf['Beam_type'].unique()
        colors = [
            "#E69F00",  # Orange
            "#56B4E9",  # Blau
            "#009E73",  # Grün
            "#F0E442",  # Gelb
            "#CC79A7",  # Violett
        ]

        for color, beam in zip(colors, beams):
            beam_df = subdf[subdf['Beam_type'] == beam].copy()
            coords = np.column_stack((beam_df['Longitude'], beam_df['Latitude']))
            proj = []

            for i, (idx, row) in enumerate(beam_df.iterrows()):
                x, y = row['Longitude'], row['Latitude']
                beam_point = Point(x, y)

                if beam == "VB":
                    proj_val = vb_df.loc[idx, 'cum_dist']
                else:
                    if idx in vb_df.index:
                        vb_idx = vb_df.index.get_loc(idx)
                    else:
                        vb_idx = np.searchsorted(vb_df.index, idx)
                        vb_idx = np.clip(vb_idx, 0, len(vb_df) - 1)
                    
                    start = max(0, vb_idx - vb_window_size)
                    end = min(len(vb_df), vb_idx + vb_window_size + 1)
                    segment = vb_df.iloc[start:end]
                    if len(segment) >= 2:
                        local_line = LineString(zip(segment['Longitude'], segment['Latitude']))
                        offset = segment['cum_dist'].iloc[0]
                        proj_val = offset + local_line.project(beam_point)
                    else:
                        proj_val = 0

                proj.append(proj_val)

            depth = beam_df['Depth (m)'].values
            ax.scatter(proj, depth, label=beam, color=color, s=15)
            for d, dep, idx in zip(proj, depth, beam_df.index):
                pts_all.append((d, dep, idx, color))

        ax.set_xlabel("Fahrstrecke (m)")
        ax.set_ylabel("Tiefe (m)")
        if pts_all:
            min_depth = min(p[1] for p in pts_all)
            ax.set_ylim(min_depth - 0.5, 0)
        ax.legend()
        plt.title(f"Messfahrt: {fid}")

        for p in pts_all:
            if p[2] in loaded_bad:
                marker, = ax.plot(p[0], p[1], 'ro', markersize=8)
                markers[p[2]] = marker
                if p[2] not in bad_indices:
                    bad_indices.append(p[2])

        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            click_disp = ax.transData.transform((event.xdata, event.ydata))
            distances = [np.hypot(*(
                click_disp - ax.transData.transform((p[0], p[1]))
            )) for p in pts_all]
            i = int(np.argmin(distances))
            if distances[i] < 10:
                sel = pts_all[i]
                if sel[2] in markers:
                    markers[sel[2]].remove()
                    del markers[sel[2]]
                    if sel[2] in bad_indices:
                        bad_indices.remove(sel[2])
                else:
                    marker, = ax.plot(sel[0], sel[1], 'ro', markersize=5)
                    markers[sel[2]] = marker
                    bad_indices.append(sel[2])
                fig.canvas.draw()

        def onselect(eclick, erelease):
            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])
            for p in pts_all:
                if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                    if p[2] in markers:
                        markers[p[2]].remove()
                        del markers[p[2]]
                        if p[2] in bad_indices:
                            bad_indices.remove(p[2])
                    else:
                        marker, = ax.plot(p[0], p[1], 'ro', markersize=8)
                        markers[p[2]] = marker
                        bad_indices.append(p[2])
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                          button=[1], minspanx=5, minspany=5,
                                          spancoords='pixels', interactive=True)

        plt.show()

    df_corrected = df.drop(bad_indices)
    df_corrected = pd.concat([df_corrected, boundary_points], ignore_index=True)
    df.loc[bad_indices].to_csv(FILTER_CSV, index=False)
    print(f"{len(bad_indices)} points were removed.")
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