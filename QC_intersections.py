import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
from shapely.geometry import Point
from shapely import wkt
from itertools import combinations, product
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict






# import multibeam geodataframe after removing faulty points
"""from multibeam_processing import main
filtered_data =main()"""



"""def load_transformed_gdf(file_path):
    
    df = pd.read_csv(file_path) 
    # Konvertiere zu GeoDataFrame
    if 'geometry' in df.columns:
        df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])  # Falls gespeichert als WKT-String 
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:25833")  # Setze das richtige CRS
    return gdf"""



def calculate_depth_differences(transformed_gdf):
    """
    Berechnet die Tiefenunterschiede an Punkten, die sich an Kreuzungspunkten befinden,
    basierend auf einem Abstand von weniger als 0,5m und einer Zeitdifferenz von mehr als 5 Minuten.
    Berücksichtigt nun auch Punkte mit dem gleichen Aufnahmedatum.
    Optimiert mit cKDTree für schnellere Nachbarschaftssuche.
    Gibt zusätzlich ein GeoDataFrame mit den verwendeten Punkten zurück.
    """
    # check for GeoDataFrame
    if not isinstance(transformed_gdf, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf muss ein GeoDataFrame sein!")

    # max dist between points
    max_distance = 0.5  # meters
    min_time_diff = pd.Timedelta(minutes=5)  # min timedifference between points
    
    # convert time column
    transformed_gdf['DateTime'] = pd.to_datetime(transformed_gdf['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    transformed_gdf['Date'] = transformed_gdf['DateTime'].dt.date
    unique_dates = sorted(transformed_gdf['Date'].unique())
    
    # setup date comparisons - also accepting same day crossings
    date_combinations = set(tuple(sorted((d1, d2))) for d1, d2 in combinations(unique_dates, 2))
    date_combinations.update((d, d) for d in unique_dates)
    
    # Initialising dicts
    depth_diff_dict = defaultdict(list)
    used_indices = set()
    
    # Extract necessary data into numpy array for fast access
    coords = np.vstack([transformed_gdf.geometry.x, transformed_gdf.geometry.y]).T
    depths = transformed_gdf['Depth (m)'].values
    datetimes = transformed_gdf['DateTime'].values # could be changed to use UTC
    dates = transformed_gdf['Date'].values
    
    # create cKDTree for efficient neighbor search
    tree = cKDTree(coords)
    
    # look on tree for neighbours in set distance - safe indices in list
    indices = tree.query_ball_tree(tree, max_distance)

    for idx, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Berechnung der Tiefenunterschiede"):
        point_time = datetimes[idx]
        point_date = dates[idx]
        point_depth = depths[idx]
        
        # Extract time-stamps as numpy-arrays
        neighbor_times = datetimes[neighbors]
        neighbor_dates = dates[neighbors]
        neighbor_depths = depths[neighbors]
        
        # calculate the time differences
        time_diffs = np.abs(neighbor_times - point_time)
        
        # find neighbors with time difference
        valid_neighbor_mask = time_diffs > min_time_diff
        valid_neighbors = np.array(neighbors)[valid_neighbor_mask]
        valid_dates = neighbor_dates[valid_neighbor_mask]
        valid_depths = neighbor_depths[valid_neighbor_mask]
        
        # save valid neighbors
        for match_date, match_depth in zip(valid_dates, valid_depths):
            date_pair = tuple(sorted((point_date, match_date)))
            depth_diff_dict[date_pair].append(abs(point_depth - match_depth))
            used_indices.add(idx)
            used_indices.update(valid_neighbors)
    
    # create dataframe with depth differences from dict with each column starting in the first line
    depth_diff_df = pd.DataFrame(dict([(f"{k[0]}-{k[1]}", pd.Series(v)) for k, v in depth_diff_dict.items()]))
    
    # Create a GeodataFrame with all used points for visual controle
    used_points_gdf = transformed_gdf.loc[list(used_indices)].copy()
    
    return depth_diff_df, used_points_gdf
    
    
    """
    old version:
    for idx, neighbors in tqdm(enumerate(indices), total=len(indices), desc="calculating depth differences"):
        point_time = datetimes[idx]
        point_date = dates[idx]
        point_depth = depths[idx]
        
        valid_matches = []
        
        for neighbor_idx in neighbors:
            if neighbor_idx == idx:
                continue  # skip itself
            
            # checks if time difference specification is met
            match_time = datetimes[neighbor_idx]
            match_date = dates[neighbor_idx]
            match_depth = depths[neighbor_idx]
            
            time_diff = abs(point_time - match_time)
            
            if time_diff > min_time_diff:
                valid_matches.append((match_date, abs(point_depth - match_depth)))
                # saves indices of used points
                used_indices.add(idx)
                used_indices.add(neighbor_idx)
        
        # fill dct with calculated values
        for date, depth_diff in valid_matches:
            date_pair = tuple(sorted((point_date, date)))
            depth_diff_dict[date_pair].append(depth_diff)
    
    # Erstelle ein DataFrame aus dem Dictionary, wobei jede Spalte nur so viele Einträge hat, wie berechnet wurden
    depth_diff_df = pd.DataFrame(dict([(f"{k[0]}-{k[1]}", pd.Series(v)) for k, v in depth_diff_dict.items()]))
    
    # Erstelle ein GeoDataFrame mit nur den verwendeten Punkten
    used_points_gdf = transformed_gdf.loc[list(used_indices)].copy()
    
    return depth_diff_df, used_points_gdf
"""




# mean and std of the depth differences
def compute_statistics(depth_diff_df:pd.DataFrame):
    """
    Berechnet den Durchschnitt und die Standardabweichung jeder Spalte im DataFrame
    und gibt das Ergebnis als neuen DataFrame zurück.
    """
    stats_df = pd.DataFrame({
        'Mean': depth_diff_df.mean(),
        'StdDev': depth_diff_df.std()
    })

    # create a boxplot with the statistics - maybe change to saveing to files later 
    box_plot = depth_diff_df.boxplot()
    plt.show()
    # add tilte, maybe better date format, axis text
    return stats_df, box_plot


def main():
    data_dir = Path("output/multibeam")
    print("reading data")
    filtered_data_df = pd.read_csv(data_dir/"filtered_data.csv")
    filtered_data_df["geometry"] = filtered_data_df["geometry"].apply(wkt.loads)
    filtered_data= gpd.GeoDataFrame(filtered_data_df, crs='EPSG:25833', geometry="geometry")


    print("Calculating depth difference")
    depth_difference, used_points_gdf= calculate_depth_differences(filtered_data)

    print("calculating statics")
    stats_df, box_plot = compute_statistics(depth_difference)
    print("saving data")

    data_output_dir = Path("output/multibeam/QC")
    used_points_gdf.to_csv(data_output_dir/"QC_used_points.csv", index=False)

    input("done!")


main()