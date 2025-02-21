import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import Point
from shapely import wkt
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import cKDTree
import numpy as np





# import multibeam geodataframe after removing faulty points
"""from multibeam_processing import main
filtered_data =main()"""



def calculate_depth_differences(transformed_gdf):
    """
    Berechnet die Tiefenunterschiede an Punkten, die sich an Kreuzungspunkten befinden,
    basierend auf einem Abstand von weniger als 0,5m und einer Zeitdifferenz von mehr als 5 Minuten.
    Optimiert mit cKDTree für schnellere Nachbarschaftssuche.
    Gibt zusätzlich ein GeoDataFrame mit den verwendeten Punkten zurück.
    """
    # Sicherstellen, dass es ein GeoDataFrame ist
    if not isinstance(transformed_gdf, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf muss ein GeoDataFrame sein!")

    # Maximale Distanz für nahe Punkte
    max_distance = 0.5  # Meter
    min_time_diff = pd.Timedelta(minutes=5)  # Zeitdifferenz von mehr als 5 Minuten
    
    # Konvertiere Zeitspalten im Voraus
    transformed_gdf['DateTime'] = pd.to_datetime(transformed_gdf['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    transformed_gdf['Date'] = transformed_gdf['DateTime'].dt.date
    unique_dates = sorted(transformed_gdf['Date'].unique())
    
    # Initialisiere eine Liste für jeden Datumsvergleich
    depth_diff_dict = {f"{d1}-{d2}": [] for d1, d2 in combinations(unique_dates, 2)}
    used_indices = set()
    
    # Erstelle cKDTree für effiziente nächste Nachbar-Suche
    coords = np.vstack([transformed_gdf.geometry.x, transformed_gdf.geometry.y]).T
    tree = cKDTree(coords)
    
    # Suche Nachbarn innerhalb des max_distance Radius
    indices = tree.query_ball_tree(tree, max_distance)
    
    for idx, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Calculationg depth differences"):
        point = transformed_gdf.iloc[idx]
        valid_matches = []
        
        for neighbor_idx in neighbors:
            if neighbor_idx == idx:
                continue  # Sich selbst überspringen
            
            match = transformed_gdf.iloc[neighbor_idx]
            time_diff = abs(point['DateTime'] - match['DateTime'])
            
            if time_diff > min_time_diff:
                valid_matches.append((match['Date'], abs(point['Depth (m)'] - match['Depth (m)'])))
                used_indices.add(idx)
                used_indices.add(neighbor_idx)
        
        # Fülle das Dictionary mit den berechneten Werten
        for date, depth_diff in valid_matches:
            date_pair = f"{point['Date']}-{date}"
            if date_pair in depth_diff_dict:
                depth_diff_dict[date_pair].append(depth_diff)
    
    # Erstelle ein DataFrame aus dem Dictionary, wobei jede Spalte nur so viele Einträge hat, wie berechnet wurden
    depth_diff_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in depth_diff_dict.items()]))
    
    # Erstelle ein GeoDataFrame mit nur den verwendeten Punkten
    used_points_gdf = transformed_gdf.loc[list(used_indices)].copy()
    
    return depth_diff_df, used_points_gdf




def main():
    data_dir = Path("output/multibeam")
    print("reading data")
    filtered_data_df = pd.read_csv(data_dir/"filtered_data.csv")
    filtered_data_df["geometry"] = filtered_data_df["geometry"].apply(wkt.loads)

    filtered_data= gpd.GeoDataFrame(filtered_data_df, crs='EPSG:25833', geometry="geometry")


    print("Calculating depth difference")
    depth_difference, used_points_gdf= calculate_depth_differences(filtered_data)
    input("saving data")
    data_output_dir = Path("output/multibeam/QC")
    used_points_gdf.to_csv(data_output_dir/"QC_used_points.csv", index=False)

    input("done!")


main()