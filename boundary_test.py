import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
from pathlib import Path


import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from scipy.spatial import cKDTree

def generate_boundary_poin(data_dir):
    spacing = 2  # Distance between points in crs-units (crs:25833 - meters)
    
    # Lade Seeumriss und transformiere CRS
    lake_boundary = gpd.read_file(data_dir/"shp_files"/"caplake_outline_final.shp")
    lake_boundary = lake_boundary.to_crs("EPSG:25833")
    
    # Lade die gemessenen Randpunkte mit Tiefenwerten
    edge_points = pd.read_csv(data_dir / "outline" / 'caplake_outpoints_final.csv')
    edge_gdf = gpd.GeoDataFrame(edge_points, geometry=gpd.points_from_xy(edge_points.E, edge_points.N), crs="EPSG:25833")

    # Extrahiere Boundary als LineString
    boundary = lake_boundary.exterior.union_all()

    # Erstelle Punkte entlang der Seegrenze mit festem Abstand
    distances = np.arange(0, boundary.length, spacing)
    boundary_points = [boundary.interpolate(dist) for dist in distances]
    boundary_gdf = gpd.GeoDataFrame(geometry=boundary_points, crs="EPSG:25833")

    # Weise jedem Messpunkt den nächstgelegenen Randpunkt zu
    edge_tree = cKDTree(np.array(list(zip(edge_gdf.geometry.x, edge_gdf.geometry.y))))
    boundary_tree = cKDTree(np.array(list(zip(boundary_gdf.geometry.x, boundary_gdf.geometry.y))))
    _, nearest_boundary_idx = boundary_tree.query(np.array(list(zip(edge_gdf.geometry.x, edge_gdf.geometry.y))))
    
    edge_gdf["nearest_boundary_idx"] = nearest_boundary_idx
    boundary_gdf["depth"] = np.nan

    # Weise den Tiefenwert der Messpunkte zu den nächstgelegenen Randpunkten zu
    for _, row in edge_gdf.iterrows():
        boundary_gdf.at[row["nearest_boundary_idx"], "depth"] = row["Depth (m)"]

    # Verknüpfe aufeinanderfolgende Messpunkte, wenn sie < 150m auseinanderliegen
    edge_gdf = edge_gdf.sort_values(by=["geometry"])
    edge_gdf["next_point"] = edge_gdf["geometry"].shift(-1)
    edge_gdf["next_depth"] = edge_gdf["Depth (m)"].shift(-1)
    edge_gdf["distance_to_next"] = edge_gdf.geometry.distance(edge_gdf["next_point"])

    for _, row in edge_gdf.iterrows():
        if row["distance_to_next"] < 150:
            # Hole Indizes der dazwischenliegenden Randpunkte
            idx1 = row["nearest_boundary_idx"]
            idx2 = boundary_tree.query((row["next_point"].x, row["next_point"].y))[1]
            if idx1 < idx2:
                range_idx = range(idx1, idx2 + 1)
            else:
                range_idx = range(idx2, idx1 + 1)

            # Interpoliere Tiefenwerte
            depth_diff = row["next_depth"] - row["Depth (m)"]
            num_points = len(range_idx)
            depth_step = depth_diff / max(1, num_points - 1)
            
            for i, idx in enumerate(range_idx):
                boundary_gdf.at[idx, "depth"] = row["Depth (m)"] + i * depth_step

    # Extrapolation für 15m, wenn kein weiterer Messpunkt innerhalb von 200m vorhanden ist
    for _, row in edge_gdf.iterrows():
        if pd.isna(row["next_depth"]) or row["distance_to_next"] >= 150:
            start_idx = row["nearest_boundary_idx"]
            for i in range(1, int(15 / spacing) + 1):  # 15m extrapolieren
                if start_idx + i < len(boundary_gdf):
                    boundary_gdf.at[start_idx + i, "depth"] = row["Depth (m)"]

    # Entferne alle Randpunkte ohne zugewiesenen Tiefenwert
    boundary_gdf = boundary_gdf.dropna(subset=["depth"])

    # Transformiere Koordinaten in WGS84 (EPSG:4326) für Longitude und Latitude
    boundary_gdf = boundary_gdf.to_crs("EPSG:25833")
    
    # Extrahiere Koordinaten in separate Spalten
    boundary_gdf["Longitude"] = boundary_gdf.geometry.x
    boundary_gdf["Latitude"] = boundary_gdf.geometry.y
    boundary_gdf["Depth (m)"] = boundary_gdf["depth"]

    # Behalte nur relevante Spalten
    boundary_gdf = boundary_gdf[["Longitude", "Latitude", "Depth (m)"]]

    return boundary_gdf


#new version
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def generate_boundary_points(data_dir):
    spacing = 2  # distamnce between artifical edge points in m (CRS EPSG:25833)
    
    # load lake edge and transform to line-geometry
    lake_boundary = gpd.read_file(data_dir / "shp_files" / "caplake_outline_final.shp").to_crs("EPSG:25833")
    boundary = lake_boundary.unary_union.exterior
    
    # load measured points and transform into gdf
    edge_points = pd.read_csv(data_dir / "outline" / "Capsee_outlinepoints_final.csv")
    edge_gdf = gpd.GeoDataFrame( edge_points, geometry=gpd.points_from_xy(edge_points.E, edge_points.N), crs="EPSG:25833")
    
    # create artifical edge points with equal distances
    distances = np.arange(0, boundary.length, spacing)
    boundary_points = [boundary.interpolate(d) for d in distances]
    boundary_gdf = gpd.GeoDataFrame(geometry=boundary_points, crs="EPSG:25833")
    boundary_gdf["depth"] = np.nan # depth column for better differentiation of interpolation

    # Assigning nearest neighbor points and find nearest edge point to measurments
    # transforming into an array for faster access
    boundary_coords = np.column_stack((boundary_gdf.geometry.x, boundary_gdf.geometry.y))
    edge_coords = np.column_stack((edge_gdf.geometry.x, edge_gdf.geometry.y))
    # find nearest neighbors of each edge point to assign measured depth to nearest edge point
    boundary_tree = cKDTree(boundary_coords)
    _, edge_gdf["nearest_boundary_idx"] = boundary_tree.query(edge_coords)
    
    # assigning measured depths to nearest artifical edge points
    boundary_gdf.loc[edge_gdf["nearest_boundary_idx"], "depth"] = edge_gdf["Depth (m)"].values

    # sort edge points along the edge
    edge_gdf = edge_gdf.sort_values("nearest_boundary_idx").reset_index(drop=True)
    edge_gdf["next_point"] = edge_gdf["geometry"].shift(-1)
    edge_gdf["next_depth"] = edge_gdf["Depth (m)"].shift(-1)
    edge_gdf["distance_to_next"] = edge_gdf.geometry.distance(edge_gdf["next_point"])

    # Interpolation between measurment points if <150 m distance
    for _, row in edge_gdf.iterrows():
        if pd.notna(row["next_depth"]) and row["distance_to_next"] < 150:
            idx1 = row["nearest_boundary_idx"]
            idx2 = boundary_tree.query([row["next_point"].x, row["next_point"].y])[1]
            idx_start, idx_end = min(idx1, idx2), max(idx1, idx2)
            range_idx = range(idx_start, idx_end + 1)
            depth_diff = row["next_depth"] - row["Depth (m)"]
            num_points = len(range_idx)
            depth_step = depth_diff / (num_points - 1) if num_points > 1 else 0
            for i, idx in enumerate(range_idx):
                boundary_gdf.at[idx, "depth"] = row["Depth (m)"] + i * depth_step

    # Extrapolate same depth as measured for 15m if no measurment point is within 150m
    for _, row in edge_gdf.iterrows():
        if pd.isna(row["next_depth"]) or row["distance_to_next"] >= 150:
            start_idx = row["nearest_boundary_idx"]
            for i in range(1, int(15 / spacing) + 1):
                if start_idx + i < len(boundary_gdf): #not necessary?
                    boundary_gdf.at[start_idx + i, "depth"] = row["Depth (m)"]

    # Delete all artifical edge points without assigned depth
    boundary_gdf = boundary_gdf.dropna(subset=["depth"]).copy()
    
    # safe to universal columns for merging with sonar measurments
    boundary_gdf["Longitude"] = boundary_gdf.geometry.x
    boundary_gdf["Latitude"] = boundary_gdf.geometry.y
    boundary_gdf["Depth (m)"] = boundary_gdf["depth"]

    return boundary_gdf[["Longitude", "Latitude", "Depth (m)"]]




def main():
    data_dir = Path("data")
    gdf_complete = generate_boundary_points(data_dir)


    print("saving output")
    output_path = Path("output/multibeam")
    gdf_complete.to_csv(output_path / "newoutline_measured.csv", index=False)
    

main()