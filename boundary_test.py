import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
from pathlib import Path

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


    # check for uniform dates
    unique_dates = edge_gdf["Date"].unique()
    if len(unique_dates) == 1:
        common_date = unique_dates[0]  # save the date
    else:
        print("Error: All edge point measuremtns must be from the same date to allow for later correction of water level fluctuations. Please fix manually")
        print("Found dates:", unique_dates)

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
    boundary_gdf.drop(columns=["depth"], inplace=True)
    boundary_gdf["Date"] = common_date # if fails, the measumrent points used different dates

    return boundary_gdf




def main():
    data_dir = Path("data")
    gdf_complete = generate_boundary_points(data_dir)


    print("saving output")
    output_path = Path("output/multibeam")
    gdf_complete.to_csv(output_path / "newoutline_measured.csv", index=False)
    

main()

# old version
def generate_boundary_points_old(geodf_projected:gpd.GeoDataFrame ,data_dir):
    spacing = 2 # Distance between points in crs-units (crs:25833 - meters)
    
    lake_boundary =gpd.read_file(data_dir/"shp_files"/"cap_see.shp")
    lake_boundary = lake_boundary.to_crs("EPSG:25833")

    # extract boundary as linegeometry
    boundary = lake_boundary.exterior.union_all()

    # create points in a set spacing
    distances = np.arange(0, boundary.length, spacing) # list of point spacings
    points = [boundary.interpolate(dist) for dist in distances]

    """# Prüfen, ob der letzte Punkt mit dem ersten übereinstimmt (geschlossene Form)
if boundary.is_ring and not points[-1].equals(points[0]):
    points.append(points[0])  # Letzten Punkt exakt auf den Startpunkt setzen"""

    df_points = pd.DataFrame([(p.x, p.y) for p in points], columns=["Interpolated_Long", "Interpolated_Lat"])
    df_points['Depth (m)']= 0
    df_points['file_id'] = "artificial_boundary_points"


    gdf_combined = gpd.GeoDataFrame(pd.concat([geodf_projected, df_points], ignore_index=True))
    return gdf_combined