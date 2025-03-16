import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
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



def calculate_depth_differences_intersections(transformed_gdf):
    """
    Berechnet die Tiefenunterschiede an Punkten, die sich an Kreuzungspunkten befinden,
    basierend auf einem Abstand von weniger als 0,5m und einer Zeitdifferenz von mehr als 5 Minuten.
    Die Differenzen werden nach Tagen gruppiert, und es wird sichergestellt, dass immer der spätere
    Wert minus den früheren Wert berechnet wird. Punkte mit file_id == "artificial_boundary_points"
    werden ignoriert.
    """
    
    # Check for GeoDataFrame
    if not isinstance(transformed_gdf, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf muss ein GeoDataFrame sein!")

    # maximal distance and maximal time difference for neighbors
    max_distance = 0.2  # meters
    min_time_diff = pd.Timedelta(minutes=5)  # min time difference between points

    # remove all "artifical edge points"
    transformed_gdf = transformed_gdf[transformed_gdf['file_id'] != "artificial_boundary_points"].copy()

    # Check if all data was removed
    if transformed_gdf.empty:
        raise ValueError("No data left after removing artifical edge points!")

    # Convert Date/Time to pdandas datetime - sonar internal Date and time being used, not backed up by GPS time!
    transformed_gdf['DateTime'] = pd.to_datetime(transformed_gdf['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Check for daulty timestamps
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
            # Calculate in correct order: earlier point - later point
            if match_time > point_time:
                depth_diff = point_depth - match_depth  # switch for later - eralier point
                earlier_date, later_date = point_date, match_date
            else:
                depth_diff = match_depth - point_depth  # switch for later - eralier point
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

def calculate_depth_differences_close_points(transformed_gdf):
    """
    Berechnet die Tiefenunterschiede für Punkte, die sich in einem Abstand von weniger als 0,2 m befinden.
    Die Differenzen werden nach Tagen gruppiert, und es wird sichergestellt, dass immer der spätere
    Wert minus den früheren Wert berechnet wird. Punkte mit file_id == "artificial_boundary_points" werden ignoriert.
    Gibt zusätzlich ein GeoDataFrame mit den verwendeten Punkten zurück.
    """
    
    # Check for GeoDataFrame
    if not isinstance(transformed_gdf, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf muss ein GeoDataFrame sein!")

    # max distance betweeen points
    max_distance = 0.2  # meters

    # Entferne alle "artificial_boundary_points"
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

        # Convert list of neigbor points to numpy-array
        neighbor_depths = depths[neighbors]
        neighbor_dates = dates[neighbors]
        neighbor_times = datetimes[neighbors]

        # Calculate difference and save it grouped by survey days
        for neighbor_idx, match_depth, match_time, match_date in zip(neighbors, neighbor_depths, neighbor_times, neighbor_dates):
            # Only calculate earlier point - later point, switch for opposit
            if match_time > point_time:
                depth_diff = point_depth - match_depth  # switch for later - erlier point!
                earlier_date, later_date = point_date, match_date
            else:
                depth_diff = match_depth - point_depth  # switch for later - erlier point!
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





# mean and std of the depth differences


###############################################
################################ change date format

def compute_statistics_intersections(depth_diff_df:pd.DataFrame):

    """
    Berechnet den Durchschnitt und die Standardabweichung jeder Spalte im DataFrame
    und gibt das Ergebnis als neuen DataFrame zurück.
    """
    stats_df = pd.DataFrame({
        'Mean': depth_diff_df.mean(),
        'StdDev': depth_diff_df.std()
    })

    print(stats_df)


    # create a boxplot with the statistics - maybe change to saveing to files later 
    # determines scale of figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # create boxplot
    box = ax.boxplot([depth_diff_df[col].dropna() for col in depth_diff_df.columns], 
                      labels=depth_diff_df.columns, patch_artist=True)

    # change date formatting


    # scales uniform height of count for used points per boxplot
    y_pos = max(depth_diff_df.max(skipna=True)) * 1.1 if not depth_diff_df.isna().all().all() else 1

    # shows count for used points per boxplot
    for i, col in enumerate(depth_diff_df.columns, start=1):
        n_points = depth_diff_df[col].count()
        ax.text(i, y_pos, f"n={n_points}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # axis label and title
    ax.set_xlabel("Daten überschneidener Messreihen")
    ax.set_ylabel("Tiefenunterschied (m)")
    ax.set_title("Tiefenunterschiede überschneidener Messreihen")

    # x-and y- axis modifications
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(None, y_pos * 1.2)  # Extra Platz nach oben

    plt.tight_layout()
    plt.show()

    # add tilte, maybe better date format, axis text, possibility to safe
    return stats_df, box


def compute_statistics_closepoints(depth_diff_df:pd.DataFrame):
    """
    Berechnet den Durchschnitt und die Standardabweichung jeder Spalte im DataFrame
    und gibt das Ergebnis als neuen DataFrame zurück.
    """
    stats_df = pd.DataFrame({
        'Mean': depth_diff_df.mean(),
        'StdDev': depth_diff_df.std()
    })

    print(stats_df)
    # create a boxplot with the statistics - maybe change to saveing to files later 
    fig, ax = plt.subplots(figsize=(12, 6))

    # create boxplot
    box = ax.boxplot([depth_diff_df[col].dropna() for col in depth_diff_df.columns], 
                      labels=depth_diff_df.columns, patch_artist=True)

    # scales uniform height of count for used points per boxplot
    y_pos = max(depth_diff_df.max(skipna=True)) * 1.1 if not depth_diff_df.isna().all().all() else 1

    # shows count for used points per boxplot
    for i, col in enumerate(depth_diff_df.columns, start=1):
        n_points = depth_diff_df[col].count()
        ax.text(i, y_pos, f"n={n_points}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # axis label and title
    ax.set_xlabel("Daten verglichener Punkte")
    ax.set_ylabel("Tiefenunterschied (m)")
    ax.set_title("Tiefenunterschiede naheliegender Punkte")

    # x-and y- axis modifications
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(None, y_pos * 1.2)  # Extra Platz nach oben

    plt.tight_layout()
    plt.show()
    # add tilte, maybe better date format, axis text, possibility to safe
    return stats_df, box





def main():
    data_dir = Path("output/multibeam")
    print("reading data")
    filtered_data_df = pd.read_csv(data_dir/"filtered_data.csv")
    filtered_data_df["geometry"] = filtered_data_df["geometry"].apply(wkt.loads)
    filtered_data= gpd.GeoDataFrame(filtered_data_df, crs='EPSG:25833', geometry="geometry")


    print("Calculating depth difference on intersections")
    depth_difference_intersections, used_points_intersections_gdf= calculate_depth_differences_intersections(filtered_data)

    print("Calculating depth difference of close points")
    depth_difference_closep, used_points_closep_gdf = calculate_depth_differences_close_points(filtered_data)

    print("calculating statics")
    print("statics depth difference intersections")
    stats_intersec_df, boxintersec_plot = compute_statistics_intersections(depth_difference_intersections)
    print("statistics depth diffference close points")
    stats_closep_df, box_closep_plot = compute_statistics_closepoints(depth_difference_closep)

    print("saving data")

    data_output_dir = Path("output/multibeam/QC")
    used_points_intersections_gdf.to_csv(data_output_dir/"QC_closepointvalidation_used_points.csv", index=False)

    input("done!")


main()