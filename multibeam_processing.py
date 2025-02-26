from pathlib import Path
import pandas as pd
import geopandas as gpd
import shapely.geometry
import numpy as np
from pymatreader import read_mat
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm import tqdm


# create dataframe
def create_dataframe(data_dir: Path):

        # recognize file with the longest header
    longest_file = max(data_dir.glob("*.sum"), key=lambda f: len(f.read_text().splitlines()[0]), default=None)
    
    # create geodataframe with longest header variables + extra columns("file_id", "Utc", "Lat", "Long")
    header = None
    with longest_file.open("r") as file:
        sum_header = file.readline().strip().split(",")

    # assign additional columns not present in sum files
    additional_columns = ["file_id", "Utc", "Lat", "Long"]
   
    # create dataframe - maybe assign geometry columns and crs later when converting to geodataframe - or smarter to do now?

    sum_dataframe = pd.DataFrame(columns=additional_columns + sum_header)

    # specify data type - add more if necessary
    sum_dataframe =sum_dataframe.astype({"Depth (m)": float})

    return(sum_dataframe, sum_header)


# function to correct utc timeline of sonar-GPS
####### maybe change later to keeping the .sec


def correct_utc(utc_list):
    # find first utc ---------------- no need to check if isinstance?
    first_valid_utc = next((utc for utc in utc_list if isinstance(utc, (int, float))), None)

    if first_valid_utc is None:
        print("Invalid UTC list!")
        return utc_list

    # seperate first utc sample into hour, minute, second, decisecond
    utc_str = f"{first_valid_utc:08.1f}"  # Format: HHMMSS.x
    base_time_str = utc_str[:6]  # HHMMSS
    decimal_part = utc_str[7]    # decimal place

    # convert base time into datetime object
    try:
        start_time = datetime.strptime(base_time_str, "%H%M%S")
    except ValueError as e:
        print(f"Error parsing time: {e}")
        return utc_list

    # create corrected timeline, beginning at base_time
    corrected_utc_pre = [start_time + timedelta(seconds=i) for i in range(len(utc_list))]

    # add decimal part back onto the corrected timestamps
    corrected_utc = [t.strftime("%H%M%S") + f".{decimal_part}" for t in corrected_utc_pre]

    return corrected_utc



# Assign sum and UTC data to dataframe
def assign_data_to_dataframe(data_dir: Path, sum_dataframe: pd.DataFrame, sum_header: list):
    data_list = []

    # Iterate through sum-files
    for file in data_dir.glob("*.sum"):
        file_id = file.stem  # Dateiname als file_id

        # read same .mat file as sum file
        mat_file = data_dir / f"{file_id}.mat"
        if mat_file.exists():
            mat_data = read_mat(str(mat_file))
            raw_utc = mat_data.get("GPS", {}).get("Utc", []) #extract .mat data
            corrected_utc = correct_utc(raw_utc if isinstance(raw_utc, list) else raw_utc.flatten()) # get corrected timeline
        else:
            print("Warrning: Mat-file -{file_id}.mat- ")  # Error if .mat-file not found

        with file.open("r") as f:
            lines = f.readlines()

        # extract data
        for i, line in enumerate(lines[1:]):  # skip header
            values = line.strip().split(",")
            row_dict = dict(zip(sum_header, values))  # assign values to columns
            row_dict.update({
                "file_id": file_id,
                "Utc": corrected_utc[i] if i < len(corrected_utc) else None,
                "Lat": None,
                "Long": None # getting assigned later
            })  
            data_list.append(row_dict)

    # put data in dataframe - is it a new dataframe?
    sum_dataframe = pd.DataFrame(data_list, columns=sum_dataframe.columns)

    return sum_dataframe


# load external GPS data and transform to UTM33N
def get_gps_dataframe(data_dir: Path):
    gps_data_list = []

    for gps_file in tqdm(data_dir.glob("*.txt")):
        gps_data = gps_file.read_text().splitlines()
        bestposa = (None, None)  # Variable to safe lat long tuple

        for line in gps_data:
            # update when new bestposa is reached
            if line.startswith("#BESTPOSA"):
                bestposa_raw = line.split(",")
                if bestposa_raw[10].replace(".", "").isdigit():
                    bestposa = (bestposa_raw[10], bestposa_raw[11])
                else:
                    bestposa = (bestposa_raw[11], bestposa_raw[12])

            # couple Bestposa with next GPZDA line
            elif line.startswith("$GPZDA"):
                try:
                    _, time, day, month, year, *_ = line.split(",")
                    date = (int(day), int(month), int(year))

                    # save current position with time stamp
                    gps_data_list.append({
                        "date": date,
                        "utc": int(float(time)),
                        "lat": bestposa[0],  
                        "long": bestposa[1]
                    })
                except ValueError as e:
                    print(f"error at {gps_file.name}: {e}")

    # turn into dataframe
    gps_df = pd.DataFrame(gps_data_list)


    # turn into geodataframe to project to UTM33N
    gps_geodf = gpd.GeoDataFrame(gps_df,
        crs='EPSG:4326',
        geometry=gpd.points_from_xy(gps_df.long, gps_df.lat)) # different opinions which variant is more performant
     
     # project geodataframe to local UTM 33N (epsg:25833)
    gps_geodf_projected = gps_geodf.to_crs(epsg=25833)
    return gps_geodf_projected


# erneute Aufteilung in dicts anch datum -  ggf. doch besser vor der beim Einlesen innseperaten files zu machen, damit auch umehrere an einem Tag kein Problem ist
# interpolate coords for missing .sec values
# Debugging-Ausgaben in der Interpolationsfunktion platzieren

def create_interpolated_coords(sum_df, gps_gdf):
    interpolated_coords = []

    # shorten Date/Time to only date in same format as in gps-files
    sum_df['date'] = pd.to_datetime(sum_df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    sum_df['date'] = sum_df['date'].apply(lambda x: (x.day, x.month, x.year) if pd.notnull(x) else None)

    # fails when using - but in general not necessary, only for different gps format
    # gps_gdf['date'] = gps_gdf['date'].apply(lambda x: (x.day, x.month, x.year))
########probably not necessary
    # make sure, gps date is in correct format
    gps_gdf['date'] = gps_gdf['date'].apply(lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # make sure sum date is in correct format
    sum_df['Utc'] = sum_df['Utc'].astype(str)
    gps_gdf['utc'] = gps_gdf['utc'].astype(str).str.zfill(6) 
#########
    # Iterate through gps file to safe same dates in one nested dict, to iterate more easy later
    gps_dict = {date: df.reset_index(drop=True) for date, df in gps_gdf.groupby('date')}

    # Iteration über die sum-Daten
    for idx, row in tqdm(sum_df.iterrows(), total=sum_df.shape[0]):
        date = row['date']
        utc_full = row['Utc']  # Format HHMMSS.s

        # Zeit in ganzzahligen und Dezimalteil aufteilen
        try:
            utc_str, decimal_str = utc_full.split('.')
            decimal_part = int(decimal_str)
        except ValueError:
            interpolated_coords.append((None, None))
            continue

        # Zugriff auf die GPS-Daten für dasselbe Datum
        gps_day = gps_dict.get(date)
        if gps_day is None:
            interpolated_coords.append((None, None))
            continue

        # Index des exakten UTC-Zeitpunkts im GPS-DataFrame finden
        gps_index = gps_day[gps_day['utc'] == utc_str].index

        if not gps_index.empty:
            idx = gps_index[0]  # Index des exakten Zeitpunkts

            # Prüfen, ob ein nachfolgender Punkt existiert
            if idx + 1 < len(gps_day):
                before_point = gps_day.iloc[idx]
                after_point = gps_day.iloc[idx + 1]

                # Interpolation der Koordinaten
                interp_factor = decimal_part / 10.0
                x_interp = before_point.geometry.x + interp_factor * (after_point.geometry.x - before_point.geometry.x)
                y_interp = before_point.geometry.y + interp_factor * (after_point.geometry.y - before_point.geometry.y)

                interpolated_coords.append((x_interp, y_interp))
            else:
                # Kein nachfolgender Punkt vorhanden
                interpolated_coords.append((None, None))
        else:
            # Kein exakter GPS-Zeitstempel gefunden
            interpolated_coords.append((None, None))

    # Add interpoalted coords to dataframe
    sum_df['Interpolated_Long'] = [coord[0] for coord in interpolated_coords]
    sum_df['Interpolated_Lat'] = [coord[1] for coord in interpolated_coords]

    return sum_df


def create_multibeam_points(sum_df: gpd.GeoDataFrame):

    # Konvertiere Boat Direction (deg) in numerische Werte (Fehlerhafte Werte werden NaN)
    sum_df['Boat Direction (deg)'] = pd.to_numeric(sum_df['Boat Direction (deg)'], errors='coerce')

    # Konvertiere Azimuth (Boat Direction) in Radian
    sum_df['Boat Direction (rad)'] = np.radians(sum_df['Boat Direction (deg)'])
    
    # Definiere die Beam-Typen und ihre entsprechenden Azimuth-Korrekturen (vorab in Radian umgewandelt)
    beams = [
        {"type": "VB", "depth_col": "VB Depth (m)", "angle": np.radians(0)},
        {"type": "Beam1", "depth_col": "BT Beam4 Depth (m)", "angle": np.radians(45)}, 
        {"type": "Beam2", "depth_col": "BT Beam3 Depth (m)", "angle": np.radians(135)}, 
        {"type": "Beam3", "depth_col": "BT Beam2 Depth (m)", "angle": np.radians(225)}, 
        {"type": "Beam4", "depth_col": "BT Beam1 Depth (m)", "angle": np.radians(315)} 
    ]
    
    # Leere Liste für das neue DataFrame
    transformed_data = []
    
    # Iteriere über das DataFrame
    for _, row in tqdm(sum_df.iterrows(), total=sum_df.shape[0], desc="Transforming sonar data"):
        base_x, base_y = row['Interpolated_Long'], row['Interpolated_Lat']
        boat_dir = row['Boat Direction (rad)']
        file_id, utc, date_time = row['file_id'], row['Utc'], row['Date/Time']
        
        for beam in beams:
            beam_type = beam["type"]
            depth = row[beam['depth_col']]
            
            if pd.notna(depth):  # Nur wenn eine gültige Tiefe vorhanden ist
                if beam_type == "VB":
                    new_x, new_y = base_x, base_y  # Vertical Beam bleibt an Originalposition
                else:
                    distance = np.tan(np.radians(25)) * float(depth)  # Berechne den seitlichen Abstand
                    azimuth_corrected = boat_dir + beam['angle']
                    new_x = base_x + distance * np.sin(azimuth_corrected)
                    new_y = base_y + distance * np.cos(azimuth_corrected)
                
                transformed_data.append({
                    "file_id": file_id,
                    "Utc": utc,
                    "Date/Time": date_time,
                    "Beam_type": beam_type,
                    "Depth (m)": depth,
                    "Longitude": new_x,
                    "Latitude": new_y,
                    "geometry": shapely.geometry.Point(new_x, new_y)
                })
    
    # Neues DataFrame aus den gesammelten Daten erstellen
    transformed_gdf = gpd.GeoDataFrame(transformed_data,geometry= "geometry", crs="EPSG:25833")
    return transformed_gdf

def adjust_depths(com_gdf: gpd.GeoDataFrame):

    # turn depths into negative values
    com_gdf['Depth (m)'] = pd.to_numeric(com_gdf["Depth (m)"], errors= 'coerce').abs() * (-1)

    return com_gdf




# filter faulty points #     old version
def detect_and_remove_faulty_depth(geodf_projected: gpd.GeoDataFrame, window_size: int = 10, threshold: float = 0.5): # change according to needs and survey circumstances
    """
    Identify faulty sum points based on moving average of sorrounding points and difference to them. Change window size and threshold for different filter

    Args:
        sum_dataframe (DataFrame): Der GeoDataFrame mit sum-Daten.
        window_size (int): Größe des gleitenden Fensters zur Durchschnittsberechnung.
        threshold (float): Maximale Abweichung in Metern, bevor ein Wert als fehlerhaft gilt.

    Returns:
        sum_dataframe (DataFrame): Aktualisiertes DataFrame ohne fehlerhafte Werte.
        faulty_df (GeoDataFrame): GeoDataFrame mit den fehlerhaften Werten.
    """
    faulty_entries = []  # for saving faulty points
    faulty_indices = []  # for saving index of faulty points

    geodf_projected['Depth (m)'] = pd.to_numeric(geodf_projected['Depth (m)'])



    # Iterate through the surveys
    for file_id, group in geodf_projected.groupby("file_id"):
        depths = group["Depth (m)"].tolist()
        indices = group.index.tolist()

        # Iterate through deep-values and check validity
        for i in range(len(depths)):
            current_depth = depths[i]
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(depths), i + window_size // 2 +1)

            # calculate average of neighbour points
            window_depths = (depths[start_idx:end_idx])
            avg_depth = (sum(window_depths) / int(len(window_depths)))

            # check if selected point is out of threshold compared to neighbour points
            if abs(current_depth - avg_depth) > threshold:
                entry = geodf_projected.loc[indices[i]].to_dict()
                faulty_entries.append(entry)
                faulty_indices.append(indices[i])

    # delete faulty points
    geodf_projected.drop(faulty_indices, inplace=True)

    # ceate dataframe from faulty points
    faulty_df =pd.DataFrame(faulty_entries)

    # create geodataframe from dataframe to be able to output shp-file later - not fully working
    faulty_gdf = gpd.GeoDataFrame(faulty_df, geometry=gpd.points_from_xy(faulty_df['Longitude'], faulty_df['Latitude']), crs="EPSG:25833")

    return geodf_projected, faulty_gdf



# Adding different error detection approach
# looking for neighbors in 5m? radius, taking average of them, without middle points. If middle point is more than ?0.5m? off this mean, it gets deleted and safed to faulty points.
    # Sicherstellen, dass es ein GeoDataFrame ist
def detect_and_remove_faulty_depths(geodf_projected: gpd.GeoDataFrame, max_distance: int = 5, threshold: float = 0.6):
    if not isinstance(geodf_projected, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf must be gdf!")
    
    # Extract relevant data as numpy array
    coords = np.vstack([geodf_projected.geometry.x, geodf_projected.geometry.y]).T
    depths = geodf_projected['Depth (m)'].values

    
    # create cKDTree for efficient neighbor search
    tree = cKDTree(coords)
    
    # look on tree for neighbours in set distance - safe indices in list
    indices = tree.query_ball_tree(tree, max_distance)

    # Lists for filtered indices
    valid_indices = []
    removed_indices = []

    for i, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Filtering points"):
        if len(neighbors) > 1:  # It must be neighbors
            neighbor_depths = [depths[j] for j in neighbors if j != i]  # dont use middlepoint
            if neighbor_depths:  # be sure they are not empty
                mean_depth = np.mean(neighbor_depths)  # calculate mean
                if abs(depths[i] - mean_depth) > threshold:  # check against threshold
                    removed_indices.append(i)
                else:
                    valid_indices.append(i)
        else:
            valid_indices.append(i)

    # check for borderpoints marked as faulty - should be only used as reference, not filtering them
    boundary_indices = [i for i in removed_indices if geodf_projected.iloc[i]['file_id'] == 'artificial_boundary_points']
    removed_indices = [i for i in removed_indices if i not in boundary_indices]
    valid_indices.extend(boundary_indices)

    # Create new gdf with filtered and faulty points
    filtered_gdf = geodf_projected.iloc[valid_indices].copy()
    removed_gdf = geodf_projected.iloc[removed_indices].copy()

    return filtered_gdf, removed_gdf














# adding correction for different lake levels

""" Select by Date,
    
    substitute or add specific number from depth values in each date

"""

def generate_boundary_points(data_dir):
    spacing = 1  # distamnce between artifical edge points in m (CRS EPSG:25833)
    
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
    """    for _, row in edge_gdf.iterrows():
        if pd.isna(row["next_depth"]) or row["distance_to_next"] >= 150:
            start_idx = row["nearest_boundary_idx"]
            for i in range(1, int(15 / spacing) + 1):
                if start_idx + i < len(boundary_gdf): #not necessary?
                    boundary_gdf.at[start_idx + i, "depth"] = row["Depth (m)"]"""


    # Berechne den Abstand zum vorherigen Messpunkt (cyclic)
    edge_gdf["prev_point"] = edge_gdf["geometry"].shift(1)
    edge_gdf["prev_depth"] = edge_gdf["Depth (m)"].shift(1)
    edge_gdf["distance_to_prev"] = edge_gdf.geometry.distance(edge_gdf["prev_point"])

    # Extrapolation entlang der Seeumrisslinie (zyklisch) in beide Richtungen
    extrapolation_distance = 15  # in Meter
    num_extrap_points = int(extrapolation_distance / spacing)

    for _, row in edge_gdf.iterrows():
        if pd.isna(row["nearest_boundary_idx"]):
            continue
        idx = int(row["nearest_boundary_idx"])
        depth_value = row["Depth (m)"]
        
        # Extrapolation in Vorwärtsrichtung:
        # Bedingung: kein nächster Messpunkt vorhanden oder Abstand >=150 m
        if pd.isna(row["next_depth"]) or row["distance_to_next"] >= 150:
            for i in range(1, num_extrap_points + 1):
                forward_idx = (idx + i) % len(boundary_gdf)
                # Fülle nur, wenn noch kein Wert gesetzt wurde
                if pd.isna(boundary_gdf.at[forward_idx, "depth"]):
                    boundary_gdf.at[forward_idx, "depth"] = depth_value
                else:
                    break  # Stoppe, wenn bereits ein Wert existiert
                    
        # Extrapolation in Rückwärtsrichtung:
        # Bedingung: kein vorheriger Messpunkt vorhanden oder Abstand >=150 m
        if pd.isna(row["prev_depth"]) or row["distance_to_prev"] >= 150:
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
    boundary_gdf['file_id'] = "artificial_boundary_points"
    boundary_gdf["Date"] = common_date # if fails, the measumrent points used different dates

    return boundary_gdf



def combine_multibeam_edge(geodf_projected, boundary_gdf):
    gdf_combined = gpd.GeoDataFrame(pd.concat([geodf_projected, boundary_gdf], ignore_index=True))

    return gdf_combined








# reduce columns to necessary for map creation
#def reduce_data(geodf_projected, faulty_gdf):
 #   filtered_gdf_sum = geodf_projected[["Interpolated_Lat", "Interpolated_Long","Depth (m)"]]
 #   filtered_faulty_gdf_sum = faulty_gdf[["Interpolated_Lat", "Interpolated_Long","Depth (m)"]]


  #  return(filtered_gdf_sum, filtered_faulty_gdf_sum)






def main():
    data_dir = Path("data")
    print("starting to create empty dataframe")
    sum_dataframe_empty, sum_header = create_dataframe(data_dir)
    print("assigning data to dataframe and correcting sonar-GPS times")
    sum_dataframe = assign_data_to_dataframe(data_dir, sum_dataframe_empty, sum_header)
    print("starting get_gps")
    gps_geodf_projected = get_gps_dataframe(data_dir) #including interpolation
    print("creating interpolated points")
    interpolated_sum = create_interpolated_coords(sum_dataframe, gps_geodf_projected)

    print("create multibeam points")
    multipoint_gdf = create_multibeam_points(interpolated_sum)
    # print("merging GPS and sum data")
    # merged_dataframe = merge_sum_gps(sum_dataframe, gps_geodf_projected)
    # print("converting to geodataframe and projecting to UTM 33N")
    # geodataframe_sum = convert_to_utm_geodf(merged_dataframe)
    print("creating lake outlines")
    boundary_gdf = generate_boundary_points(data_dir)
    
    print("merging boundary points and measurment points")
    gdf_combined = combine_multibeam_edge(multipoint_gdf, boundary_gdf)
    
    print("adjusting depths")
    adjusted_gdf = adjust_depths(gdf_combined)
    
    print("detecting and removing faulty depths")
    filtered_data, faulty_data = detect_and_remove_faulty_depths(adjusted_gdf) # Reihenfolge umgekehrt
    #-print("reducing data")
    #-selected_sum_data, selected_faulty_sum_data = reduce_data(filtered_data, faulty_data)
    print("saving output")
    output_path = Path("output/multibeam")
    filtered_data.to_csv(output_path / "m_filtered_newedge.csv", index=False)
    faulty_data.to_csv(output_path / "m_error_newedge.csv", index=False)
    #-filtered_data.to_csv(output_path / "sum_int_collection_filtered_cleandup.csv", index=False)
    #-  faulty_data.to_csv(output_path / "sum_multibeam_error.csv", index=False)
    #-selected_faulty_sum_data.to_csv(output_path / "sum_int_errors_selected.csv", index=False)
    #--gdf_complete.to_csv(output_path / "depth_and_average_sum_int_filtered_outline.csv", index=False)
    #--faulty_data.to_csv(output_path / "error_depth_and_average_sum_int_filtered.csv", index=False)
    # adjusted_gdf.to_csv(output_path / "multibeam_nozeros.csv", index=False)

    # output data as shp-file
    # filtered_data.to_file(output_path / "sum_int.shp", driver='ESRI Shapefile')
    # selected_faulty_sum_data.to_file(output_path / "sum_int_error.shp", driver='ESRI Shapefile')
    

    input("we're all done!")
    return
    

main()
