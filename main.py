from pathlib import Path
import pandas as pd
import geopandas as gpd
from pymatreader import read_mat
from datetime import datetime, timedelta
from shapely.geometry import Point
from tqdm import tqdm
import json 

# add argparse
def get_snr_mat(data_dir: Path):
    """
    find and read all snr and mat files. match them and build a dict.
    data = {
        "file_id": [
            snr_header_line_as_string,
            [
                utc_timestamp (may need correction later),
                lat (None for now),
                long (None for now),
                *snr_data_unpacked
            ],
            ...
        ],
        ...
    }


    args:
        data_dir Path: Path object pointing to dir with data files

    returns:
        data dict as laid out above
    """
    data = {}
    for snr_file in tqdm(data_dir.glob("*.snr")):
        snr_data_raw = snr_file.read_text().splitlines()
        snr_data = snr_data_raw[1:]
        snr_header = snr_data_raw[0]

        mat_data_Utc = read_mat(str(snr_file)[:-4] + ".mat")["GPS"]["Utc"]

        data_id = snr_file.name[:-4]
        data[data_id] = ["Utc,X,Y," + snr_header]
        for snr, utc in zip(snr_data, mat_data_Utc):
            # TODO: save the snr data we need
            data[data_id].append([utc, None, None, *snr.split(",")])
    return data



# Correct the timestamps to full second steps
# maybe need to adjust for sub seconds - then interpolate the GNS tracks
def correct_utc_in_snr(snr_data: dict):
    for data_id, entries in snr_data.items():
        # Extract first and last time stamp
        utc_times = [int(entry[0]) for entry in entries[1:] if isinstance(entry, list) and entry]
        
        if not utc_times:
            continue  # skip if no timestamps avaiable

        # transform UTC to datetime-object
        start_time = datetime.strptime(str(utc_times[0]), "%H%M%S")
        end_time = datetime.strptime(str(utc_times[-1]), "%H%M%S")
        
        # count of time steps to fill
        num_steps = len(utc_times)
        
        # create new list with setps in seconds
        corrected_utc = [start_time + timedelta(seconds=i) for i in range(num_steps)]
        
        # transform back to HHMMSS format
        corrected_utc_str = [t.strftime("%H%M%S") for t in corrected_utc]

        # correct snr_data with new time series
        for i, entry in enumerate(entries[1:]):
            if isinstance(entry, list) and entry:
                entry[0] = corrected_utc_str[i] 

    return snr_data

def get_gps(data_dir: Path):
    """
    find and read all gps files. parse them and build a dict.
    data = {
        "date": {
            "time": (X, Y)
        }
    }
    """
    data = {}
    bestposa = None
    for gps_file in tqdm(data_dir.glob("*.txt")):
        date = None
        gps_data = gps_file.read_text().splitlines()
        for line in tqdm(gps_data):
            if line.startswith("#BESTPOSA"):
                # 0        ,1   ,2,   ,3   ,4   ,5         ,6,7,8              ,9   ,10 lat    ,11 long
                # #BESTPOSA,COM2,0,0.0,FINE,2349,479940.000,,,;INSUFFICIENT_OBS,NONE,0.00000000,0.00000000,0.000,17.230,WGS84,0.0000,0.0000,0.0000,"",0.0,0.0,31,0,,,,,,*9cefb0ad
                bestposa_raw = line.split(",")
                if bestposa_raw[10].replace(".","").isdigit():
                    bestposa = (bestposa_raw[10], bestposa_raw[11])
                else:
                    bestposa = (bestposa_raw[11], bestposa_raw[12])

            elif line.startswith("$GPZDA"):
                # $GPZDA,131842.00,17,1,2025,,*59
                _, time, day, month, year, *_ = line.split(",")
                if date is None:
                    date = (int(day), int(month), int(year))
                    data[date] = {}

                data[date][int(float(time))] = bestposa

    return data

def convert_coords_to_utm(gps_data: dict) -> dict:
    """
    Converts latitude and longitude in gps_data to UTM32 coordinates using GeoPandas.
    Removes entries where lat or lon is '0.00000000' before transformation.

    args:
        gps_data (dict): GPS data in the original format.

    returns:
        dict: GPS data with UTM32 coordinates, or removed entries where lat or lon is '0.00000000'.
    """
    for date, times in tqdm(gps_data.items()):
        times_to_delete = []

        for time, (X_str, Y_str) in times.items():
            # Convert coordinates from strings to floats
            try:
                X = float(X_str) # using X and Y to avoid confusion after converting later on
                Y = float(Y_str)
            except ValueError:
                print(f"Invalid coordinates at {date}, {time}: {X_str}, {Y_str}")
                continue

            # Skip entries where X or Y is 0
            if X == 0.0 or Y == 0.0:
                times_to_delete.append(time)
                continue

            # Create a Point geometry from Y, X
            point = Point(Y, X)  # GeoPandas uses (longitude, latitude)
            
            # Convert to GeoDataFrame with WGS84 (EPSG:4326)
            gdf = gpd.GeoDataFrame([{'geometry': point}], crs="EPSG:4326")
            
            # Transform to UTM32 (EPSG:32632)
            gdf_utm = gdf.to_crs(epsg=25833)
            
            # Extract the UTM coordinates
            utm_x, utm_y = gdf_utm.geometry.x[0], gdf_utm.geometry.y[0]

            # Update the gps_data with the UTM coordinates
            gps_data[date][time] = (str(utm_x), str(utm_y))

        # Delete invalid entries with X/Y == 0
        for time in times_to_delete:
            del gps_data[date][time]

    return gps_data


#####
# Ergränzen der Interpolation    
#####

def merge_snr_mat_gps(snr_data: dict, gps_data: dict):
    for snr_id_key, snr_id_value in tqdm(snr_data.items()):
        # get key and value of the snr_data dict
        for index, snr_point in enumerate(snr_id_value[1:]):
            # iterate through the data points of the current snr_id
            # skip header line

            # find date and time in snr data point 
            # 0  , 1  , 2   , 3       , 4        , 5
            # Utc, lat, long, sample #, date/time,Frequency (MHz),Profile Type,Depth (m), ... rest varies
            # date/time looks like ,1/17/2025 4:02:38 PM,
            month, day, year = snr_point[4].split(" ")[0].split("/")
            date = (int(day), int(month), int(year))
            time = int(snr_point[0])

            # use date and time found in snr data point to get X and Y from gps data
            try:
                X, Y = gps_data[date][time]
            except KeyError as e:
                print(f"date or time not found with {date}, {time}; error: {e}")


            # use snr_id_key to find the list we're dealing with. then use the index to find the current data point. then assign X and Y
            snr_data[snr_id_key][index + 1][1] = X
            snr_data[snr_id_key][index + 1][2] = Y
    return snr_data

def merge_dicts(full_data):
    # Master-Columnlist to keep the order
    master_columns = ["Survey_ID"]
    combined_data = []

    # Assign headers to the list
    for key, nested_data in full_data.items():
        header_columns = nested_data[0].split(",")  # Extract headers
        for col in header_columns:
            if col not in master_columns:
                master_columns.append(col)  # keep order

    # fill with values
    for key, nested_data in full_data.items():
        header_columns = nested_data[0].split(",")
        
        for entry in nested_data[1:]:
            if isinstance(entry, list):
                # assign NA to missing values
                row_data = dict(zip(header_columns, entry))
                row_data["Survey_ID"] = key  # Füge die Survey-ID hinzu
                combined_data.append({col: row_data.get(col, pd.NA) for col in master_columns})

    # Survey_ID an zweiter Stelle einfügen
    if "Survey_ID" not in master_columns:
        master_columns.insert(1, "Survey_ID")
    # create geopanda dataframe
    snr_dataframe = pd.DataFrame(combined_data, columns=master_columns)
    return snr_dataframe



def detect_and_remove_faulty_depths(snr_dataframe: pd.DataFrame, window_size: int = 10, threshold: float = 1.0):
    """
    Identifiziert fehlerhafte Tiefenwerte in den SNR-Daten innerhalb derselben Surveys basierend auf dem Durchschnitt
    eines gleitenden Fensters und speichert diese Werte in einem GeoDataFrame.

    Args:
        snr_dataframe (DataFrame): Der GeoDataFrame mit SNR-Daten.
        window_size (int): Größe des gleitenden Fensters zur Durchschnittsberechnung.
        threshold (float): Maximale Abweichung in Metern, bevor ein Wert als fehlerhaft gilt.

    Returns:
        snr_dataframe (DataFrame): Aktualisiertes DataFrame ohne fehlerhafte Werte.
        faulty_df (GeoDataFrame): GeoDataFrame mit den fehlerhaften Werten.
    """
    faulty_entries = []  # Zum Speichern der fehlerhaften Werte
    faulty_indices = []  # Liste zum Speichern der zu löschenden Indizes

    # Iteriere durch die Surveys
    for survey_id, group in snr_dataframe.groupby("Survey_ID"):
        depths = group["Depth (m)"].tolist()
        indices = group.index.tolist()

        # Iteriere durch Tiefenwerte und prüfe auf fehlerhafte Einträge
        for i in range(len(depths)):
            current_depth = depths[i]
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(depths), i + window_size // 2 + 1)

            # Berechne Durchschnitt der benachbarten Tiefenwerte
            window_depths = depths[start_idx:end_idx]
            avg_depth = sum(window_depths) / len(window_depths)

            # Prüfen, ob der aktuelle Wert außerhalb des erlaubten Bereichs liegt
            if abs(current_depth - avg_depth) > threshold:
                entry = snr_dataframe.loc[indices[i]].to_dict()
                faulty_entries.append(entry)
                faulty_indices.append(indices[i])

    # Lösche fehlerhafte Einträge gesammelt nach der Iteration
    snr_dataframe.drop(faulty_indices, inplace=True)

    # Erstelle GeoDataFrame aus den fehlerhaften Einträgen
    faulty_df = gpd.GeoDataFrame(faulty_entries, geometry=gpd.points_from_xy(
        [entry["X"] for entry in faulty_entries], [entry["Y"] for entry in faulty_entries]
    ), crs="EPSG:25833")

    return snr_dataframe, faulty_df



def reduce_data(full_data):
    filtered_snr = full_data[["X","Y","Depth (m)"]]

    return(filtered_snr)






def main():
    data_dir = Path("data")
    print("starting get_snr_mat")
    snr_data = get_snr_mat(data_dir)
    print("correcting utc timestamps in snr")
    snr_data = correct_utc_in_snr(snr_data)
    print("starting get_gps")
    gps_data = get_gps(data_dir)
    print("converting crs")
    gps_data= convert_coords_to_utm(gps_data)
    print("starting merge_snr_mat_gps")
    full_data = merge_snr_mat_gps(snr_data, gps_data)
    print("merging data")
    snr_full_dataframe = merge_dicts(full_data)
    print("detecting and removing faulty depths")
    filtered_data, faulty_data = detect_and_remove_faulty_depths(snr_full_dataframe)
    print("reducing data")
    selected_data = reduce_data(filtered_data)
    print("saving output")
    selected_data.to_csv("snr_selected_filtered.csv", index=False)
    snr_full_dataframe.to_csv("snr_collection_filtered.csv", index=False)
    faulty_data.to_csv("snr_collection.csv", index=False)
    input("we're all done!")
    

main()
