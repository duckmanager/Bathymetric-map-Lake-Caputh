from pathlib import Path
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
from pymatreader import read_mat
from datetime import datetime, timedelta
from tqdm import tqdm
import json 

# add argparse
def create_dataframe(data_dir: Path):
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

    # TODO: scan all snr files for max number of CellXX needed
    # TODO: create pandas dataframe with propper headers like so:
    """
    {
        "file_id": [ ids_as_int ],
        "Utc": [ empty_for_now ],
        "Lat": [ empty_for_now ],
        "Long": [ empty_for_now ],
        "Sample": [ index_as_int ],
        "Date/Time": [ as_str ],
        "CellXX": [ data_as_float ],
        ... account for number of cells gathered previously
    }
    """
    # TODO: walk all snr and corresponding mat files.
    #   match and put them into the dataframe.


        # recognize file with the longest header
    longest_file = max(data_dir.glob("*.snr"), key=lambda f: len(f.read_text().splitlines()[0]), default=None)
    
    # create geodataframe with longest header variables + extra columns("file_id", "Utc", "Lat", "Long")
    header = None
    with longest_file.open("r") as file:
        snr_header = file.readline().strip().split(",")

    # assign additional columns not present in snr files
    additional_columns = ["file_id", "Utc", "Lat", "Long"]
   
    # create dataframe - maybe assign geometry columns and crs later when converting to geodataframe - or smarter to do now?

    snr_dataframe = pd.DataFrame(columns=additional_columns + snr_header)

    # specify data type - add more if necessary
    snr_dataframe =snr_dataframe.astype({"Depth (m)": float})

    return(snr_dataframe, snr_header)


# function to correct utc timeline of sonar-GPS
####### maybe change later to keeping the .sec


def correct_utc(utc_list):
    # find first time point
    first_valid_utc = next((utc for utc in utc_list if isinstance(utc, (int, float))), None)

    # return if no valid time found
    if first_valid_utc is None:
        print("Invalid UTC list!")
        return utc_list

    # Überprüfe, ob der erste gültige Zeitstempel nicht der erste Eintrag ist
    #if utc_list.index(first_valid_utc) != 0:
    #    print(f"Important Warning: used utc-list starts with invalid point - allocation of utc time is faulty - need to fix ") # add possibility to detect where it started and interpoalte forwards and backwards in time

    # start_time based on first time stamp
    start_time = datetime.strptime(str(int(first_valid_utc)).zfill(6), "%H%M%S")

    # create corrected time line, beginning with first time stamp and same length as orginal time line
    corrected_utc_pre = [start_time + timedelta(seconds=i) for i in range(len(utc_list))]

    # transform back to HHMMSS format
    corrected_utc = [t.strftime("%H%M%S") for t in corrected_utc_pre]

    return corrected_utc



# Assign snr and UTC data to dataframe
def assign_data_to_dataframe(data_dir: Path, snr_dataframe: pd.DataFrame, snr_header: list):
    data_list = []

    # Iterate through SNR-files
    for file in data_dir.glob("*.snr"):
        file_id = file.stem  # Dateiname als file_id

        # read same .mat file as snr file
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
            row_dict = dict(zip(snr_header, values))  # assign values to columns
            row_dict.update({
                "file_id": file_id,
                "Utc": corrected_utc[i] if i < len(corrected_utc) else None,
                "Lat": None,
                "Long": None # getting assigned later
            })  
            data_list.append(row_dict)

    # put data in dataframe - is it a new dataframe?
    snr_dataframe = pd.DataFrame(data_list, columns=snr_dataframe.columns)

    return snr_dataframe



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
                    print(f"Fehler beim Parsen von GPZDA in Datei {gps_file.name}: {e}")

    # turn into dataframe
    gps_df = pd.DataFrame(gps_data_list)
    return gps_df




# merge external GPS data and snr_dataframe based on date and GPS-retrieved-UTC

def merge_snr_gps(snr_dataframe: pd.DataFrame, gps_dataframe: pd.DataFrame):
    # Extract date from the 'Date/Time' column
    snr_dataframe['date'] = pd.to_datetime(snr_dataframe['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Extract day, month, year and safe as touple
    snr_dataframe['date'] = snr_dataframe['date'].apply(lambda x: (x.day, x.month, x.year) if pd.notnull(x) else None)

    # convert both utc to int 64 - maybe better fix in earlyer parts
    snr_dataframe['Utc'] = pd.to_numeric(snr_dataframe['Utc'], errors='coerce').astype('Int64')
    gps_dataframe['utc'] = pd.to_numeric(gps_dataframe['utc'], errors='coerce').astype('Int64')

    # merge both dataframes based on utc and date , by 'left'- merge the gps_dataframe is the important to stay
    merged_df = snr_dataframe.merge(gps_dataframe, how='left', left_on=['date', 'Utc'], right_on=['date', 'utc'])

    # put lat long in already created columns - propably cleaner way
    merged_df['Lat'] = merged_df['lat']
    merged_df['Long'] = merged_df['long']

    # delete temporary columns
    merged_dataframe = merged_df.drop(columns=['lat', 'long', 'utc', 'date'])

    return merged_dataframe


def convert_to_utm_geodf(merged_dataframe: pd.DataFrame):
     # turn pd.dataframe into geopanda.dataframe
     geodf = gpd.GeoDataFrame(merged_dataframe.drop(['Lat', 'Long'], axis=1),
                       crs={'init': 'epsg:4326'},
                       geometry=merged_dataframe.apply(lambda row: shapely.geometry.Point((row.Lat, row.Long)), axis=1)) 
     
     # project geodataframe to local UTM 33N (epsg:25833)
     geodf_projected = geodf.to_crs(epsg=25833)
     return(geodf_projected)


# no long lat or x y column, just geometry - maybe extract to keep columns?


# filter faulty points
def detect_and_remove_faulty_depths(geodf_projected: gpd.GeoDataFrame, window_size: int = 10, threshold: float = 1.0):
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

    geodf_projected['Depth (m)'] = pd.to_numeric(geodf_projected['Depth (m)'])



    # Iteriere durch die Surveys
    for file_id, group in geodf_projected.groupby("file_id"):
        depths = group["Depth (m)"].tolist()
        indices = group.index.tolist()

        # Iteriere durch Tiefenwerte und prüfe auf fehlerhafte Einträge
        for i in range(len(depths)):
            current_depth = depths[i]
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(depths), i + window_size // 2 + 1)

            # Berechne Durchschnitt der benachbarten Tiefenwerte
            window_depths = (depths[start_idx:end_idx])
            avg_depth = sum(window_depths) / int(len(window_depths))

            # Prüfen, ob der aktuelle Wert außerhalb des erlaubten Bereichs liegt
            if abs(current_depth - avg_depth) > threshold:
                entry = geodf_projected.loc[indices[i]].to_dict()
                faulty_entries.append(entry)
                faulty_indices.append(indices[i])

    # Lösche fehlerhafte Einträge gesammelt nach der Iteration
    geodf_projected.drop(faulty_indices, inplace=True)

    # Erstelle GeoDataFrame aus den fehlerhaften Einträgen
    faulty_gdf = gpd.GeoDataFrame(faulty_entries, geometry='geometry', crs="EPSG:25833")

    return geodf_projected, faulty_gdf

# reduce columns to necessary for map creation
def reduce_data(geodf_projected, faulty_gdf):
    filtered_gdf_snr = geodf_projected[["geometry","Depth (m)"]]
    filtered_faulty_gdf_snr = faulty_gdf[["geometry","Depth (m)"]]


    return(filtered_gdf_snr, filtered_faulty_gdf_snr)





#    -- -  - - -  - - -old
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
#old
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
#old
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


#old
def detect_and_remove_faulty_depths2(snr_dataframe: pd.DataFrame, window_size: int = 10, threshold: float = 1.0):
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


#old
def reduce_data2(full_data):
    filtered_snr = full_data[["X","Y","Depth (m)"]]

    return(filtered_snr)






def main():
    data_dir = Path("data")
    print("starting to create empty dataframe")
    snr_dataframe_empty, snr_header = create_dataframe(data_dir)
    print("assigning data to dataframe and correcting sonar-GPS times")
    snr_dataframe = assign_data_to_dataframe(data_dir, snr_dataframe_empty, snr_header)
    #print("correcting utc timestamps in snr")
    #snr_data = correct_utc_in_snr(snr_data)
    print("starting get_gps")
    gps_dataframe = get_gps_dataframe(data_dir)
    print("merging GPS and snr data")
    merged_dataframe = merge_snr_gps(snr_dataframe, gps_dataframe)
    print("converting to geodataframe and projecting to UTM 33N")
    geodataframe = convert_to_utm_geodf(merged_dataframe)
    #print("converting crs")
    #gps_data= convert_coords_to_utm(gps_data)
    #print("starting merge_snr_mat_gps")
    #full_data = merge_snr_mat_gps(snr_data, gps_data)
    #print("merging data")
    #snr_full_dataframe = merge_dicts(full_data)
    print("detecting and removing faulty depths")
    filtered_data, faulty_data = detect_and_remove_faulty_depths(geodataframe)
    print("reducing data")
    selected_snr_data, selected_faulty_snr_data = reduce_data(filtered_data, faulty_data)
    print("saving output")
    output_path = Path("output")
    selected_snr_data.to_csv(output_path / "snr_selected_filtered_new.csv", index=False)
    filtered_data.to_csv(output_path / "snr_collection_filtered_new.csv", index=False)
    faulty_data.to_csv(output_path / "snr_errors_collection.csv", index=False)
    selected_faulty_snr_data.to_csv(output_path / "snr_errors_selected.csv", index=False)

    input("we're all done!")
    

main()
