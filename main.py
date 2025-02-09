from pathlib import Path
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
from pymatreader import read_mat
from datetime import datetime, timedelta
from tqdm import tqdm


# create dataframe
def create_dataframe(data_dir: Path):

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

    # check if first used time stamp is actual first time stamp - can be done differently
    #if utc_list.index(first_valid_utc) != 0:
    #    print(f"Important Warning: used utc-list starts with invalid point - allocation of utc time is faulty - need to fix ") # add possibility to detect where it started and interpoalte forwards and backwards in time

    # start_time based on first time stamp
    start_time = datetime.strptime(str(int(first_valid_utc)).zfill(6), "%H%M%S%f")

    # create corrected time line, beginning with first time stamp and same length as orginal time line
    corrected_utc_pre = [start_time + timedelta(seconds=i) for i in range(len(utc_list))]

    # transform back to HHMMSS format - [:-5] to shorten the decimal place to 1/10 of second
    corrected_utc = [t.strftime("%H%M%S.%f")[:-5] for t in corrected_utc_pre]

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
def create_interpolated_coords(snr_df, gps_gdf):
    interpolated_coords = []

    # shorten Date/Time to only date in same format as in gps-files
    snr_df['date'] = pd.to_datetime(snr_df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    snr_df['date'] = snr_df['date'].apply(lambda x: (x.day, x.month, x.year) if pd.notnull(x) else None)

    # fails when using - but in general not necessary, only for different gps format
    # gps_gdf['date'] = gps_gdf['date'].apply(lambda x: (x.day, x.month, x.year))

    # Iterate through gps file to safe same dates in one nested dict, to iterate more easy later
    gps_dict = {date: df for date, df in gps_gdf.groupby('date')}

    # Iteriere through gps data
    for idx, row in tqdm(snr_df.iterrows(), total=snr_df.shape[0]):
        date = row['date']  
        utc_full = row['Utc']  # Format HHMMSS.1

        # Seperate time from .second - part
        utc_str, decimal_str = utc_full.split('.')
        decimal_part = int(decimal_str)

        # access gps data from same day
        gps_day = gps_dict.get(date)
        if gps_day is None:
            interpolated_coords.append((None, None))
            continue

        # find neighbour gps points -> can be eliminated as they are always consecutive, but maybe more reliant like this?
        utc_int = int(utc_str)
        before_point = gps_day[gps_day['utc'] == utc_int]
        after_point = gps_day[gps_day['utc'] == utc_int + 1]

        # If datapoint is exaxtly .0 - keep coordinates
        if decimal_part == 0 and not before_point.empty:
            interpolated_coords.append((before_point.iloc[0].geometry.x, before_point.iloc[0].geometry.y))
            continue

        # safe NA if no neighbour points can be found
        if before_point.empty or after_point.empty:
            interpolated_coords.append((None, None))
            continue

        # Interpolation of coordinates based on decimal place - procedure like vector calculation
        interp_factor = decimal_part / 10.0
        x_interp = before_point.iloc[0].geometry.x + interp_factor * (after_point.iloc[0].geometry.x - before_point.iloc[0].geometry.x)
        y_interp = before_point.iloc[0].geometry.y + interp_factor * (after_point.iloc[0].geometry.y - before_point.iloc[0].geometry.y)

        interpolated_coords.append((x_interp, y_interp))

    # Add interpoalted coords to dataframe
    snr_df['Interpolated_Long'] = [coord[0] for coord in interpolated_coords]
    snr_df['Interpolated_Lat'] = [coord[1] for coord in interpolated_coords]

    return snr_df

    



# old version

"""    for idx, row in tqdm(snr_df.iterrows(), total=snr_df.shape[0]):
        date = row['date']  # Bereits ein Tupel (Tag, Monat, Jahr)
        utc_full = row['Utc']  # Format HHMMSS.1

        # Trenne Zeit in volle Sekunde und Nachkommastelle
        utc_str, decimal_str = utc_full.split('.')
        decimal_part = int(decimal_str)

        # Filter GPS-Daten nach Datum und erstelle eine Kopie
        gps_day = gps_gdf[gps_gdf['date'] == date].copy()
        if gps_day.empty:
            interpolated_coords.append((None, None))
            continue

        # Finde die benachbarten GPS-Punkte basierend auf UTC-Zeit
        before_point = gps_day[gps_day['utc'] == int(utc_str)].copy()
        after_point = gps_day[gps_day['utc'] == int(utc_str) + 1].copy()

        # Falls exakter GPS-Zeitpunkt vorhanden ist
        if decimal_part == 0 and not before_point.empty:
            interpolated_coords.append((before_point.iloc[0].geometry.x, before_point.iloc[0].geometry.y))
            continue

        # Falls keine benachbarten Punkte gefunden werden
        if before_point.empty or after_point.empty:
            interpolated_coords.append((None, None))
            continue

        # Interpolation der Koordinaten basierend auf Nachkommastelle
        interp_factor = decimal_part / 10.0
        x_interp = before_point.iloc[0].geometry.x + interp_factor * (after_point.iloc[0].geometry.x - before_point.iloc[0].geometry.x)
        y_interp = before_point.iloc[0].geometry.y + interp_factor * (after_point.iloc[0].geometry.y - before_point.iloc[0].geometry.y)

        interpolated_coords.append((x_interp, y_interp))

    # Interpolierte Koordinaten dem DataFrame hinzufügen
    snr_df['Interpolated_Long'] = [coord[0] for coord in interpolated_coords]
    snr_df['Interpolated_Lat'] = [coord[1] for coord in interpolated_coords]

    return snr_df"""







# not necessary anymore as already merged
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

# not necessary anymore
def convert_to_utm_geodf(merged_dataframe: pd.DataFrame):
     # turn pd.dataframe into geopanda.dataframe
     geodf = gpd.GeoDataFrame(merged_dataframe.drop(['Lat', 'Long'], axis=1),
                       crs='EPSG:4326',
                       geometry=merged_dataframe.apply(lambda row: shapely.geometry.Point((row.Long, row.Lat)), axis=1)) 
     
     # project geodataframe to local UTM 33N (epsg:25833)
     geodf_projected = geodf.to_crs(epsg=25833)
     return geodf_projected


# no long lat or x y column, just geometry - maybe extract to keep columns?


# filter faulty points
def detect_and_remove_faulty_depths(geodf_projected: gpd.GeoDataFrame, window_size: int = 10, threshold: float = 0.5): # change according to needs and survey circumstances
    """
    Identify faulty snr points based on moving average of sorrounding points and difference to them. Change window size and threshold for different filter

    Args:
        snr_dataframe (DataFrame): Der GeoDataFrame mit SNR-Daten.
        window_size (int): Größe des gleitenden Fensters zur Durchschnittsberechnung.
        threshold (float): Maximale Abweichung in Metern, bevor ein Wert als fehlerhaft gilt.

    Returns:
        snr_dataframe (DataFrame): Aktualisiertes DataFrame ohne fehlerhafte Werte.
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
            avg_depth = sum(window_depths) / int(len(window_depths))

            # check if selected point is out of threshold compared to neighbour points
            if abs(current_depth - avg_depth) > threshold:
                entry = geodf_projected.loc[indices[i]].to_dict()
                faulty_entries.append(entry)
                faulty_indices.append(indices[i])

    # delecte faulty points
    geodf_projected.drop(faulty_indices, inplace=True)

    # create geodataframe from faulty points
    faulty_gdf = gpd.GeoDataFrame(faulty_entries, geometry='geometry', crs="EPSG:25833")

    return geodf_projected, faulty_gdf

# reduce columns to necessary for map creation
def reduce_data(geodf_projected, faulty_gdf):
    filtered_gdf_snr = geodf_projected[["geometry","Depth (m)"]]
    filtered_faulty_gdf_snr = faulty_gdf[["geometry","Depth (m)"]]


    return(filtered_gdf_snr, filtered_faulty_gdf_snr)






def main():
    data_dir = Path("data")
    print("starting to create empty dataframe")
    snr_dataframe_empty, snr_header = create_dataframe(data_dir)
    print("assigning data to dataframe and correcting sonar-GPS times")
    snr_dataframe = assign_data_to_dataframe(data_dir, snr_dataframe_empty, snr_header)
    print("starting get_gps")
    gps_geodf_projected = get_gps_dataframe(data_dir) #including interpolation
    print("creating interpolated points")
    interpolated_snr = create_interpolated_coords(snr_dataframe, gps_geodf_projected)
    # print("merging GPS and snr data")
    # merged_dataframe = merge_snr_gps(snr_dataframe, gps_geodf_projected)
    # print("converting to geodataframe and projecting to UTM 33N")
    # geodataframe_snr = convert_to_utm_geodf(merged_dataframe)
    print("detecting and removing faulty depths")
    filtered_data, faulty_data = detect_and_remove_faulty_depths(interpolated_snr)
    print("reducing data")
    selected_snr_data, selected_faulty_snr_data = reduce_data(filtered_data, faulty_data)
    print("saving output")
    output_path = Path("output")
    selected_snr_data.to_csv(output_path / "snr_int_selected_filtered_new.csv", index=False)
    filtered_data.to_csv(output_path / "snr_int_collection_filtered_new.csv", index=False)
    faulty_data.to_csv(output_path / "snr_int_errors_collection.csv", index=False)
    selected_faulty_snr_data.to_csv(output_path / "snr_int_errors_selected.csv", index=False)
    # output data as shp-file
    selected_snr_data.to_file(output_path / "snr_int.shp", driver='ESRI Shapefile')
    selected_faulty_snr_data.to_file(output_path / "snr_int_error.shp", driver='ESRI Shapefile')

    input("we're all done!")
    

main()
