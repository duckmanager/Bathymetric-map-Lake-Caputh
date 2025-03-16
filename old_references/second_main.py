from pathlib import Path
import pandas as pd
from pymatreader import read_mat
from datetime import datetime, timedelta
from tqdm import tqdm
import json 

# add argparse
def get_vel_mat(data_dir: Path):


    data = {}
    for vel_file in tqdm(data_dir.glob("*.vel")):
        vel_data_raw = vel_file.read_text().splitlines()
        vel_data = vel_data_raw[1:]
        vel_header = vel_data_raw[0]

        mat_data_Utc = read_mat(str(snr_file)[:-4] + ".mat")["GPS"]["Utc"]

        data_id = snr_file.name[:-4]
        data[data_id] = ["Utc,lat,long" + snr_header]
        for snr, utc in zip(snr_data, mat_data_Utc):
            # TODO: save the snr data we need
            data[data_id].append([utc, None, None, *snr.split(",")])
    return data

def get_gps(data_dir: Path):
    """
    find and read all gps files. parse them and build a dict.
    data = {
        "date": {
            "time": (lat, long)
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

# Assign snr data to dataframe
def assign_data_to_dataframe(data_dir: Path, snr_dataframe: pd.DataFrame, snr_header: list):
    data_list = []

    # iterate through all snr files
    for file in data_dir.glob("*.snr"):
        file_id = file.stem  # file name as file_id

        with file.open("r") as f:
            lines = f.readlines()
        
        # extract data
        for line in lines[1:]: #skip header
            values = line.strip().split(",")
            row_dict = dict(zip(snr_header, values))  # assign values to header
            row_dict.update({"file_id": file_id, "Utc": None, "Lat": None, "Long": None})  # Zusätzliche Spalten
            data_list.append(row_dict)

    # Daten in das DataFrame einfügen
    snr_dataframe = pd.DataFrame(data_list, columns=snr_dataframe.columns)

    return snr_dataframe


def create_interpolated_coords(snr_df, gps_gdf):
    interpolated_coords = []

    # shorten Date/Time to only date in same format as in gps-files
    snr_df['date'] = pd.to_datetime(snr_df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    snr_df['date'] = snr_df['date'].apply(lambda x: (x.day, x.month, x.year) if pd.notnull(x) else None)

    # fails when using - but in general not necessary, only for different gps format
    # gps_gdf['date'] = gps_gdf['date'].apply(lambda x: (x.day, x.month, x.year))
########
    # Stelle sicher, dass das Datum im GPS-DataFrame im gleichen Format vorliegt
    gps_gdf['date'] = gps_gdf['date'].apply(lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # Sicherstellen, dass auch die UTC-Werte gleich formatiert sind
    snr_df['Utc'] = snr_df['Utc'].astype(str)
    gps_gdf['utc'] = gps_gdf['utc'].astype(str).str.zfill(6)  # Sicherstellen, dass führende Nullen nicht fehlen
#########
    # Iterate through gps file to safe same dates in one nested dict, to iterate more easy later
    gps_dict = {date: df for date, df in gps_gdf.groupby('date')}

    # Iterate through gps data
    for idx, row in tqdm(snr_df.iterrows(), total=snr_df.shape[0]):
        date = row['date']  
        utc_full = row['Utc']  # Format HHMMSS.1

        # Seperate time from .second - part
        utc_str, decimal_str = utc_full.split('.')
        decimal_part = int(decimal_str)
        utc_int = int(utc_str)

        # access gps data from same day
        gps_day = gps_dict.get(date)
        if gps_day is None:
            interpolated_coords.append((None, None))
            continue

        # find neighbour gps points -> can be eliminated as they are always consecutive, but maybe more reliant like this?
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


####newest version before debugging with gpt

def create_interpolated_coords(snr_df, gps_gdf):
    interpolated_coords = []

    # shorten Date/Time to only date in same format as in gps-files
    snr_df['date'] = pd.to_datetime(snr_df['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    snr_df['date'] = snr_df['date'].apply(lambda x: (x.day, x.month, x.year) if pd.notnull(x) else None)

    # fails when using - but in general not necessary, only for different gps format
    # gps_gdf['date'] = gps_gdf['date'].apply(lambda x: (x.day, x.month, x.year))
########
    # Stelle sicher, dass das Datum im GPS-DataFrame im gleichen Format vorliegt
    gps_gdf['date'] = gps_gdf['date'].apply(lambda x: (int(x[0]), int(x[1]), int(x[2])))

    # Sicherstellen, dass auch die UTC-Werte gleich formatiert sind
    snr_df['Utc'] = snr_df['Utc'].astype(str)
    gps_gdf['utc'] = gps_gdf['utc'].astype(str).str.zfill(6)  # Sicherstellen, dass führende Nullen nicht fehlen
#########
    # Iterate through gps file to safe same dates in one nested dict, to iterate more easy later
    gps_dict = {date: df.reset_index(drop=True) for date, df in gps_gdf.groupby('date')}

    # Iteration über die SNR-Daten
    for idx, row in tqdm(snr_df.iterrows(), total=snr_df.shape[0]):
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
    snr_df['Interpolated_Long'] = [coord[0] for coord in interpolated_coords]
    snr_df['Interpolated_Lat'] = [coord[1] for coord in interpolated_coords]

    return snr_df

####
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
    start_time = datetime.strptime(str(int(first_valid_utc)).zfill(6), "%H%M%S %f")

    # create corrected time line, beginning with first time stamp and same length as orginal time line
    corrected_utc_pre = [start_time + timedelta(seconds=i) for i in range(len(utc_list))]

    # transform back to HHMMSS format - [:-5] to shorten the decimal place to 1/10 of second
    corrected_utc = [t.strftime("%H%M%S.%f")[:-5] for t in corrected_utc_pre]

    return corrected_utc





# correct time backup:
# Assign sum-depth and UTC data to dataframe
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

# function to correct utc timeline of sonar-GPS - applied in "assign_data_to_dataframe"
def correct_utc(utc_list,data_dir):
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






def calculate_depth_differences_intersections(transformed_gdf):
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
    max_distance = 0.2  # meters
    min_time_diff = pd.Timedelta(minutes=5)  # min timedifference between points
    
    # convert time column
    transformed_gdf['DateTime'] = pd.to_datetime(transformed_gdf['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    transformed_gdf['Date'] = transformed_gdf['DateTime'].dt.date

    # notna for the artifical boundary points
    unique_dates = sorted(transformed_gdf.loc[transformed_gdf['Date'].notna(), 'Date'].unique())

    # unique_dates = sorted(transformed_gdf['Date'].unique())
    
    # setup date comparisons - also accepting same day crossings
    date_combinations = set(tuple(sorted((d1, d2))) for d1, d2 in combinations(unique_dates, 2))
    date_combinations.update((d, d) for d in unique_dates)
    
    # Initialising dicts - defaultdict just adds column if it dowsnt exists yet so its obsolete to check first
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

    for idx, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Calculating depth difference"):
        point_time = datetimes[idx]
        point_date = dates[idx]
        point_depth = depths[idx]
        
        for neighbor_idx in neighbors:
            if neighbor_idx > idx:  # Comparing only wth points of higher idx - so points wont be double counted
                continue

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
            depth_diff_dict[date_pair].append((point_depth - match_depth))#abs - take the value
            used_indices.add(idx)
            used_indices.update(valid_neighbors)
        # Bestimme die richtige Reihenfolge (älteres Datum zuerst)


    
    # create dataframe with depth differences from dict with each column starting in the first line
    depth_diff_df = pd.DataFrame(dict([(f"{k[0]}-{k[1]}", pd.Series(v)) for k, v in depth_diff_dict.items()]))
    
    # Create a GeodataFrame with all used points for visual controle
    used_points_gdf = transformed_gdf.loc[list(used_indices)].copy()
    
    return depth_diff_df, used_points_gdf



def calculate_depth_differences_close_points(transformed_gdf):
    """
    Berechnet die Tiefenunterschiede für Punkte, die sich in einem Abstand von weniger als 0,1 m befinden.
    Die zeitliche Komponente wird ignoriert.
    Gibt zusätzlich ein GeoDataFrame mit den verwendeten Punkten zurück,
    gruppiert nach dem Aufnahmedatum.
    """
    # Check for GeoDataFrame
    if not isinstance(transformed_gdf, gpd.GeoDataFrame):
        raise ValueError("transformed_gdf muss ein GeoDataFrame sein!")

    # Maximaler Abstand zwischen Punkten
    max_distance = 0.2  # Meter

    # Konvertiere Zeitspalte in datetime und extrahiere Datum
    transformed_gdf['DateTime'] = pd.to_datetime(transformed_gdf['Date/Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    transformed_gdf['Date'] = transformed_gdf['DateTime'].dt.date
    unique_dates = sorted(transformed_gdf.loc[transformed_gdf['Date'].notna(), 'Date'].unique())
    
    # Initialisiere Dicts
    depth_diff_dict = defaultdict(list)
    used_indices = set()
    
    # Extrahiere notwendige Daten als NumPy-Arrays für schnelleren Zugriff
    coords = np.vstack([transformed_gdf.geometry.x, transformed_gdf.geometry.y]).T
    depths = transformed_gdf['Depth (m)'].values
    dates = transformed_gdf['Date'].values
    
    # Erstelle cKDTree für effiziente Nachbarschaftssuche
    tree = cKDTree(coords)
    
    # Suche Nachbarn innerhalb der Distanzgrenze
    indices = tree.query_ball_tree(tree, max_distance)

    for idx, neighbors in tqdm(enumerate(indices), total=len(indices), desc="Calculating depth difference"):
        point_depth = depths[idx]
        point_date = dates[idx]
        
        for neighbor_idx in neighbors:
            if neighbor_idx > idx:  # Comparing only wth points of higher idx - so points wont be double counted
                neighbor_date = dates[neighbor_idx]
                depth_diff = (point_depth - depths[neighbor_idx]) # abs
                date_pair = tuple(sorted((point_date, neighbor_date)))
                depth_diff_dict[date_pair].append(depth_diff)
                used_indices.add(idx)
                used_indices.add(neighbor_idx)
    
    # Erstelle DataFrame mit den Tiefenunterschieden
    depth_diff_df = pd.DataFrame(dict([(f"{k[0]}-{k[1]}", pd.Series(v)) for k, v in depth_diff_dict.items()]))
    
    # Erstelle ein GeoDataFrame mit allen verwendeten Punkten zur visuellen Kontrolle
    used_points_gdf = transformed_gdf.loc[list(used_indices)].copy()
    
    return depth_diff_df, used_points_gdf




def create_interpolated_coords(sum_df, gps_gdf):
    interpolated_coords = []
    used_idx = set() # set for used GPS-index

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

    # Iterate over sum-data
    for idx, row in tqdm(sum_df.iterrows(), total=sum_df.shape[0]):
        date = row['date']
        utc_full = row['Utc']  # Format HHMMSS.s

        # seperate time in full seconds and decimal part
        try:
            utc_str, decimal_str = utc_full.split('.')
            decimal_part = int(decimal_str)
        except ValueError:
            interpolated_coords.append((None, None))
            continue

        # access gps data for same date
        gps_day = gps_dict.get(date)
        if gps_day is None:
            interpolated_coords.append((None, None))
            continue

        # Find index of same UTC in gps-data
        gps_index = gps_day[gps_day['utc'] == utc_str].index

        if not gps_index.empty:
            idx = gps_index[0]  # Index of exact time
            used_idx.add(idx) # save used GPS-Indexes

            # check for following point
            if idx + 1 < len(gps_day):
                before_point = gps_day.iloc[idx]
                after_point = gps_day.iloc[idx + 1]

                # Interpolation of coordinates
                interp_factor = decimal_part / 10.0
                x_interp = before_point.geometry.x + interp_factor * (after_point.geometry.x - before_point.geometry.x)
                y_interp = before_point.geometry.y + interp_factor * (after_point.geometry.y - before_point.geometry.y)

                interpolated_coords.append((x_interp, y_interp))
            else:
                # if no following point exists
                interpolated_coords.append((None, None))
        else:
            # empty if no fitting gps point is found
            interpolated_coords.append((None, None))

    # Add interpoalted coords to dataframe
    sum_df['Interpolated_Long'] = [coord[0] for coord in interpolated_coords]
    sum_df['Interpolated_Lat'] = [coord[1] for coord in interpolated_coords]

    used_gps_gdf = gps_gdf.iloc[list(used_idx)].copy()


    return sum_df, used_gps_gdf





def detect_and_remove_faulty_depths(geodf_projected: gpd.GeoDataFrame, max_distance: int = 5, threshold: float = 0.5):
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