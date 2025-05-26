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

def compute_statistics_intersections(depth_diff_df:pd.DataFrame):

    """
    Compute summary statistics and visualize depth differences from intersecting survey lines.

    Calculates the mean and standard deviation of depth differences for each date pair in the input DataFrame and displays the distributions using a boxplot, including point counts above each box.

    args:
        depth_diff_df: DataFrame - depth differences grouped by date combinations, typically from calculate_depth_differences_intersections

    returns:
        stats_df: DataFrame - statistical summary with 'Mean' and 'StdDev' for each date combination
        box: matplotlib object - the generated boxplot object (for optional further use or saving)
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
    y_pos = max(depth_diff_df.max(skipna=True)) * 1.2 if not depth_diff_df.isna().all().all() else 1

    # shows count and mean for used points per boxplot
    for i, col in enumerate(depth_diff_df.columns, start=1):
        n_points = depth_diff_df[col].count()
        mean_val = depth_diff_df[col].mean()
        ax.text(i, y_pos, f"n={n_points}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i, y_pos * 0.97, f"Ø={mean_val:.2f}", ha='center', va='top', fontsize=9, color='black')


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

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm
from shapely.geometry import Point, LineString

def interactive_error_correction(
    faulty_points_dir: Path,
    filtered_gdf: gpd.GeoDataFrame,
    manual_overwrite: bool = True,
    vb_window_size: int = 3  # Anzahl VB-Punkte nach vorne und hinten für lokale Projektion
):
    FILTER_CSV = faulty_points_dir / "faulty_points.csv"
    df = filtered_gdf.copy()

    if 'orig_index' not in df.columns:
        df['orig_index'] = df.index

    if FILTER_CSV.exists():
        if not manual_overwrite:
            removed_points_df = pd.read_csv(FILTER_CSV)
            bad = removed_points_df['orig_index'].tolist()
            boundary_points = df[df['file_id'] == "artificial_boundary_points"]
            df_corrected = pd.concat([df.drop(bad), boundary_points], ignore_index=True)
            print(f"({len(bad)}) marked error points got loaded and removed from data")
            return df_corrected
        else:
            removed_points_df = pd.read_csv(FILTER_CSV)
            loaded_bad = removed_points_df['orig_index'].tolist()
            print(f"({len(loaded_bad)}) marked error points loaded for manual check")
    else:
        loaded_bad = []

    boundary_points = df[df['file_id'] == "artificial_boundary_points"]
    df = df[df['file_id'] != "artificial_boundary_points"]
    bad_indices = []

    for fid in tqdm(df['file_id'].unique(), desc="Messfahrten"):
        subdf = df[df['file_id'] == fid].copy()
        vb_df = subdf[subdf['Beam_type'] == "VB"].copy()
        if vb_df.empty:
            print(f"Messfahrt {fid}: Keine VB-Punkte gefunden. Überspringe diese Fahrfahrt.")
            continue

        vb_coords = np.column_stack((vb_df['Longitude'], vb_df['Latitude']))
        vb_dist = np.r_[0, np.cumsum(np.sqrt(np.sum(np.diff(vb_coords, axis=0)**2, axis=1)))]
        vb_df['cum_dist'] = vb_dist

        pts_all = []
        fig, ax = plt.subplots(figsize=(10, 6))
        markers = {}
        beams = subdf['Beam_type'].unique()
        colors = [
            "#E69F00",  # Orange
            "#56B4E9",  # Blau
            "#009E73",  # Grün
            "#F0E442",  # Gelb
            "#CC79A7",  # Violett
        ]

        for color, beam in zip(colors, beams):
            beam_df = subdf[subdf['Beam_type'] == beam].copy()
            coords = np.column_stack((beam_df['Longitude'], beam_df['Latitude']))
            proj = []

            for i, (idx, row) in enumerate(beam_df.iterrows()):
                x, y = row['Longitude'], row['Latitude']
                beam_point = Point(x, y)

                if beam == "VB":
                    proj_val = vb_df.loc[idx, 'cum_dist']
                else:
                    if idx in vb_df.index:
                        vb_idx = vb_df.index.get_loc(idx)
                    else:
                        vb_idx = np.searchsorted(vb_df.index, idx)
                        vb_idx = np.clip(vb_idx, 0, len(vb_df) - 1)
                    
                    start = max(0, vb_idx - vb_window_size)
                    end = min(len(vb_df), vb_idx + vb_window_size + 1)
                    segment = vb_df.iloc[start:end]
                    if len(segment) >= 2:
                        local_line = LineString(zip(segment['Longitude'], segment['Latitude']))
                        offset = segment['cum_dist'].iloc[0]
                        proj_val = offset + local_line.project(beam_point)
                    else:
                        proj_val = 0

                proj.append(proj_val)

            depth = beam_df['Depth (m)'].values
            ax.scatter(proj, depth, label=beam, color=color, s=15)
            for d, dep, idx in zip(proj, depth, beam_df.index):
                pts_all.append((d, dep, idx, color))

        ax.set_xlabel("Fahrstrecke (m)")
        ax.set_ylabel("Tiefe (m)")
        if pts_all:
            min_depth = min(p[1] for p in pts_all)
            ax.set_ylim(min_depth - 0.5, 0)
        ax.legend()
        plt.title(f"Messfahrt: {fid}")

        for p in pts_all:
            if p[2] in loaded_bad:
                marker, = ax.plot(p[0], p[1], 'ro', markersize=8)
                markers[p[2]] = marker
                if p[2] not in bad_indices:
                    bad_indices.append(p[2])

        def on_click(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            click_disp = ax.transData.transform((event.xdata, event.ydata))
            distances = [np.hypot(*(
                click_disp - ax.transData.transform((p[0], p[1]))
            )) for p in pts_all]
            i = int(np.argmin(distances))
            if distances[i] < 10:
                sel = pts_all[i]
                if sel[2] in markers:
                    markers[sel[2]].remove()
                    del markers[sel[2]]
                    if sel[2] in bad_indices:
                        bad_indices.remove(sel[2])
                else:
                    marker, = ax.plot(sel[0], sel[1], 'ro', markersize=5)
                    markers[sel[2]] = marker
                    bad_indices.append(sel[2])
                fig.canvas.draw()

        def onselect(eclick, erelease):
            x_min, x_max = sorted([eclick.xdata, erelease.xdata])
            y_min, y_max = sorted([eclick.ydata, erelease.ydata])
            for p in pts_all:
                if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max:
                    if p[2] in markers:
                        markers[p[2]].remove()
                        del markers[p[2]]
                        if p[2] in bad_indices:
                            bad_indices.remove(p[2])
                    else:
                        marker, = ax.plot(p[0], p[1], 'ro', markersize=8)
                        markers[p[2]] = marker
                        bad_indices.append(p[2])
            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)
        rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                          button=[1], minspanx=5, minspany=5,
                                          spancoords='pixels', interactive=True)

        plt.show()

    df_corrected = df.drop(bad_indices)
    df_corrected = pd.concat([df_corrected, boundary_points], ignore_index=True)
    df.loc[bad_indices].to_csv(FILTER_CSV, index=False)
    print(f"{len(bad_indices)} points were removed.")
    return df_corrected


    spacing = 1  # distance between artifical edge points in m (CRS EPSG:25833)
    interpolation_distance = 150  # distance to cover between measured points
    extrapolation_distance = 15  # distance to extrapolate measured depth to the side without measured point within interpolation_distance

    # load lake edge and transform to line-geometry
    lake_boundary = gpd.read_file(shp_data_dir / "waterbody.shp").to_crs(
        "EPSG:25833"
    )
    boundary = lake_boundary.unary_union.exterior

    # load measured points and transform into gdf
    edge_points = pd.read_csv(
        point_data_dir / "measured_edgepoints.csv" # oder "measured_edgepoints.csv"
    )  # change name and N E - change in readme
    edge_gdf = gpd.GeoDataFrame(
        edge_points,
        geometry=gpd.points_from_xy(edge_points.E, edge_points.N),
        crs="EPSG:25833",
    )

    # check for uniform dates
    unique_dates = edge_gdf["Date"].unique()
    if len(unique_dates) == 1:
        common_date = unique_dates[0]  # save the date
    else:
        print(
            "Error: All edge point measurements must be from the same date to allow for later correction of water level fluctuations. Please fix manually"
        )
        print("Found dates:", unique_dates)

    # create artifical edge points with equal distances
    distances = np.arange(0, boundary.length, spacing)
    boundary_points = [boundary.interpolate(d) for d in distances]
    boundary_gdf = gpd.GeoDataFrame(geometry=boundary_points, crs="EPSG:25833")
    boundary_gdf["depth"] = (
        np.nan
    )  # depth column for better differentiation of interpolation

    # Assigning nearest neighbor points and find nearest edge point to measurments
    # transforming into an array for faster access
    boundary_coords = np.column_stack(
        (boundary_gdf.geometry.x, boundary_gdf.geometry.y)
    )
    edge_coords = np.column_stack((edge_gdf.geometry.x, edge_gdf.geometry.y))
    # find nearest neighbors of each edge point to assign measured depth to nearest edge point
    boundary_tree = cKDTree(boundary_coords)
    _, edge_gdf["nearest_boundary_idx"] = boundary_tree.query(edge_coords)

    # assigning measured depths to nearest artifical edge points
    boundary_gdf.loc[edge_gdf["nearest_boundary_idx"], "depth"] = edge_gdf[
        "Depth (m)"
    ].values

    # sort edge points along the edge
    edge_gdf = edge_gdf.sort_values("nearest_boundary_idx").reset_index(drop=True)
    edge_gdf["next_point"] = edge_gdf["geometry"].shift(-1)
    edge_gdf["next_depth"] = edge_gdf["Depth (m)"].shift(-1)
    edge_gdf["distance_to_next"] = edge_gdf.geometry.distance(edge_gdf["next_point"])

    # Interpolation between measurment points if <interpoaltion_distance m distance
    for _, row in edge_gdf.iterrows():
        if (
            pd.notna(row["next_depth"])
            and row["distance_to_next"] < interpolation_distance
        ):
            idx1 = row["nearest_boundary_idx"]
            idx2 = boundary_tree.query([row["next_point"].x, row["next_point"].y])[1]
            idx_start, idx_end = min(idx1, idx2), max(idx1, idx2)
            range_idx = range(idx_start, idx_end + 1)
            depth_diff = row["next_depth"] - row["Depth (m)"]
            num_points = len(range_idx)
            depth_step = depth_diff / (num_points - 1) if num_points > 1 else 0
            for i, idx in enumerate(range_idx):
                boundary_gdf.at[idx, "depth"] = row["Depth (m)"] + i * depth_step

    # Berechne den Abstand zum vorherigen Messpunkt (cyclic)
    edge_gdf["prev_point"] = edge_gdf["geometry"].shift(1)
    edge_gdf["prev_depth"] = edge_gdf["Depth (m)"].shift(1)
    edge_gdf["distance_to_prev"] = edge_gdf.geometry.distance(edge_gdf["prev_point"])

    # Extrapolation entlang der Seeumrisslinie (zyklisch) in beide Richtungen

    num_extrap_points = int(extrapolation_distance / spacing)

    for _, row in edge_gdf.iterrows():
        if pd.isna(row["nearest_boundary_idx"]):
            continue
        idx = int(row["nearest_boundary_idx"])
        depth_value = row["Depth (m)"]

        # Extrapolation in forwards dircetion:
        # Condition: no measured points within >=interpolation_distance m
        if (
            pd.isna(row["next_depth"])
            or row["distance_to_next"] >= interpolation_distance
        ):
            for i in range(1, num_extrap_points + 1):
                forward_idx = (idx + i) % len(boundary_gdf)
                # Fülle nur, wenn noch kein Wert gesetzt wurde
                if pd.isna(boundary_gdf.at[forward_idx, "depth"]):
                    boundary_gdf.at[forward_idx, "depth"] = depth_value
                else:
                    break  # Stoppe, wenn bereits ein Wert existiert

        # Extrapolate in backwards direction:
        # Condition: no measured points within >=interpolation_distance m
        if (
            pd.isna(row["prev_depth"])
            or row["distance_to_prev"] >= interpolation_distance
        ):
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
    boundary_gdf["file_id"] = "artificial_boundary_points"
    boundary_gdf["Date"] = (
        common_date  # if fails, the measumrent points used multiple different dates
    )

    return boundary_gdf