## Data
### .mat files
```
{
    ...,
    "System" = {
        "Heading" = array of float: degree of where the measuring unit is heading,
        "True_North_ADP_Heading" = array of float: degree of where the compute unit is heading with correction for geographic north,
        ...,
    },
    "Summary" = {
        "Boat_Vel" = array of arrays of four floats: first index seems to be boat velocity in m/s,
        "Boat Direction (deg)" = array of float: degree of where the measuring unit is heading in compass degreees /or the azimuth,
        ...,
    }
    "BottomTrack" = {
        "VB_Depth" = array of float: depth of vertical beam in m,
        "BT_Depth" = array of float: depth of bottom track (mix of all four outer beams),
        "BT_Vel" = array of arrays of four floats: east, north, up, difference,
        "BT_Beam_Depth" = array of arrays of four floats: depth per (outer) beam in m/s, 
        "BT_Frequency" = array of float: frequency,
        "Units" = dict of units
    }

    {
    "RawGPSData" = {
        "GgaUTC" = array of decimal second step UTC times, always starting with the same time?
                        pobably the start of UTC in GPS - maybe the cause of the problems or the solutions why its not a problem?


    }



    }
}


Functions of multibeam_processing
  
    general informations:
    Combine sonar and GPS data, optimize data quality and add seperatly measured edge points to create a dataset usable in GIS-interpolations.

    basic starter guide for Riversurveyor M9 and ProNivo PNR21
    Put ASCII and matlab export (has to contain correct date) of each survey in /data
    Put the .txt of ProNivo PNR21 for each measurment day in /data

    - waterbody.shp in data/shp_files
    - edge_depth_measurements.csv with edge measuremnts (in m) in data/outline

    - waterlevel.csv with ["date"],["waterlevel"] - (youngest waterlevel cant be older than earliest survey, oldest waterlevel cant be younger than last survey), in data/waterlevel
    - variables for automatic filtering


required packages:
- pathlib
- pandas
- geopandas
- shapely.geometry
- numpy
- pymatreader
- datetime
- scipy.spatial
- collections
- tqdm


create_dataframe
    input: data_dir: path of data
                                [data has to contain: .mat, .sum of each sample track to analyse with the same file names, as well as .txt of each ecternal GPS per day (at the moment is has to be one external GPS file per sampling day)]
    output: sum_dataframe: pandas.DataFrame with same headers as the .sum file with the most columns (should be all the same)
            sum_header: list of all headers of the longest .sum file

    funtionality: prepares empty dataframe with all necessary columns and list of headers by finding the longest .sum file and extracting the headers (should all have the same length)


assign_data_to_dataframe
    input: data_dir: path of data
            sum_dataframe: pandas DataFrame with headers of .sum file 
                [file_id, Utc, Lat, Long Sample #, Date/Time, Frequency (MHz), Profile Type, Depth (m), Latitude (deg), Longitude (deg), Heading (deg), Pitch (deg), Roll (deg), BT Depth (m), VB Depth (m), BT Beam1 Depth (m), BT Beam2 Depth (m), BT Beam3 Depth (m), BT Beam4 Depth (m), Track (m), DMG (m), Mean Speed (m/s), Boat Speed (m/s), Direction (deg), Boat Direction (deg)]
            sum_header: list of .sum file header

    output: sum_dataframe: pandas DataFrame with all data from the .sum files with corrected Utc times of the sonar internal GPS and the file_id (filename without fileendings) for identification

    functionality: reads each .sum file and the associated .mat file (same file name). From .mat file, Utc (['GPS']['Utc']) and Gps_Quality (['GPS']['GPS_Quiality']) get extracted and handed over to correct_utc.
    Gets back corrected Utc. Saves Utc with associated .sum-sample details to sum_dataframe.


correct_utc (nested function in assign_data_to_dataframe)
    input: utc_list: list of utc times extracted from .mat file
            gps_quality_list: list of GPS-Quality extracted from .mat-file for each sample (5 = RTK float, 4 = RTK fixed, 2 = Differential, 1 = Standard, 0 = Invalid)
    
    output: utc_array: array containing corrected Utc for each sample

    functionality: takes first utc sample and creates an optimal timeline with one second interval of same length as utc_list, while keeping the decimal second. 
    Fills all faulty utc samples by using the corrected utc timeline. 
    Requirements for faulty points: 
            GPS_Quality =0 - impliying bad sample quality
            Utc = 0 implying unuseable time
    
    This procedure assumes that all other points are reliable Utc-samples.
    Switches in single decimal-second can be observed, espacially after Utc=0 without GPS_quality=0 points. maybe an switch of sub-deciaml-seconds results in a rounding change.
    If the single decimal-second changes are viewed as unrealiable, corrected_utc_full can be handed back to assign_data_to_dataframe


get_gps_dataframe
    input: data_dir :path of data including the .txt files with external GPS data. This cant contain other .txt files than the external GPS. Only one external GPS file per day can be processed. May need to combine them amnually. Take care of keeping the BESTPOSA and GPZDA order!
                external GPS: external GPS - PNR21 with BESTPOSA and GPZDA at 1Hz sampling rate

    output: gps_geodf_projected : GeoDataFrame with date, utc, lat -in WGS84, long  -in WGS84,hgt above mean sea level (meters), DOP (lat), DOP (lon), VDOP and pointgeometry - in UTM 32N - for each second of the external GPS turned on

    functionality: iterates through the .txt files in data_dir. 
    Iterates through each file and saves longitude and latitude in WGS84 from BESTPOSA and couples with date and utc from next following GPZDA. This assumes that first BESTPOSA comes before first GPZDA and they keep the same order. These samples are saved into a DataFrame, converted into a GeoDataFrame and projected from WGS84 to EPSG:25833.


create_interpolated_points
    input: sum_df:dataframe including .sum data, file_id and corrected utc
            gps_gdf: GeoDataFrame including external GPS samples with Utc, lat,long /x,y, hgt , DOP (lon), DOP (lat), VDOP, pointgeometry

    output: sum_df:DataFrame including .sum data, corrected utc and interpolated Long, -Lat /-X,-Y coordinates
            used_gps_gdf: GeoDataFrame including all data of GPS points matched with sonar data points in same format as 'gps_gdf'

    Message: outputs message with number of gps points not used due to DOP and consecutive seconds requirments that had valid sonar data matches.  (default is 0.1 (m))
    
    variables:  DOP_threshold: threshold for DOP (lat) and DOP (lon) (Dilution of Precision) above  which gps_points wont be used if one is higher.
    functionality: matches sonar samples with external GPS- Lat Long data by date and Utc.
    External GPS has utc at the full second, sonar-GPS has utc with decimal-second. For higher precision, the approximated location at decimal second point gets interpolated. Only valid gps points are used. They both have to meet the DOP threshold for lat and lon and have to be exaclty one second apart. For interpolation, vector between valid consecutive samples gets created. X and Y - component of vector get multiplied by decimal-second part. Vector gets added to original Long Lat data and gets saved in Interpolated_log, - Lat. Creates a seperate GeoDataFrame with all rows in gps_gdf used for matching with the sonar data and the same columns as the original gps_gdf dataframe.
    Outputs error and amount of sonar samples with missing equivalents in gps data. Possible causes: wrong date in sonar-data, GPS not turned on all the time.
    



create_multibeam_points
    input: sum_df: GeoDataFrame including .sum data, corrected utc and interpolated Long, -Lat /-X,-Y coordinates------------------------------------------------- is it gdf?

    output: transformed_gdf: GeoDataFrame including [file_id], [Utc] - corrected utc, [Date/Time] - Date and time from the sonar internal clock without precise and reliable time, [Beam_type] (VB, Beam1-4), [Depth (m)] - Depth for each beam, [Longitude] -x,[Latitude] -y for each  Depth, [geometry] - pointgeometry in epsg 25833

    functionality: The function iterates through sum_df rows, georeferecning the Depth of all five beams.
    The boat heading direction in azimuth from .sum files gets used as orientation of the beams relative to the central/vertical beam (VB) and is therefore transformed to degrees.
    Angles of beams to the VB are assigned in clockwise janus configuration (top view) and 25° angle relative to VB, based on (Sontek: RiverSurveyor S5/M9 System Manual) and (Sontek, a Xylem Brand: Webinar [https://www.youtube.com/watch?v=ukb-B9e5OTY], accessed: 15.02.25). Long/Lat of VB remain the same. Lat/Long of Beam 1-4 get calculated by 
        new_x = VB_x + tan(radians(25)* VB-Depth) + Beam-y-angle    - with x= x-component/Longitude and y= Beamtyp (1-4)
    


generate_boundary_points
    input: data_dir - containing folder 'shp_files' with shp_file of studyed area
                    - containing folder 'outline' with .csv with depth of all edge points measured (They dont have to cover all around the water body)
                                        .csv has to contain for each point: [N] and [E] /y and x in ETRS89 / UTM zone 33N, [Depth (m)], [Date] - Date of edge point measuring (only one Date is possible) 

    output: boundary_gdf: GeoDataFrame (epsg:25833): [geometry]- point geometry of each point, [Longitude], [Latitude] - x/y of each point in EPSG:25833, [Depth (m)] - Depth of each point, [file_id] = "artificial_boundary_points", [Date] - day of point measurement

    input variables: spacing: distance between each edge point, in m (default is 1)
                    interpoaltion_distance: distance between measured depth of edge point, within they get connected, in m (default is 150)
                    extrapolation_distance: distance to extrapolate measured depth to the side without measured point within interpolation_distance, in m (default is 15)

    functionality: creates artifical edge points for enhanced interpoaltion precision at the edges. Only interpolates near measured points.
    Creates points with "spacing" distance between on the outline of -shp file of the water body. Uses cKDTress for neighor identification. Connects each measured Depth point with the closest edge point. For measured points within "interpoaltion_distance" to each other, each point in between gets a depth assigned. The Depth gets linearly interpolated by the difference in depth between the measured points and the number of artifical edge points between. If no meassured point is within "interpolation_distance", edge points ~ within "extrapolation_distance" get assigend the same depth as last measured. All edge points without depth assigned get discraded.



combine_multibeam_edge
    input: geodf_projected: GeoDataFrame (output of generate_boundary_points)
            boundary_gdf : GeoDataFrame (output of create_multibeam_points) or (output of detect_and_remove_faulty_depths if error detection is done without edge points)

    output: gdf_combined: GeoDataFrame (epsg: 25833) [file_id], [Utc] - only for Beam-points, [Beam_type] - only for beam points, [Depth (m)], [Longitude], [Latitude], [geometry] - point geometry, [Date] - only for artifical_boundary_points

    functionality: merges Beam measurements of create_multibeam_points and artifical edge points of generate_boundary_points into one GeoDataFrame. boundary_gdf can be inspected without multibeam points easily.


adjust_depths
    input: com_gdf: GeoDataFrame, output of combine_multibeam_edge
    output: com_gdf, unchanged except [Depth (m)] has negativ values

    functionality: changes measured depth distance into neagtiv depth values

filter_validation_points
    input: com_gdf, Geodataframe - containing the full set of bathymetric data points
            sample_rate, int - rate at which every point gets assigned to validation dataset
            create_validation_data, bool - if set FALSE, dataset splitting will be skipped and empty dataframes will be returned.

    output: gdf_interpol_points, Geodataframe: Contains all remaining sonar points used for interpolation, excluding boundary points and regularly sampled validation points. 
            gdf_validation_points, Geodataframe: Contains a spatially uniform subset of sonar points used for validation, selected at the specified interval, also excluding boundary points.

    functionality:
    This function is designed to support validation of spatial interpolation methods by systematically splitting the dataset into two subsets: one for interpolation and one for validation. Boundary points, which are  used to constrain edge conditions in interpolation, are excluded from both subsets, because of lower point density and pointlessnes in validation. Among the remaining sonar points, every sample_rate-th point is selected as a validation point, ensuring an even spatial distribution. This method helps maintain the representativeness of the validation set across the surveyed area while minimizing bias. The remaining points form the interpolation dataset. This approach is particularly useful for testing interpolation accuracy or for cross-validation during model development.

correct_waterlevel
    input: gdf: GeoDataFrame containing sonar measurement data, including depth and spatial coordinates.
            data_dir: Path to data-folder including one "waterlevel" folder with a CSV file "waterlevel.csv" containing mesaured water levels. "waterlevel.scv" contains "waterlevel" in m and "date" in (DD/MM/YYYY-format)column  The earliest measurement must be of same day or earlier than the first sonar measurment. The latest measuremnt must be at the same day or later than latest sonar-measurement.
            reference_day (optional): Reference day for depth correction (MM/DD/YYYY format), if empty, reference day gets determined automatically

    output: gdf_corrected: GeoDataFrame with updated depth values:
                                    [Depth (m)]: Corrected depth values, adjusted for water level fluctuations.
                                    [Depth_uncorrected (m)]: Original depth values before correction.
                                        Other columns remain unchanged.

            message: "Reference day from user input: MM/DD/YYYY" - if user gave reference date
                       or
                        "Reference day automatically set to: MM/DD/YYYY" - if reference day gt automatically determined
                        with MM/DD/YYY being the reference day used for the calculations.

    functionality: The function corrects measured depth values (Depth (m)) based on water level fluctuations recorded in waterlevel_csv. 
    Uses "Date" from, that gets completed from date/Time if it has missing values. Date gets srted by unique values. 
    Matches are searched with date in waterlevel. Converts date in numerical values and uses np.interp() to linearly interpolate missing water levels for measurement days. Reference day gets determined, if not given by user input, by Choose the first measurement day with an exact match in the water level dataset. 
    If no exact match exists, select the closest date based on proximity to available water level records.
    Calculate the depth correction: Depth (m) = Depth (m) - (waterlevel_measurement - waterlevel_reference), except if Depth (m) == 0  (should only happen at artifical edge points). 
    Store corrected data in Depth (m) and the original uncorrected data in "Depth_uncorrected (m)".


Combining fault detection functions:
If both techniques should be applied:
    automatic_detection & manual_overwrite(in interactive_error_correction) = True
If only one technique should be applied - turn this one True, other False

If a a sufficent filtering was figuered out and safed in "faulty_points_dir" but want to run the code again, turn both False. The filtering will be applied again. Just make sure the point order stays the same.



detect_and_remove_faulty_depths
    input: geodf_projected, GeoDataFrame - output of adjust_depths or create_multibeam_points for correction without edge points
    output: if automatic_detection = True
            filtered_gdf: GeoDataFrame: all points after error filtering, columns unchanged
            removed_gdf: GeoDataFrame containing all faulty filtered points, original columns + column with the orginal index of each point

            if automatic detection = False
            geodf_projected: unchanged gdf
            empty gdf
    user variable: automatic_detection= True / False -> determines if function will run (True) or return input and empty gdf
    input variables: max_distance (int) - radius of mean calculation (default is )
                    threshold (float) - depth difference above which points get discarded (default is )

    functionality: removes faulty points by comparing to averaged depth of sorrounding points.
    Checks if automatic_detection =False - if so, the rest will be skipped. geodf_projected will be skipped and removed_gdf an empty gdf
    Saves original index in "orig_index". Iterates through every point. Calculate average depth of all points within "max_distance" radius, excluding evaluated point. Uses cKDTress for neighor identification. If Depth of point differs more than "threshold" from average depth of surrounding points, it gets discarded and saved in removed_gdf, except file_id = artifical_boundary_point. Artifical edge points get recognised for average depth but wont be discarded as faulty points.


interactive_error_correction

libraries: 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm

    input:
        GeoDataFrame (filtered_gdf) containing preprocessed sonar depth data with the following columns:
            - file_id: Identifier for each survey run. Points with the file_id "artificial_boundary_points" are automatically excluded from correction but re-included after processing. These should be placed at the end of the DataFrame to avoid index mismatches.
            - Beam_type: Identifier for the measurement beam (e.g., VB, OB, SB, CB). If this column is missing or contains only a single unique value, all points are treated as vertical beams (VB).
            - Longitude, Latitude: UTM 32N coordinates (in meters), used for calculating the along-track (X-axis) distance.
            - Depth (m): Depth value in meters, plotted along the Y-axis (positive downward).
        Optional: previously saved interactive_error_points.csv (typically located in output/multibeam/interactive_error/), containing a column "orig_index" with the original indices of faulty points.

    output:
        - FILTER_CSV: A CSV file ("faulty_points.csv") containing the indices and full metadata of all manually marked faulty points.
        - df_corrected: A filtered GeoDataFrame with all marked faulty points removed and boundary points restored.

    user variable:
        - manual_overwrite:
            = True → The interactive plots will open regardless of whether a CSV of faulty points already exists. Previously marked points will be pre-selected for editing.
            = False → If a CSV exists, the faulty points are automatically removed without showing plots. If no CSV exists, the interactive selection is initiated.

    Input Variables & Settings:
        - threshold_pixels (default: 10): Pixel distance within which a point is considered selectable by click.
        - rectangle_selector_minspan (default: 5): Minimum pixel size for activating rectangle selection.
        - Interactive Modes:
            - Click Selection: Click near a point to toggle its selection (mark/unmark as faulty).
            - Rectangle Selection: Click and drag to select or deselect all points within the rectangle.
            - Toggle Behavior: Re-clicking or re-selecting a previously marked point removes the marker.


        functionality:
    This function enables manual inspection and correction of sonar depth measurements through interactive plotting using Matplotlib. For each individual survey run, identified by its file_id, a dedicated plot window is opened to allow visual evaluation of the recorded points. The plot displays all beam types using distinct colors, with along-track distance (in meters) on the X-axis and depth (in meters) on the Y-axis, where 0 meters (representing the water surface) is at the top of the plot.

    Upon execution, the function first checks whether a CSV file containing previously marked faulty points exists. If the manual_overwrite parameter is set to False and such a file is found, the listed points are immediately removed from the dataset, and no plot is displayed. If manual_overwrite is True, or if no file exists, the function proceeds to open the interactive plots and initialize the selection interface. Points identified as "artificial_boundary_points" are temporarily removed from the dataset before the correction begins and are re-added once the process is complete.

    For each survey run, the cumulative track distance is computed based on UTM32N coordinates (stored in the Longitude and Latitude columns) using Euclidean distances between successive VB (vertical beam) points. If only one beam type is present in the dataset or if the Beam_type column is missing, all points are treated as VB, and their position along the track is determined solely through cumulative distance. In this simplified case, no spatial projection is necessary.

    In contrast, when multiple beam types are available, the vertical beam points are used to define the actual survey track. All other beam types (e.g., side or off-center beams) are then projected orthogonally onto a local segment of this VB-defined track. This segment is centered around the current point and includes a configurable number of VB points before and after the current index. The projection is calculated using Shapely’s LineString.project() method, ensuring that each point is positioned accurately along the actual track path, even in cases of turning maneuvers or overlapping tracks.

    During the interactive correction phase, users can click near individual points to toggle their selection status. A point is only affected if the click occurs within a specified pixel threshold. Alternatively, users can click and drag to define a rectangle and select or deselect all points within that region. Points marked as faulty are highlighted with a red marker. Previously selected points can be deselected by clicking or re-selecting them.

    After reviewing all surveys, the user closes the plots, and the selected faulty points are saved to a CSV file (faulty_points.csv). These points are then removed from the dataset, and the previously separated artificial boundary points are appended again. The resulting cleaned dataset is returned and can be used for further steps in the sonar processing workflow, such as gridding, interpolation, or mapping.

    


Functions of QC_closepoints:
    Used to determine the Depth difference of close points and show their distribution grouped by dates.

    used libraries:
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely import wkt
    from pathlib import Path
    from tqdm import tqdm
    from scipy.spatial import cKDTree
    import numpy as np
    from collections import defaultdict

    general informations:
    filtered_data.csv from multibeam_processing has to be saved in output/multibeam. It has to include the filtered sonar measurments as it is used for interpoaltion. It can conatin the edge points, those wont be assessed in the quality controle. 
    Important: The Date/Time column has to contain the correct date and roughly correct timestamps. The timestamps can be wrong, as long as they are wrong for the whole survey day.



    filtered_data.csv is read from output/multibeam and converted to GeoDataFrame.

    calculate_depth_differences_intersections
    input: transformed_gdf (filtered_data_GeoDataFrame)
    output: depth_diff_df: pandas.DataFrame including one column for each date_combination including itself with itself and depth differences measured in these date-combinations within the specfied ranges. Column formatting [YYYY-MM-DD-YYYY-MM-DD].
            used_points_gdf: GeoDataFrame including all points compared in 'calculate_depth_differences_intersections'. Formatting is the same as filtered_data_gdf.

    variables: max_distance: max distance between poits for them to be compaired as neighbor points (default is )
                min_time_diff: min time difference between 'Date/Time' column of two points, for them to be compaired as neighbor points. (default is )
    
    functionlity: Determines all neighbor points in the requirments, so only survey crossings get recognized, not points in direct sequence. Neighbor points are determined using a cKDTree algorithm. The Depth differences get grouped by the combination of dates, they were measured at. each Depth combination only gets caluclated once. This is ensured as only the neighbor of a single point get used for difference calculations, that were measured to an earlier time. This prevents calculation of a-b and b-a the depth difference gets calculated by earlier point - later point.

    calculate_depth_differences_close_points
    input: same as 'calculate_depth_differences_intersections' (transformed_gdf)
    output: same as 'calculate_depth_differences_intersections' (depth_diff_df, used_points_gdf)

    variables: max_distance: max distance between poits for them to be compaired as neighbor points

    functionality: The functionality is the same as 'max distance between poits for them to be compaired as neighbor points'. But no time difference is defined, so all points within the distance get calculated resulting in comparison of consecutive points of the same survey as well as the comparison of crossing points if they are within the distance requirements. The difference groups of differnt days are the same as in 'calculate_depth_differences_intersections', if max_distance ist the same.


compute_statistics_intersections
    input: depth_diff_df - pandas DataFrame, ouput of 'calculate_depth_differences_intersections'
    output: stats_df: DataFrame with columns [Mean], [StdDev] and row for each date combination
            box: boxplot-informations ?????????????????????????????????????????????????

    functionality: Caluclates mean and standartdeviation of 'calculate_depth_differences_intersections' for each date-combination. Outputs a boxplot with distribution of depth differences for each date-combination and the count of depth difference values in each boxplot above it. 

compute_statistics_closepoints
input: depth_diff_df - pandas DataFrame, ouput of 'calculate_depth_differences_close_points'
    output: stats_df: DataFrame with columns [Mean], [StdDev] and row for each date combination
            box: boxplot-informations ?????????????????????????????????????????????????
    functionality: same as 'compute_statistics_intersections' but boxplot labels optimized for 'calculate_depth_differences_close_points'.



temperature_plot
process and show temeprature- depth profiles form manual measurements to decide about proceeding in manual temperature correction of echo sounder data

Required packages:
- pandas
- numpy
- matplotlib.pyplot
- pathlib

temperature_profile_plotter.py

Functionality overview:
This script processes multiple temperature–depth profiles stored in CSV format.
Each CSV file must contain a column "Depth" and one or more columns representing
individual temperature measurements at varying depths. The script performs interpolation 
of the profiles on a uniform vertical grid, plots all valid profiles along with the mean 
temperature curve, and saves the output as a .png figure.

Input:
- data_dir: path to folder containing CSV files  
  Each file must:
  - Use ";" as separator
  - Contain a column labeled "Depth" (in meters)
  - Include one or more additional columns with temperature values (°C) for each profile

Output:
- PNG files: temperature–depth plots with all valid profiles and the average temperature curve  
- Each plot is saved as:  
  filename_temperaturschichtung.png in the same directory as the input CSV

Functionality:

1. File handling:
   - Iterates through all .csv files in the specified folder
   - Skips files that do not contain a "Depth" column

2. Interpolation:
   - Creates a uniform vertical grid (default step: 0.1 m) based on the deepest measurement 
     in the current file
   - Interpolates each profile individually using linear interpolation (numpy.interp)
   - Only interpolates within the valid depth range of each profile; outside values are set to NaN

3. Averaging:
   - Calculates a mean temperature profile from all valid interpolated profiles, ignoring NaNs
   - Additionally computes:
     - Mean temperature across the entire depth
     - Mean temperature for the top 4 meters

4. Plotting:
   - Plots each individual temperature profile as a line
   - Adds the mean temperature curve as a thick black semi-transparent line
   - Displays average temperatures (total and 0–4 m) as a labeled text box in the top-left corner
   - Y-axis is inverted to display depth increasing downwards
   - X-axis limits are dynamically set based on min/max temperature values
   - Includes grid, axis labels, title, and legend (with average curve labeled only as "Ø Profil",
     no temperature shown in legend)

5. Export:
   - Saves the figure as a high-resolution .png file in the same directory
   - Clears the figure before processing the next file

```