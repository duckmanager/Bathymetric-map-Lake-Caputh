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


detect_and_remove_faulty_depths
    input: geodf_projected, GeoDataFrame - output of adjust_depths or create_multibeam_points for correction without edge points
    output: filtered_gdf: GeoDataFrame: all points after error filtering, columns unchanged
            removed_gdf: GeoDataFrame containing all faulty filtered points, columns unchanged

    input variables: max_distance (int) - radius of mean calculation (default is )
                    threshold (float) - depth difference above which points get discarded (default is )

    functionality: removes faulty points by comparing to averaged depth of sorrounding points.
    Iterates through every point. Calculate average depth of all points within "max_distance" radius, excluding evaluated point. Uses cKDTress for neighor identification. If Depth of point differs more than "threshold" from average depth of surrounding points, it gets discarded and saved in removed_gdf, except file_id = artifical_boundary_point. Artifical edge points get recognised for average depth but wont be discarded as faulty points.

filter_validation_points




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

    functionality: The function corrects measured depth values (Depth (m)) based on water level fluctuations recorded in waterlevel_csv. Uses "Date" from, that gets completed from date/Time if it has missing values. Date gets srted by unique values. Matches are searched with date in waterlevel. Converts date in numerical values and uses np.interp() to linearly interpolate missing water levels for measurement days. Reference day gets determined, if not given by user input, by Choose the first measurement day with an exact match in the water level dataset. If no exact match exists, select the closest date based on proximity to available water level records.Calculate the depth correction: Depth (m) = Depth (m) - (waterlevel_measurement - waterlevel_reference), except if Depth (m) == 0  (should only happen at artifical edge points). Store corrected data in Depth (m) and the original uncorrected data in "Depth_uncorrected (m)".



Functions of interactive_error_correction

libraries: 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm


    input: csv-file containing the geodataframe with the preprocessed sonar data, containing at least the colloumns:
            file_id: Identifier for each survey run. Points with the file_id "artificial_boundary_points" will be automatically excluded.
                        All "artifical_boundary_points" should be by default from earlier processing be at the end of the list. Else problems with the indices could arise.
            Beam_type: Identifier for the measurement beam (VB (Vertical Beam, Beam 1-4))
            Longitude and Latitude: UTM32N coordinates (in meters) used to compute the along-track distance.
            Depth (m): Depth in meters

    output: FILTER_CSV: A CSV file containing the indices and all data as the original csv-file from all selected points
            df_corrected: DataFrame with original data excluding all selected points

    Input Variables & Settings:
            threshold_pixels (default: 10): The selection threshold in pixel units. A point is only toggled (selected/deselected) if the click is within this distance from it.
            rectangle_selector_minspan (default: 5): The minimum span in pixels required for the rectangle selection to activate.
            Interactive Selection Modes:
                Click Selection: Clicking near a point toggles its selection. The nearest point is determined based on both x and y coordinates in display space.
                Rectangle Selection: By dragging a rectangle over the plot, all points within that area are toggled.
            Toggle Behavior: Clicking on an already marked point (or selecting it via rectangle) removes its selection.

        functionality:
            This tool provides a manual error-correction for sonar measurment points via an interactive Matplotlib window. One figure per survey file gets shown. Points can be selected by clicking or click and drag to select multiple points wihtin a rectangle. Selected points are marked as red. By selecting a point a again, it gets unselected. The plot shows all Beam-Types in different colors . On the X-axis this distance between following point in meters is shown. On the Y-axis the Depth in m ist shown, starting at 0m (waterlevel).
                Initialization:
                    Upon startup, the tool checks if a CSV file (FILTER_CSV) with previously marked faulty points exists. If found, those points are automatically removed from the dataset, after removing artifical boundary points first, and adiing them back in after.Further interactive filtering is skipped.
                    If no FILTER_CSV exists, the tool reads the DATA_FILE, excludes any points with the file_id "artificial_boundary_points" and saves them seperately, so they wont get processed in error correction.
                Data Segmentation & Plotting:
                    The data is segmented by each unique survey run (file_id).
                    For each survey run, the tool computes the cumulative along-track distance using the UTM32N coordinates (Longitude, Latitude) to serve as the x-axis.
                    Scatter plots are generated (without connecting lines) where each Beam_type is displayed in a distinct color.
                    The y-axis is configured so that the water surface (0 m) is at the top and the deepest measurement is extended by a 0.5 m buffer (i.e., y-limit is set from min_depth – 0.5 to 0).
                Interactive Point Selection:
                    Click Selection:
                        The tool listens for click events. It converts the click position into display (pixel) coordinates and calculates the distance to each point.
                        If the nearest point is within the threshold (e.g., 10 pixels), its selection state is toggled (red marker is added or removed).
                    Rectangle Selection:
                        A RectangleSelector is enabled. By dragging a rectangle over the plot, all points within that defined area are toggled (i.e., marked as faulty if unmarked, or unmarked if already selected).
                    Finalization:
                         After all survey runs have been reviewed (each plot is closed to proceed to the next), the tool saves the indices of all marked (faulty) points to FILTER_CSV.
                         The faulty points are removed from the dataset, the artifical_boundary_points are added to the dataset again and the resulting filtered dataset is then ready for further processing

    














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
```