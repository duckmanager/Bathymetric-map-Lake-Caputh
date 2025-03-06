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

    output: gps_geodf_projected : GeoDataFrame with date, utc, lat, long and pointgeometry for each second of the external GPS turned on

    functionality: iterates through the .txt files in data_dir. 
    Iterates through each file and saves longitude and latitude in WGS84 from BESTPOSA and couples with date and utc from next following GPZDA. This assumes that first BESTPOSA comes before first GPZDA and they keep the same order. These samples are saved into a DataFrame, converted into a GeoDataFrame and projected from WGS84 to EPSG:25833.


create_interpolated_points
    input: sum_df:dataframe including .sum data, file_id and corrected utc
            gps_df: GeoDataFrame including external GPS samples with Utc, lat,long /x,y

    output: sum_df:DataFrame including .sum data, corrected utc and interpolated Long, -Lat /-X,-Y coordinates

    functionality: matches sonar samples with external GPS- Lat Long data by date and Utc.
    External GPS has utc at the full second, sonar-GPS has utc with decimal-second. For higher precision, the approximated location at decimal second point gets interpolated. If deciaml second in corrected UTC= .0, the Long lat are used as interpolated long lat. If decimalsecond =/ 0, approximated position gets interpolated. For that, vector between consecutive samples gets created. X and Y - component of vector gets divided by decimal-second factor. Vector gets added to original Long Lat data and gets saved in Interpolated_log, - Lat.



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

    input variables: spacing: distance between each edge point, in m
                    interpoaltion_distance: distance between measured depth of edge point, within they get connected, in m
                    extrapolation_distance: distance to extrapolate measured depth to the side without measured point within interpolation_distance, in m

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

    input variables: max_distance (int) - radius of mean calculation
                    threshold (float) - depth difference above which points get discarded

    functionality: removes faulty points by comparing to averaged depth of sorrounding points.
    Iterates through every point. Calculate average depth of all points within "max_distance" radius, excluding evaluated point. Uses cKDTress for neighor identification. If Depth of point differs more than "threshold" from average depth of surrounding points, it gets discarded and saved in removed_gdf, except file_id = artifical_boundary_point. Artifical edge points get recognised for average depth but wont be discarded as faulty points.









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

    variables: max_distance: max distance between poits for them to be compaired as neighbor points
                min_time_diff: min time difference between 'Date/Time' column of two points, for them to be compaired as neighbor points.
    
    functionlity: Determines all neighbor points in the requirments, so only survey crossings get recognized, not points in direct sequence. Neighbor points are determined using a cKDTree algorithm. The Depth differences get grouped by the combination of dates, they were measured at. each Depth combination only gets caluclated once. This is ensured as only the neighbor of a single point get used for difference calculations, that were measured to an earlier time. This prevents calculation of a-b and b-a. The depth difference gets calculated by earlier point - later point.

    calculate_depth_differences_close_points
    input: same as 'calculate_depth_differences_intersections' (transformed_gdf)
    output: same as 'calculate_depth_differences_intersections' (depth_diff_df, used_points_gdf)

    variables: max_distance: max distance between poits for them to be compaired as neighbor points

    functionality: The functionality is the same as 'max distance between poits for them to be compaired as neighbor points'. But no min time difference is defined, so all points within the distance get calculated resulting in comparison of consecutive points of the same survey if they are within the distance requirements. The difference groups of differnt days are the same as in 'calculate_depth_differences_intersections', if max_distance ist the same.


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