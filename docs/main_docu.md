# Documentation for main.py
No map interpolation is included.
The survey is expected to be done with an additional RTK-GNSS (specifically the Pro Nivo PNR21) but can be adapted to work with a different or without a seperate GNSS.
The RiverSurveyor M9 (RS) data is expected as the result of the RiverSurveyor Live-Software Export in matlab and ASCII format.

General functionality, combined in main.py:

- The echo-sounder data gets linked with the GNSS data and is corrected for faulty data and differences in the recording times (load_data.py).

- To achieve maximum data richness, each depth-beam of the RS gets geolocated individually instead of using it combined in "Bottom Track". (multibeam_location.py)

- In order to optimally include the bank areas of the waterbody in the interpolation, measurements of the bank can be entered, from which points along the bank are created.
Alternativly an general depth of 0m at the bank can be assumed.
Waterlevel changes from different survey dates can be corrected in the data, based on waterlevel measurements. (survey_adjustments.py)

- To optimal filter the echo sounder data for faulty measurements, two detection variants are implemented:
The a  utoamtic filtering, checks each point based on the mean of all sourrounding points (based on user defined variables). (automatic_detection.py)

-   The manual filtering opens each survey as a individual plot, for the user to mark or unmark f  aulty points by hand. (manual_correction.py)
In order to validate the interpolation quality, points can be filtered out at regular variable i  ntervals to create a validation and interpolation data set.

## Options
### Automatic Filtering
Automatic detection will overwrite an existing list of faulty points.
To disable automatic filtering, set:
```
--automatic_detection
```
Automatic filtering works by iterating over every point and comparing its depth with the mean depth of surrounding points.   
To change the parameters [m], which will be applied to select faulty points, use:  
To change the radius to clacluate the mean depth [in meters]:
```
--filtering_max_distance X
```
To change the difference [in meters], which each point needs to have compared to the sorrounding mean, to be marked as faulty, use:
```
--filtering_threshold X
```
specific [description](#detect_and_remove_faulty_depths)

### Manual Filtering
Marked faulty points will be shown in the plots. As long as no points get marked, the faulty-points list won't be changed.  
To only show plots when no filter list of earlier filtering exists, use:
```
--manual_correction_overwrite
```
One filter has to be used.   
To apply an existing filter list without changing it, use:
```
--automatic_detection --manual_correction_overwrite
```
specific [description](#automatic-filtering)

### Waterlevel Corrections
To specify the day to whose water level all measurements are referenced, use:
```
--level_reference_date
```
Otherwise the best fitting will be automatically applied and shown. The best fit is:
- The first survey day that has an equivalent in the waterlevel measurements.
- If no exact match exists, the date closest to a waterlevel emasurement will be used

The waterlevel emasurements have to be at least on the first (or earlier) and last (or later) survey day. 

specific [description](#correct_waterlevel)

### Shore-points
To not use depth measurments from the shore line but to create points of 0m depth instead, use:
```
--edge_points_zero
```
specific [description](#generate_boundary_points)

### Create Validation dataset
The points along the shoreline wont be in the two downsized datasets (interpolation-, validation dataset)  
To change the interval for assignment to the validation dataset, use (every Xth point will be assigned to val-dataset):
```
--validation_sample_rate X
```

To disable creation of an validation dataset, set:
```
--skip_validation_sampling
```
specific [description](#filter_validation_points)

## Specific functions in main.py
### create_dataframe
input: 
- data_dir: path of data [data has to contain: .mat, .sum of each sample track to analyse with the same file names, as well as .txt of each external GPS per day (at the moment is has to be one external GPS file per sampling day)]

output: 
- sum_dataframe: pandas.DataFrame with same headers as the .sum file with the most columns (should be all the same)
- sum_header: list of all headers of the longest .sum file

functionality: 

Prepares empty dataframe with all necessary columns and list of headers by finding the longest .sum file and extracting the headers (should all have the same length)

---
### assign_data_to_dataframe
input: 
- data_dir: path of data
- sum_dataframe: pandas DataFrame with headers of .sum file 
                [file_id, Utc, Lat, Long Sample #, Date/Time, Frequency (MHz), Profile Type, Depth (m), Latitude (deg), Longitude (deg), Heading (deg), Pitch (deg), Roll (deg), BT Depth (m), VB Depth (m), BT Beam1 Depth (m), BT Beam2 Depth (m), BT Beam3 Depth (m), BT Beam4 Depth (m), Track (m), DMG (m), Mean Speed (m/s), Boat Speed (m/s), Direction (deg), Boat Direction (deg)]
- sum_header: list of .sum file header

output: 
- sum_dataframe: pandas DataFrame with all data from the .sum files with corrected Utc times of the sonar internal GPS and the file_id (filename without fileendings) for identification

functionality:

Reads each .sum file and the associated .mat file (same file name). From .mat file, Utc (['GPS']['Utc']) and Gps_Quality (['GPS']['GPS_Quiality']) get extracted and handed over to correct_utc.
Gets back corrected Utc. Saves Utc with associated .sum-sample details to sum_dataframe.

---
### correct_utc (nested function in assign_data_to_dataframe)
input: 
- utc_list: list of utc times extracted from .mat file
- gps_quality_list: list of GPS-Quality extracted from .mat-file for each sample (5 = RTK float, 4 = RTK fixed, 2 = Differential, 1 = Standard, 0 = Invalid)
    
output: 
- utc_array: array containing corrected Utc for each sample

functionality: 

takes first utc sample and creates an optimal timeline with one second interval of same length as utc_list, while keeping the decimal second. 
Fills all faulty utc samples by using the corrected utc timeline. 
Requirements for faulty points: 
- GPS_Quality =0 - impliying bad sample quality
- Utc = 0 implying unusable time
    
This procedure assumes that all other points are reliable Utc-samples.
Switches in single decimal-second can be observed, espacially after Utc=0 without GPS_quality=0 points. maybe an switch of sub-deciaml-seconds results in a rounding change.
If the single decimal-second changes are viewed as unrealiable, corrected_utc_full can be handed back to assign_data_to_dataframe

---
### get_gps_dataframe
input: 
- data_dir :path of data including the .txt files with external GPS data. This cant contain other .txt files than the external GPS. Only one external GPS file per day can be processed. May need to combine them manually. Beware to keep the BESTPOSA and GPZDA order!
(external GPS: external GPS - PNR21 with BESTPOSA and GPZDA at 1Hz sampling rate, others can be implemented by changing the parsing)

output: 
- gps_geodf_projected : GeoDataFrame with date, utc, lat -in WGS84, long  -in WGS84,hgt above mean sea level (meters), DOP (lat), DOP (lon), VDOP and pointgeometry - in UTM 33N - for each second of the external GPS turned on

functionality: 

iterates through the .txt files in data_dir. 
Iterates through each file and saves longitude and latitude in WGS84 from BESTPOSA and couples with date and utc from next following GPZDA. This assumes that first BESTPOSA comes before first GPZDA and they keep the same order. These samples are saved into a DataFrame, converted into a GeoDataFrame and projected from WGS84 to EPSG:25833.

---
### create_interpolated_points
input: 
- sum_df:dataframe including .sum data, file_id and corrected utc
- gps_gdf: GeoDataFrame including external GPS samples with Utc, lat,long /x,y, hgt , DOP (lon), DOP (lat), VDOP, pointgeometry

output: 
- sum_df:DataFrame including .sum data, corrected utc and interpolated Long, -Lat /-X,-Y coordinates
- used_gps_gdf: GeoDataFrame including all data of GPS points matched with sonar data points in same format as 'gps_gdf'

Message: 
- outputs message with number of gps points not used due to DOP and consecutive seconds requirments that had valid sonar data matches.  (default is 0.1 (m))
    
variables:  
- DOP_threshold: threshold for DOP (lat) and DOP (lon) (Dilution of Precision) above  which gps_points wont be used if one is higher.
    
functionality: 

matches sonar samples with external GPS- Lat Long data by date and Utc.
External GPS has utc at the full second, sonar-GPS has utc with decimal-second. For higher precision, the approximated location at decimal second point gets interpolated. Only valid gps points are used. They both have to meet the DOP threshold for lat and lon and have to be exaclty one second apart. For interpolation, vector between valid consecutive samples gets created. X and Y - component of vector get multiplied by decimal-second part. Vector gets added to original Long Lat data and gets saved in Interpolated_log, - Lat. Creates a seperate GeoDataFrame with all rows in gps_gdf used for matching with the sonar data and the same columns as the original gps_gdf dataframe.

Outputs error and amount of sonar samples with missing equivalents in gps data. Possible causes: wrong date in sonar-data, GPS not turned on all the the time.

---
### create_multibeam_points
input: 
- sum_df: GeoDataFrame including .sum data, corrected utc and interpolated Long, -Lat /-X,-Y coordinates

output: 
- transformed_gdf: GeoDataFrame including [file_id], [Utc] - corrected utc, [Date/Time] - Date and time from the sonar internal clock without precise and reliable time, [Beam_type] (VB, Beam1-4), [Depth (m)] - Depth for each beam, [Longitude] -x,[Latitude] -y for each  Depth, [geometry] - pointgeometry in epsg 25833

functionality: 

The function iterates through sum_df rows, georeferecning the Depth of all five beams.
The boat heading direction in azimuth from .sum files gets used as orientation of the beams relative to the central/vertical beam (VB) and is therefore transformed to degrees.
Angles of beams to the VB are assigned in clockwise janus configuration (top view) and 25° angle relative to VB, based on (Sontek: RiverSurveyor S5/M9 System Manual) and (Sontek, a Xylem Brand: Webinar [https://www.youtube.com/watch?v=ukb-B9e5OTY], accessed: 15.02.25). Long/Lat of VB remain the same. Lat/Long of Beam 1-4 get calculated by 
```
new_x = VB_x + tan(radians(25)* VB-Depth) + Beam-y-angle    - with x= x-component/Longitude and y= Beamtyp (1-4)
```
---
### generate_boundary_points
input: 
- shp_data_dir – directory containing the shapefile of the studied waterbody
- point_data_dir – directory containing .csv file (no other .csv files) with measured depth values along the shoreline (e.g., in subfolder outline). The CSV file must contain the following columns: [Longitude], [Latitude] – coordinates in ETRS89 / UTM Zone 33N, [Depth (m)] – measured water depth at the lakeside, and [Date] – date of measurement (only one unique date is allowed) format (MM/DD/YYYY).

Variables:
- edge_points_zero – if True, skips all interpolation and assigns zero depth to all edge points.

output: 
- boundary_gdf – GeoDataFrame (EPSG:25833) containing: [geometry] – point geometry, [Longitude], [Latitude] – X/Y coordinates, [Depth (m)] – water depth value, [file_id] – "artificial_boundary_points", and [Date] – date of depth assignment.

In code variables: 
- spacing – distance between generated edge points in meters (default: 1)
- interpoaltion_distance: distance between measured depth of edge point, within they get connected, in m (default is 150)
- extrapolation_distance: distance to extrapolate measured depth to the side without measured point within interpolation_distance, in m (default is 15)

functionality: 

Generates evenly spaced artificial shoreline points along the polygon outline of the waterbody with the specified spacing. If edge_points_zero is set to True, all points receive a fixed depth of 0 m and no measured data is used. Otherwise, each measured point is matched to its nearest generated shoreline point using geometric distance. If two measured points are within interpolation_distance, all intermediate points are assigned linearly interpolated depth values. If no second measured point is within reach, the depth is locally extrapolated over a distance of up to extrapolation_distance. All shoreline points without assigned depth are removed from the output.


---
### combine_multibeam_edge
input: 
- geodf_projected: GeoDataFrame (output of generate_boundary_points)
- boundary_gdf : GeoDataFrame (output of create_multibeam_points) or (output of detect_and_remove_faulty_depths if error detection is done without edge points)

output: 
- gdf_combined: GeoDataFrame (epsg: 25833) [file_id], [Utc] - only for Beam-points, [Beam_type] - only for beam points, [Depth (m)], [Longitude], [Latitude], [geometry] - point geometry, [Date] - only for artifical_boundary_points

functionality: 

Merges Beam measurements of create_multibeam_points and artifical edge points of generate_boundary_points into one GeoDataFrame. boundary_gdf can be inspected without multibeam points easily.

---
### adjust_depths
input: 
- com_gdf: GeoDataFrame, output of combine_multibeam_edge

output: 
- com_gdf, unchanged except [Depth (m)] has negativ values

functionality: 

changes measured depth distance into neagtiv depth values

---
### correct_waterlevel
input: 
- gdf: GeoDataFrame containing sonar measurement data, including depth and spatial coordinates.
- data_dir: Path to data-folder including one "waterlevel" folder with a CSV file "waterlevel.csv" containing mesaured water levels. "waterlevel.scv" contains "waterlevel" in m and "date" in (DD/MM/YYYY-format)column  The earliest measurement must be of same day or earlier than the first sonar measurment. The latest measuremnt must be at the same day or later than latest sonar-measurement.

Variables:
- reference_day (optional): Reference day for depth correction (MM/DD/YYYY format), if empty, reference day gets determined automatically

output: 
- gdf_corrected: GeoDataFrame with updated depth values:
[Depth (m)]: Corrected depth values, adjusted for water level fluctuations.
[Depth_uncorrected (m)]: Original depth values before correction. Other columns remain unchanged.

message: 

"Reference day from user input: MM/DD/YYYY" - if user gave reference date or  
"Reference day automatically set to: MM/DD/YYYY" - if reference day gt automatically determined
with MM/DD/YYYY being the reference day used for the calculations.

functionality: 

The function corrects measured depth values (Depth (m)) based on water level fluctuations recorded in waterlevel_csv. 
Uses "Date" from, that gets completed from date/Time if it has missing values. Date gets srted by unique values. 
Matches are searched with date in waterlevel. Converts date in numerical values and uses np.interp() to linearly interpolate missing water levels for measurement days. Reference day gets determined, if not given by user input, by Choose the first measurement day with an exact match in the water level dataset. 
If no exact match exists, select the closest date based on proximity to available water level records.
Calculate the depth correction: Depth (m) = Depth (m) - (waterlevel_measurement - waterlevel_reference), except if Depth (m) == 0  (should only happen at artifical edge points). 
Store corrected data in Depth (m) and the original uncorrected data in "Depth_uncorrected (m)".


Combining fault detection functions:

If both techniques should be applied:
    set automatic_detection & manual_overwrite True
If only one technique should be applied - turn this one True, other False
    At least one 
If a sufficent filtering was figuered out and safed in "faulty_points_dir" but want to run the code again, turn both False. The filtering will be applied again. Just make sure the point order stays the same.

---
### detect_and_remove_faulty_depths
input: 
- geodf_projected, GeoDataFrame - output of adjust_depths or create_multibeam_points for correction without edge points
output: 

if automatic_detection = True
- filtered_gdf: GeoDataFrame: all points after error filtering, columns unchanged
- removed_gdf: GeoDataFrame containing all faulty filtered points, original columns + column with the orginal index of each point

if automatic detection = False
- geodf_projected: unchanged gdf / empty gdf

user variable: 
- automatic_detection= True / False -> determines if function will run (True) or return input and empty gdf

input variables: 
- max_distance (int) - radius of mean calculation
- threshold (float) - depth difference above which points get discarded

functionality: 

removes faulty points by comparing to averaged depth of sorrounding points.
Checks if automatic_detection =False - if so, the rest will be skipped. geodf_projected will be skipped and removed_gdf an empty gdf
Saves original index in "orig_index". Iterates through every point. Calculate average depth of all points within "max_distance" radius, excluding evaluated point. Uses cKDTress for neighor identification. If Depth of point differs more than "threshold" from average depth of surrounding points, it gets discarded and saved in removed_gdf, except file_id = artifical_boundary_point. Artifical edge points get recognised for average depth but wont be discarded as faulty points.

---
### interactive_error_correction
input:
- GeoDataFrame (filtered_gdf) containing preprocessed sonar depth data with the following columns:
    - file_id: Identifier for each survey run. Points with the file_id "artificial_boundary_points" are automatically excluded from correction but re-included after processing. These should be placed at the end of the DataFrame to avoid index mismatches.
    - Beam_type: Identifier for the measurement beam (e.g., VB, OB, SB, CB). If this column is missing or contains only a single unique value, all points are treated as vertical beams (VB).
    - Longitude, Latitude: UTM 33N coordinates (in meters), used for calculating the along-track (X-axis) distance.
    - Depth (m): Depth value in meters, plotted along the Y-axis (positive downward).
    
    Optional: previously saved interactive_error_points.csv (typically located in output/multibeam/interactive_error/), containing a column "orig_index" with the original indices of faulty points.

output:
- FILTER_CSV: A CSV file ("faulty_points.csv") containing the indices and full metadata of all manually marked faulty points.
- df_corrected: A filtered GeoDataFrame with all marked faulty points removed and boundary points restored.

user variable:
- manual_overwrite:
    - = True → The interactive plots will open regardless of whether a CSV of faulty points already exists. Previously marked points will be pre-selected for editing.
    - = False → If a CSV exists, the faulty points are automatically removed without showing plots. If no CSV exists, the interactive selection is initiated.

In Code Variables & Settings:
 - threshold_pixels (default: 10): Pixel distance within which a point is considered selectable by click.
- rectangle_selector_minspan (default: 5): Minimum pixel size for activating rectangle selection.
- Interactive Modes:
    - Click Selection: Click near a point to toggle its selection (mark/unmark as faulty).
    - Rectangle Selection: Click and drag to select or deselect all points within the rectangle.
    - Toggle Behavior: Re-clicking or re-selecting a previously marked point removes the marker.


functionality:

This function enables manual inspection and correction of sonar depth measurements through interactive plotting using Matplotlib. For each individual survey run, identified by its file_id, a dedicated plot window is opened to allow visual evaluation of the recorded points. The plot displays all beam types using distinct colors, with along-track distance (in meters) on the X-axis and depth (in meters) on the Y-axis, where 0 meters (representing the water surface) is at the top of the plot.

Upon execution, the function first checks whether a CSV file containing previously marked faulty points exists. If the manual_overwrite parameter is set to False and such a file is found, the listed points are immediately removed from the dataset, and no plot is displayed. If manual_overwrite is True, or if no file exists, the function proceeds to open the interactive plots and initialize the selection interface. Points identified as "artificial_boundary_points" are temporarily removed from the dataset before the correction begins and are re-added once the process is complete.

For each survey run, the cumulative track distance is computed based on UTM33N coordinates (stored in the Longitude and Latitude columns) using Euclidean distances between successive VB (vertical beam) points. If only one beam type is present in the dataset or if the Beam_type column is missing, all points are treated as VB, and their position along the track is determined solely through cumulative distance. In this simplified case, no spatial projection is necessary.

In contrast, when multiple beam types are available, the vertical beam points are used to define the actual survey track. All other beam types (e.g., side or off-center beams) are then projected orthogonally onto a local segment of this VB-defined track. This segment is centered around the current point and includes a configurable number of VB points before and after the current index. The projection is calculated using Shapely’s LineString.project() method, ensuring that each point is positioned accurately along the actual track path, even in cases of turning maneuvers or overlapping tracks.

During the interactive correction phase, users can click near individual points to toggle their selection status. A point is only affected if the click occurs within a specified pixel threshold. Alternatively, users can click and drag to define a rectangle and select or deselect all points within that region. Points marked as faulty are highlighted with a red marker. Previously selected points can be deselected by clicking or re-selecting them.

After reviewing all surveys, the user closes the plots, and the selected faulty points are saved to a CSV file (faulty_points.csv). These points are then removed from the dataset, and the previously separated artificial boundary points are appended again. The resulting cleaned dataset is returned and can be used for further steps in the sonar processing workflow, such as gridding, interpolation, or mapping.

---
### filter_validation_points
input: 
- com_gdf, Geodataframe - containing the full set of bathymetric data points
sample_rate, int - rate at which every point gets assigned to validation dataset
create_validation_data, bool - if set FALSE, dataset splitting will be skipped and empty dataframes will be returned.

output: 
- gdf_interpol_points, Geodataframe: Contains all remaining sonar points used for interpolation, excluding boundary points and regularly sampled validation points. 
- gdf_validation_points, Geodataframe: Contains a spatially uniform subset of sonar points used for validation, selected at the specified interval, also excluding boundary points.

functionality:

This function is designed to support validation of spatial interpolation methods by systematically splitting the dataset into two subsets: one for interpolation and one for validation. Boundary points, which are  used to constrain edge conditions in interpolation, are excluded from both subsets, because of lower point density and pointlessnes in validation. Among the remaining sonar points, every sample_rate-th point is selected as a validation point, ensuring an even spatial distribution. This method helps maintain the representativeness of the validation set across the surveyed area while minimizing bias. The remaining points form the interpolation dataset. This approach is particularly useful for testing interpolation accuracy or for cross-validation during model development.    