# Documentation for QC_point_consistency script
To validate the measuring consistency of the RS-data, close points within a variable spacial distance `calculate_depth_differences_close_points` as well as with a time and space difference `calculate_depth_differences_intersections` get compared and the differences displayed in boxplots.  
QC_point_consistency.py is a separate script to main.py but uses output from main.py. 
Used to determine the Depth difference of close points and show their distribution grouped by dates.


General informations:

filtered_data.csv from multibeam_processing has to be saved in output/multibeam (default). It has to include the filtered sonar measurements as it is used for interpolation. It can contain the edge points, those won't be assessed in the quality control. 
- Important: The Date/Time column has to contain the correct date and roughly correct timestamps. The timestamps can be wrong, as long as they are wrong for the whole survey day.



    filtered_data.csv is read from output/multibeam and converted to GeoDataFrame.
## Options
All neighboring points within a specified radius are getting compared.
Change how close points have to be, to be compared [in meters]: - Default: 0.2m
```
--matching_radius X
```
Spacial difference important in [calculate_depth_differences_intersections](#calculate_depth_differences_intersections) and [calculate_depth_differences_close_points](#calculate_depth_differences_close_points).

Additionally, points, closer than the spacial limit, but temporal further than specified. Default: 5min
Change minimal time difference between points to be compared [in minutes]:
```
--min_time_diff X
```
Temporal difference only important in [calculate_depth_differences_intersections](#calculate_depth_differences_intersections)

---
## Specific functions
### calculate_depth_differences_intersections
Input: 
- transformed_gdf (filtered_data_GeoDataFrame)
output: 
- depth_diff_df: pandas.DataFrame including one column for each date_combination including itself with itself and depth differences measured in these date-combinations within the specified ranges. Column formatting [YYYY-MM-DD-YYYY-MM-DD].
- used_points_gdf: GeoDataFrame including all points compared in 'calculate_depth_differences_intersections'. Formatting is the same as filtered_data_gdf.

Variables: 
- max_distance: max distance between points for them to be compared as neighbor points (default is 0.2 m)
- min_time_diff: min time difference between 'Date/Time' column of two points, for them to be compared as neighbor points. (default is 5min)
    
Functionality: 

Determines all neighbor points in the requirements, so only survey crossings get recognized, not points in direct sequence. Neighbor points are determined using a cKDTree algorithm. The Depth differences get grouped by the combination of dates, they were measured at. Each Depth combination only gets calculated once. This is ensured as only the neighbor of a single point get used for difference calculations, that were measured to an earlier time. This prevents calculation of a-b and b-a the depth difference gets calculated by earlier point - later point.

---
### calculate_depth_differences_close_points
Input:
- same as 'calculate_depth_differences_intersections' (transformed_gdf)

Output: 
- same as 'calculate_depth_differences_intersections' (depth_diff_df, used_points_gdf)

Variables: 
- max_distance: max distance between points for them to be compared as neighbor points

Functionality: 

The functionality is the same as max distance between points for them to be compared as neighbor points'. But no time difference is defined, so all points within the distance get calculated resulting in comparison of consecutive points of the same survey as well as the comparison of crossing points if they are within the distance requirements. The difference groups of different days are the same as in 'calculate_depth_differences_intersections', if max_distance is the same.

---
### compute_statistics_intersections
Input: 
- depth_diff_df - pandas DataFrame, output of 'calculate_depth_differences_intersections'
Output: 
- stats_df: DataFrame with columns [Mean], [StdDev] and row for each date combination
- box: boxplot-informations

Functionality: 

Calculates mean and standartdeviation of 'calculate_depth_differences_intersections' for each date-combination. Outputs a boxplot with distribution of depth differences for each date-combination and the count of depth difference values in each boxplot above it. 
---
### compute_statistics_closepoints
Input: 
- depth_diff_df - pandas DataFrame, output of 'calculate_depth_differences_close_points'
Output: 
- stats_df: DataFrame with columns [Mean], [StdDev] and row for each date combination
- box: boxplot-informations

Functionality: 

Same as 'compute_statistics_intersections' but boxplot labels optimized for 'calculate_depth_differences_close_points'.

