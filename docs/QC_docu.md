# Documentation for QC_point_consistency script

    Used to determine the Depth difference of close points and show their distribution grouped by dates.


    general informations:
    filtered_data.csv from multibeam_processing has to be saved in output/multibeam. It has to include the filtered sonar measurments as it is used for interpoaltion. It can conatin the edge points, those wont be assessed in the quality controle. 
    Important: The Date/Time column has to contain the correct date and roughly correct timestamps. The timestamps can be wrong, as long as they are wrong for the whole survey day.



    filtered_data.csv is read from output/multibeam and converted to GeoDataFrame.

## calculate_depth_differences_intersections
    input: transformed_gdf (filtered_data_GeoDataFrame)
    output: depth_diff_df: pandas.DataFrame including one column for each date_combination including itself with itself and depth differences measured in these date-combinations within the specfied ranges. Column formatting [YYYY-MM-DD-YYYY-MM-DD].
            used_points_gdf: GeoDataFrame including all points compared in 'calculate_depth_differences_intersections'. Formatting is the same as filtered_data_gdf.

    variables: max_distance: max distance between poits for them to be compaired as neighbor points (default is )
                min_time_diff: min time difference between 'Date/Time' column of two points, for them to be compaired as neighbor points. (default is )
    
    functionlity: Determines all neighbor points in the requirments, so only survey crossings get recognized, not points in direct sequence. Neighbor points are determined using a cKDTree algorithm. The Depth differences get grouped by the combination of dates, they were measured at. each Depth combination only gets caluclated once. This is ensured as only the neighbor of a single point get used for difference calculations, that were measured to an earlier time. This prevents calculation of a-b and b-a the depth difference gets calculated by earlier point - later point.

## calculate_depth_differences_close_points
    input: same as 'calculate_depth_differences_intersections' (transformed_gdf)
    output: same as 'calculate_depth_differences_intersections' (depth_diff_df, used_points_gdf)

    variables: max_distance: max distance between poits for them to be compaired as neighbor points

    functionality: The functionality is the same as 'max distance between poits for them to be compaired as neighbor points'. But no time difference is defined, so all points within the distance get calculated resulting in comparison of consecutive points of the same survey as well as the comparison of crossing points if they are within the distance requirements. The difference groups of differnt days are the same as in 'calculate_depth_differences_intersections', if max_distance ist the same.


## compute_statistics_intersections
    input: depth_diff_df - pandas DataFrame, ouput of 'calculate_depth_differences_intersections'
    output: stats_df: DataFrame with columns [Mean], [StdDev] and row for each date combination
            box: boxplot-informations

    functionality: Caluclates mean and standartdeviation of 'calculate_depth_differences_intersections' for each date-combination. Outputs a boxplot with distribution of depth differences for each date-combination and the count of depth difference values in each boxplot above it. 

## compute_statistics_closepoints
input: depth_diff_df - pandas DataFrame, ouput of 'calculate_depth_differences_close_points'
    output: stats_df: DataFrame with columns [Mean], [StdDev] and row for each date combination
            box: boxplot-informations
    functionality: same as 'compute_statistics_intersections' but boxplot labels optimized for 'calculate_depth_differences_close_points'.

