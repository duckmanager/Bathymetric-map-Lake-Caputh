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


The edge measurments must be of the same day

```



Functions of multibeam_processing

create_dataframe
    input: data_dir: path of data
                                [data has to contain: .mat, .sum of each sample track to analyse with the same file names, as well as .txt of each ecternal GPS per day (at the moment is has to be one external GPS file per sampling day)]
    output: sum_dataframe: pandas.DataFrame with same headers as the .sum file with the most columns (should be all the same)
            sum_header: list of all headers of the longest .sum file

    funtionality: prepares empty dataframe with all necessary columns and list of headers by finding the longest .sum file and extracting the headers (should all have the same length)


assign_data_to_dataframe
    input: data_dir: path of data
            sum_dataframe: pandas DataFrame with headers of .sum file
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
    External GPS has utc at the full second, sonar-GPS has utc with decimal-second. For higher precision, the approximated location at decimal second point gets interpolated. If deciaml second in corrected UTC= .0, the Long lat are used as interpolated long lat. If decimalsecond =/ 0, approximated position gets interpolated. For that, vector between consecutive samples gets created. X and Y - component of vector gets divided by decimal-second factor. Vecotr gets added to original Long Lat data and gets saved in Interpolated_log, - Lat.

    

