# Script to process data from the Riversurveyor M9 echo-sounder to optimize for bathymetric mapping. 

Einleitender satz  
**more info at the [docs](/docs/main_docu.md#create_interpolated_points)**

## Table of contents
- [Core functionality](#core-functionality)
- [Additional functions](#additional-functions)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Overview](#overview)

## What this project can do for you
explain when to use this project
## How to use this project
explain flags here
## Core functionality -> mv to docs

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

## Additional functions -> mv to docs  
- A themperature profile and average temperature for the water column can be calculated based on manual temperature measuremnts  e.g. to use for manual temperature correction in the RiverSurveyor Live Software, as an alternative to CTD-probes. (temperature_plot.py)


- To validate the measuring consistency of the RS-data, close points within a variable distance as well as with a time and space difference get compared and the differences displayed in boxplots. (QC_point_consistency.py)

## Requirements
All CSV and shp files must be stored individually in separate folders, i.e. without other files of the same type.

**required python packages:** (as provided by requirements .txt)
- pandas
- geopandas
- shapely
- numpy
- pymatreader
- scipy
- tqdm
- matplotlib


**Exemplary structure of the data dir:**
```
data
├───gps_data        external gps data as .txt
├───outline         .csv
│   └───others
├───shp_files
├───sonar_data
├───temperature
└───waterlevel
```

## Quickstart

make it work and pls without the docs  
from `src` run like so:
```bash
python main.py --im_a_flag
```
## Overview
main.py flowchart:
![alt text](docs/flowchart_main.png)
