# Data needed for full functionality#
All CSV and shp files must be stored individually in separate folders, i.e. without other files of the same type.

### Temperature Stratification
- **File type**: `.csv`
- **function**: `temperature_plot`
- **Structure**:
  - One column named `Depth`
  - Sequentially numbered columns for each measurement location on a given day, containing temperatures at corresponding depths
  - One file per measurement day and function run
  - Filename not relevant
---
### GNSS Data
- **File type**: `.txt`
- **function**: `get_gps_dataframe`
- **Structure**:
  - 1 Hz `Bestposa` and `GPZDA` records
  - All files should be placed in one folder
  - Filename is not relevant
---
### Echo Sounder Data
- **File types**: `.mat` and `.sum` (Export by matlab and ASCII-format)
- **function**: `create_dataframe & assign_data_to_dataframe`
- **Structure**:
  - Exported data from RSL software (MATLAB and ASCII format)
  - All files should be placed in one folder
  - Filenames of a single recording must be identical, otherwise filenames are not relevant
---
### Lake Outline
- **File type**: `.shp`
- **function**: `generate_boundary_points`
- **Structure**:
  - Shapefile of the waterbody outline
  - Filename is not relevant
---
### Shoreline Measurements
- **File type**: `.csv`
- **function**: `generate_boundary_points`
- **Structure**:
  - Measurement points from manual shoreline recordings — all recorded on the same day or adjusted accordingly
  - Filename is not relevant
  - For each measurement point:
    - `Longitude`, `Latitude`, `Depth (m)`, `Date` (Format: MM/DD/YYYY)
---
### Water Level Measurements
- **File type**: `.csv`
- **function**: `correct_waterlevel`
- **Structure**:
  - File must contain water level data for at least the first and last measurement day
  - Filename is not relevant
  - For each measurement day:
    - `date` (Format: DD/MM/YYYY), `waterlevel` (in m)