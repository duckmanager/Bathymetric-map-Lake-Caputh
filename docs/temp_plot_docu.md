# Documentation for temperature_plot script
A temperature profile and average temperature for the water column can be calculated based on manual temperature measurements e.g. to use for manual temperature correction in the RiverSurveyor Live Software, as an alternative to CTD-probes. (temperature_plot.py)


Process and show temperature- depth profiles form manual measurements to decide about manual temperature correction of echo sounder data

`temperature_plot.py`is separate from `main.py`
## Options
To use temperature measurements of different depths, the temperature gets interpolated for different steps.   
To change the interpolation steps [in m] (Default 0.1m), use: 
```
--interpolation_steps X.X
```

---
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
     - Mean temperature across the entire measured depth
     - Mean temperature for the top 4 meters

4. Plotting:
   - Plots each individual temperature profile as a line
   - Adds the mean temperature curve as a thick black semi-transparent line
   - Displays average temperatures (total and 0–4 m) as a labeled text box in the top-left corner
   - Y-axis is inverted to display depth increasing downwards
   - X-axis limits are dynamically set based on min/max temperature values
   - Includes grid, axis labels, title, and legend (with average curve labeled only as "Ø Profile",
     no temperature shown in legend)

5. Export:
   - Saves the figure as a high-resolution .png file in the same directory
   - Clears the figure before processing the next file
