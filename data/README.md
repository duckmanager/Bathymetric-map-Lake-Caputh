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