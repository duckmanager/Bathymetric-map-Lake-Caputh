from pathlib import Path
import pandas as pd
from pymatreader import read_mat
from datetime import datetime, timedelta
from tqdm import tqdm
import json 

# add argparse
def get_vel_mat(data_dir: Path):


    data = {}
    for vel_file in tqdm(data_dir.glob("*.vel")):
        vel_data_raw = vel_file.read_text().splitlines()
        vel_data = vel_data_raw[1:]
        vel_header = vel_data_raw[0]

        mat_data_Utc = read_mat(str(snr_file)[:-4] + ".mat")["GPS"]["Utc"]

        data_id = snr_file.name[:-4]
        data[data_id] = ["Utc,lat,long" + snr_header]
        for snr, utc in zip(snr_data, mat_data_Utc):
            # TODO: save the snr data we need
            data[data_id].append([utc, None, None, *snr.split(",")])
    return data

def get_gps(data_dir: Path):
    """
    find and read all gps files. parse them and build a dict.
    data = {
        "date": {
            "time": (lat, long)
        }
    }
    """
    data = {}
    bestposa = None
    for gps_file in tqdm(data_dir.glob("*.txt")):
        date = None
        gps_data = gps_file.read_text().splitlines()
        for line in tqdm(gps_data):
            if line.startswith("#BESTPOSA"):
                # 0        ,1   ,2,   ,3   ,4   ,5         ,6,7,8              ,9   ,10 lat    ,11 long
                # #BESTPOSA,COM2,0,0.0,FINE,2349,479940.000,,,;INSUFFICIENT_OBS,NONE,0.00000000,0.00000000,0.000,17.230,WGS84,0.0000,0.0000,0.0000,"",0.0,0.0,31,0,,,,,,*9cefb0ad
                bestposa_raw = line.split(",")
                if bestposa_raw[10].replace(".","").isdigit():
                    bestposa = (bestposa_raw[10], bestposa_raw[11])
                else:
                    bestposa = (bestposa_raw[11], bestposa_raw[12])

            elif line.startswith("$GPZDA"):
                # $GPZDA,131842.00,17,1,2025,,*59
                _, time, day, month, year, *_ = line.split(",")
                if date is None:
                    date = (int(day), int(month), int(year))
                    data[date] = {}

                data[date][int(float(time))] = bestposa

    return data