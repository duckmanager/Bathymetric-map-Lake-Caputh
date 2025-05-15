#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""
This script processes temperature-depth profiles from temperature-depth csv,
interpolates the data at a fixed depth resolution, and plots all
individual profiles along with their mean temperature curve.

Two average temperature values are calculated and displayed in the plot:
- the overall mean temperature across the full depth range,
- the mean temperature for the upper 4 meters.

Each plot is saved as a PNG image in the output folder.
"""

def get_args():
    arg_par = argparse.ArgumentParser()

#############
# Paths
#############
    arg_par.add_argument(
        "--temperature_data_dir",
        "-dd",
        default=Path().joinpath("output", "temperature_plot"),
        type=Path,
        help="Path to folder with csv of temperature measurments.",
    )

    arg_par.add_argument(
        "--output_data_dir",
        "-odd",
        default=Path().joinpath("output", "temperature_plot"),
        type=Path,
        help="Path to folder to store the temperature plots.",
    )

###########
# Options
###########
    arg_par.add_argument(
        "--interpolation_steps",
        default=0.1,
        type=float,
        help="Interpolation steps to use measurment of differnt depths for one mean curve.",
    )

    return arg_par.parse_args()
args = get_args()



data_folder = args.temperature_data_dir  # path to csv-data
output_folder = args.output_data_dir # path to save the results
depth_step: float=0.1 # interpoaltion steps

# read data, iterate through all csv files
for csv_file in data_folder.glob("*.csv"):
    df = pd.read_csv(csv_file, sep=";")
    if "Depth" not in df.columns:
        print("Depth column not found")
        continue

    # prepare interpoaltion raster
    depths = df["Depth"]
    temp_cols = df.columns.drop("Depth")
    max_depth = depths.max()
    interp_depths = np.arange(0, max_depth + depth_step, depth_step)

    # iterate through each measurment-series
    profiles = {}
    for col in temp_cols:
        valid = df[col].notna()
        if valid.sum() < 2:
            continue
        d = depths[valid].values
        t = df[col][valid].values
        interp_t = np.interp(interp_depths, d, t, left=np.nan, right=np.nan)
        interp_t[(interp_depths < d.min()) | (interp_depths > d.max())] = np.nan
        profiles[col] = interp_t

    if not profiles:
        print(f" No temperature profile in {csv_file.name}")
        continue

    # calculate average graph
    arr = np.array(list(profiles.values()))
    mean_profile = np.nanmean(arr, axis=0)

    # calculate average temps for all depths
    avg_total = np.nanmean(mean_profile)
    mask_4 = interp_depths <= min(4, max_depth)
    avg_4m = np.nanmean(mean_profile[mask_4])

    # plot individual-profiles
    for col, vals in profiles.items():
        mask = ~np.isnan(vals)
        plt.plot(vals[mask], interp_depths[mask], alpha=1.0, label=f"Profil {col}")

    # plot average-profile
    mean_mask = ~np.isnan(mean_profile)
    if mean_mask.any():
        plt.plot(mean_profile[mean_mask], interp_depths[mean_mask],
                 "k-", lw=2, alpha=0.5, label="Ø Profil")

    # insert average values
    plt.text(0.05, 0.95,
             f"Ø bis max: {avg_total:.2f}°C\nØ bis 4m: {avg_4m:.2f}°C",
             transform=plt.gca().transAxes, verticalalignment="top",
             bbox=dict(facecolor="white", alpha=0.7))

    # Plot-Layout
    plt.gca().invert_yaxis()
    plt.xlabel("Temperatur (°C)")
    plt.ylabel("Tiefe (m)")
    plt.grid(True)
    all_temps = df[temp_cols].values.flatten()
    plt.xlim(np.nanmin(all_temps) - 1, np.nanmax(all_temps) + 1)
    plt.title(f"{csv_file.stem} Temperaturschichtung")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_folder / f"{csv_file.stem}_temperatureplot.png", dpi=300)
    plt.clf()

print("finished")
