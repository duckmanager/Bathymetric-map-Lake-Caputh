#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import logging

from load_data import (create_dataframe,
                    assign_data_to_dataframe,
                    get_gps_dataframe,
                    create_interpolated_coords)
from multibeam_location import create_multibeam_points
from edge_points import (generate_boundary_points,
                    combine_multibeam_edge)
from survey_adjustments import (adjust_depths,
                    correct_waterlevel)
from automatic_correction import detect_and_remove_faulty_depths
from manual_correction import (interactive_error_correction,
                    filter_validation_points)




def get_args():
    arg_par = argparse.ArgumentParser()
    ######################################
    # OPTIONS
    ######################################
    arg_par.add_argument(
        "--automatic_detection",
        default=True,
        const=False,
        nargs="?",
        type=bool,
        help="SET to DISABLE automatically detecting faulty points.",
    )

    arg_par.add_argument(
        "--filtering_max_distance",
        default=0.5,
        type=float,
        help="Radius of area for automatic filtering.",
    )

    ######################################
    # PATHS
    ######################################
    arg_par.add_argument(
        "--sonar_data_dir",
        "-sdd",
        default=Path().joinpath("data", "sonar_data"),
        type=Path,
        help="path to folder with sonar data",
    )

    arg_par.add_argument(
        "--gps_data_dir",
        "-gpsd",
        default=Path().joinpath("data", "gps_data"),
        type=Path,
        help="path to folder with the external gps data",
    )
    # TODO: add ->
    # flag for automatic correction
    # automatic correction radius
    # automatic correction threshold
    # optional reference date
    # flag for manual correction
    # path to water level correction file
    # path to waterbody shape file
    # path to outline measurements file
    # path to dir with sonar data
    # path to dir with gps data
    # path to store faulty_points
    # sample rate - in validation points

    return arg_par.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s : %(asctime)s : %(message)s", level=logging.INFO
    )

    # logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    # logging.info(f"path: {args.some_file} with type {type(args.some_file)}. exists: {args.some_file.is_file()}")

    data_dir = Path("data")
    faulty_points_dir= Path("output/faulty_points")
    logging.info("starting to create empty dataframe")
    sum_dataframe_empty, sum_header = create_dataframe(args.sonar_data_dir)
    logging.info("assigning data to dataframe and correcting sonar-GPS times")
    sum_dataframe = assign_data_to_dataframe(
        args.sonar_data_dir, sum_dataframe_empty, sum_header
    )
    logging.info("starting get_gps")
    gps_geodf_projected = get_gps_dataframe(args.gps_data_dir)  # including interpolation
    logging.info("creating interpolated points")
    interpolated_sum, used_gps_gdf = create_interpolated_coords(
        sum_dataframe, gps_geodf_projected
    )

    logging.info("create multibeam points")
    multipoint_gdf = create_multibeam_points(interpolated_sum)

    logging.info("creating lake outlines")
    boundary_gdf = generate_boundary_points(data_dir)

    logging.info("merging boundary points and measurment points")
    gdf_combined = combine_multibeam_edge(multipoint_gdf, boundary_gdf)

    logging.info("adjusting depths")
    adjusted_gdf = adjust_depths(gdf_combined)

    gdf_waterlevel_corrected = correct_waterlevel(adjusted_gdf, data_dir)

    logging.info("detecting and removing faulty depths")
    filtered_data, faulty_data = detect_and_remove_faulty_depths(
        gdf_waterlevel_corrected,
        faulty_points_dir,
        max_distance=args.filtering_max_distance,
        automatic_detection=args.automatic_detection
    )  # Reihenfolge umgekehrt

    logging.info("detecting and removing faulty depths")
    auto_filtered_data, faulty_data = detect_and_remove_faulty_depths(
        gdf_waterlevel_corrected,
        faulty_points_dir,
        max_distance=args.filtering_max_distance,
        automatic_detection=args.automatic_detection
    )     

    logging.info("starting manual error detection")
    filtered_data = interactive_error_correction(faulty_points_dir, gdf_waterlevel_corrected, manual_overwrite= True )# change to argparse.


    logging.info("Removing points for validation")
    gdf_com, gdf_validation_points = filter_validation_points(filtered_data)
    # -logging.info("reducing data")
    # -selected_sum_data, selected_faulty_sum_data = reduce_data(filtered_data, faulty_data)
    logging.info("saving output")
    output_path = Path("output/multibeam")
    filtered_data.to_csv(output_path / "filtered_data.csv", index=False)
    gdf_waterlevel_corrected.to_csv(output_path / "unfiltered_data.csv", index=False)
    filtered_data.to_csv(output_path / "m_newutc_filtered_newedge.csv", index=False)
    if (
        not faulty_data.empty
    ):  # only save fautly data, when detect_and_remove_faulty_depths did run
        faulty_data.to_csv(
            output_path / "interactive_error" / "interactive_error_points.csv",
            index=False,
        )

    gdf_com.to_csv(output_path / "multibeam_filtered_for_validation.csv", index=False)
    gdf_validation_points.to_csv(
        output_path / "multibeam_validation_points.csv", index=False
    )

    output_QC_path = Path("output/multibeam/QC")
    used_gps_gdf.to_csv(output_QC_path / "used_GPS_points.csv", index=False)

    # -filtered_data.to_csv(output_path / "sum_int_collection_filtered_cleandup.csv", index=False)
    # -  faulty_data.to_csv(output_path / "sum_multibeam_error.csv", index=False)
    # -selected_faulty_sum_data.to_csv(output_path / "sum_int_errors_selected.csv", index=False)
    # --gdf_complete.to_csv(output_path / "depth_and_average_sum_int_filtered_outline.csv", index=False)
    # --faulty_data.to_csv(output_path / "error_depth_and_average_sum_int_filtered.csv", index=False)
    # adjusted_gdf.to_csv(output_path / "multibeam_nozeros.csv", index=False)

    # output data as shp-file
    # filtered_data.to_file(output_path / "sum_int.shp", driver='ESRI Shapefile')
    # selected_faulty_sum_data.to_file(output_path / "sum_int_error.shp", driver='ESRI Shapefile')

    input("we're all done!")
