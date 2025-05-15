#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Skript to transform River Surveyor M9 data and connect the datapoints to externally collected GPS points
"""
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
    
    ####### automatic point filtering
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
        default=5,
        type=float,
        help="Radius of area for automatic filtering. (in meters)",
    )

    arg_par.add_argument(
        "--filtering_threshold",
        default=0.5,
        type=float,
        help="Threshold of discarding-difference between points to neighbor-points-mean in automatic filtering. (in meters)",
    )
    
    ####### manual point filtering

    arg_par.add_argument(
        "--manual_correction_overwrite",
        action="store_false",
        help="Set this flag to disable triggering manual point inspection, when marked points already exist (default: manual_overwirte= True)."
    )

    ####### waterlevel correction

    arg_par.add_argument(
        "--level_reference_date",
        default="",
        type=str,
        help="Reference day for water level correction in format: m/d/y. (optional)",
    )

    ####### edge points

    arg_par.add_argument(
        "--edge_points_zero",
        action="store_true",
        help="Set this flag to assign 0m depth along the entire lakeside (no manual measurements used)",
    )

    ####### creating validation dataset
   
    arg_par.add_argument(
        "--skip_validation_sampling",
        action="store_false",
        help="Set this flag to disable sampling of validation data."
    )

    arg_par.add_argument(
        "--validation_sample_rate",
        default=3,
        type=int,
        help="Every n-th point gets added to the validation dataset.",
    )

    ######################################
    # PATHS
    ######################################
    
    # Note: Each folder should contain only one of the specific filetypes. 
    ####### input data paths

    arg_par.add_argument(
        "--sonar_data_dir",
        "-sdd",
        default=Path().joinpath("data", "sonar_data"),
        type=Path,
        help="Path to folder with sonar data.",
    )

    arg_par.add_argument(
        "--gps_data_dir",
        "-gpsd",
        default=Path().joinpath("data", "gps_data"),
        type=Path,
        help="Path to folder with the external gps data.",
    )

    arg_par.add_argument(
        "--water_level_dir",
        "-wld",
        default=Path().joinpath("data", "waterlevel"),
        type=Path,
        help="Path to folder with waterlevel csv.",
    )

    arg_par.add_argument(
        "--lake_shp_dir",
        "-lsd",
        default=Path().joinpath("data", "shp_files"),
        type=Path,
        help="Path to folder with shp-file of the waterbody",
    )

    arg_par.add_argument(
        "--point_data_dir",
        "-pdd",
        default=Path().joinpath("data", "outline"),
        type=Path,
        help="Path to folder with csv of edge measurments.",
    )
    
    ####### output data paths
    
    arg_par.add_argument(
        "--finished_dataset_dir",
        "-fdd",
        default=Path().joinpath("output", "processed_data"),
        type=Path,
        help="Path to folder to store csv of finished dataset.",
    )

    arg_par.add_argument(
        "--validation_dataset_dir",
        "-vdd",
        default=Path().joinpath("output", "validation_data"),
        type=Path,
        help="Path to folder to store csv's of validation datasets.",
    )

    arg_par.add_argument(
        "--QC_dataset_dir",
        "-qcd",
        default=Path().joinpath("output", "QC"),
        type=Path,
        help="Path to folder to store csv's for later quality assesment.",
    )

    arg_par.add_argument(
        "--faulty_points_dir",
        "-fpd",
        default=Path().joinpath("output", "faulty_points"),
        type=Path,
        help="Path to folder to store csv with marked erroneous sample points.",
    )

    return arg_par.parse_args()
    
    ######################################
    ######################################


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s : %(asctime)s : %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)

    # logging.getLogger().setLevel(logging.INFO)
    args = get_args()

    ######################################
    # functions
    ######################################

    logging.info("starting to create empty dataframe")
    sum_dataframe_empty, sum_header = create_dataframe(args.sonar_data_dir)

    logging.info("assigning data to dataframe and correcting sonar-GPS times")
    sum_dataframe = assign_data_to_dataframe(
        args.sonar_data_dir, sum_dataframe_empty, sum_header)
    
    logging.info("reading external-GPS data")
    gps_geodf_projected = get_gps_dataframe(args.gps_data_dir)  # including interpolation

    logging.info("creating interpolated points")
    interpolated_sum, used_gps_gdf = create_interpolated_coords(
        sum_dataframe, gps_geodf_projected)

    logging.info("creating multibeam points")
    multipoint_gdf = create_multibeam_points(interpolated_sum)

    logging.info("creating lake outlines")
    boundary_gdf = generate_boundary_points(args.lake_shp_dir, args.point_data_dir, args.edge_points_zero)

    logging.info("merging boundary points and measurment points")
    gdf_combined = combine_multibeam_edge(multipoint_gdf, boundary_gdf)

    logging.info("adjusting depths")
    adjusted_gdf = adjust_depths(gdf_combined)

    logging.info("adjusting waterlevels")
    gdf_waterlevel_corrected = correct_waterlevel(adjusted_gdf, args.water_level_dir, reference_day="")

    logging.info("Applying automated point filtering")
    filtered_data, faulty_data = detect_and_remove_faulty_depths(
        gdf_waterlevel_corrected,
        faulty_points_dir= args.faulty_points_dir,
        max_distance=args.filtering_max_distance,
        threshold=args.filtering_threshold,
        automatic_detection=args.automatic_detection)  

    logging.info("Starting manual point filtering")
    filtered_data = interactive_error_correction(
        args.faulty_points_dir,
        gdf_waterlevel_corrected, 
        args.manual_correction_overwrite)


    logging.info("Removing points for validation")
    gdf_com, gdf_validation_points = filter_validation_points(
        filtered_data,
        sample_rate=args.validation_sample_rate,
        create_validation_data=args.skip_validation_sampling)



    logging.info("saving output")

    filtered_data.to_csv(args.finished_dataset_dir / "filtered_data.csv", index=False)
    gdf_waterlevel_corrected.to_csv(args.finished_dataset_dir / "unfiltered_data.csv", index=False)

    gdf_com.to_csv(args.validation_dataset_dir / "filtered_for_validation.csv", index=False)
    gdf_validation_points.to_csv(args.validation_dataset_dir / "validation_points.csv", index=False)

    used_gps_gdf.to_csv(args.QC_dataset_dir / "used_GPS_points.csv", index=False)

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
