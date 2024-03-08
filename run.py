"""
File to run the executables for both signal processing and machine learning parts
"""

import argparse
from panoradar_SP import motion_estimation, imaging_result, process_raw_data 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run various scripts for PanoRadar',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("task", metavar="TASK", help="Which task to run")
    parser.add_argument("--SP_root_name", help='Root folder name for signal processing data. REQUIRED for signal processing scripts')
    parser.add_argument("--traj_name", help="Desired imaging trajectory name")
    parser.add_argument("--frame_num", type=int, default=0, help="Desired imaging frame number within the trajectory")
    parser.add_argument("--out_folder", default="SP_out/", help="Output folder for processed data")
    parser.add_argument("--plot_out_path", help="Output path to save the imaging plot")
    
    args = parser.parse_args()

    if (args.task == "motion_estimation"):
        motion_estimation.estimate_whole_dataset(args.SP_root_name)
    elif (args.task == "imaging"):
        imaging_result.imaging_frame_from_traj(args.SP_root_name, args.traj_name, args.frame_num, args.plot_out_path)
    elif (args.task == "process_raw"):
        process_raw_data.process_dataset(args.SP_root_name, args.out_folder)
    