import os
import numpy as np
import cupy as cp
import torch
from torchvision.transforms.functional import resize
import cv2
from tqdm import tqdm

from panoradar_SP.dataset import list_all_trajs_complete, get_all_from_dataset, get_motion_params
import panoradar_SP.radar_imaging_3d as imaging_3d
from panoradar_SP.visualization import get_range_image_from_lidar


def process_data(rf: np.ndarray, lidar: np.ndarray):
    """
    Process rf and lidar data to ready-to-use format by reshaping and resizing.

    Args:
        rf: rf data for 1 frame
        lidar: lidar data for 1 frame

    Returns:
        rf, lidar as np arrays after processing
    """
    azimuth_size = 512

    rf = np.log10(rf + 0.001) / 3
    rf = torch.from_numpy(rf.transpose((1, 0, 2)))  # (#range_bin, #elev, #azimuth)
    rf = resize(rf, (rf.shape[1], azimuth_size))
    rf = rf.numpy().astype(np.float32)

    lidar = torch.from_numpy(lidar.copy()) / 10
    lidar[lidar == 0] = -1e3  # avoid failure points to affect the resizing
    lidar = resize(lidar.unsqueeze(0), (lidar.shape[0], azimuth_size))
    lidar = lidar.numpy().astype(np.float32)

    # use median filter to fix lidar failure regions
    lidar_mf = cv2.medianBlur(lidar[0], ksize=3)[np.newaxis]  # (1, #elev, #azimuth)
    fail_region = lidar < 0
    lidar[fail_region] = lidar_mf[fail_region]

    return rf, lidar


def save_static_data(traj_name: str, root_folder_name: str, out_folder: str):
    """
    Process the raw data of one specific *static* trajectory

    Args:
        traj_name: the name of the trajectory to process
        root_folder_name: the name of the root folder that holds the signal data
        out_folder: the output folder for the processed data

    Saves for one trajectory:
        - rf: (#elev, #range_bin, #azimuth), float32
        - lidar: (#elev, #azimuth), float32
    """
    target_parent_folder = os.path.join(out_folder, traj_name)
    os.makedirs(os.path.join(target_parent_folder, 'lidar_npy'), exist_ok=True)
    os.makedirs(os.path.join(target_parent_folder, 'rf_npy'), exist_ok=True)

    print('loading data...')
    radar_frames, lidar_frames, *_, params = get_all_from_dataset(traj_name, root_folder_name)
    r_radar = params["r_radar"]
    N_syn_ante = params["N_syn_ante"]
    N_rings = params["N_rings"]
    lambda_i = params["lambda_i"]
    lidar_transform = lidar_frames.lidar_transform

    # prepare for 3D imaging
    elevs = np.linspace(np.deg2rad(45), np.deg2rad(-45), 64)
    vbf_compens = [imaging_3d.vertical_beam_forming_compen(elev, lambda_i) for elev in elevs]
    static_refl = np.mean(radar_frames.radar_frames, axis=0, keepdims=True)
    window = imaging_3d.get_window(N_syn_ante, N_rings)

    # Don't need this if there is no redundancy in dataset anymore
    # print('finding static frames...')
    # static_frame_inds = get_static_frame_inds(lidar_frames)
    # print('static inds: ', static_frame_inds)

    print('imaging...')
    for f in tqdm(range(len(radar_frames))):
        # prepare the signal
        radar_frame, beam_angles = radar_frames.get_a_frame_data(f, pad=True)
        radar_frame = cp.array(radar_frame - static_refl, dtype=cp.complex64)
        N_beams = len(beam_angles)

        # rf img
        heatmaps = []
        for elev, vbf_compen in zip(elevs, vbf_compens):
            signal_bf = imaging_3d.vertical_beam_forming(radar_frame, vbf_compen)
            azimuth_compen = imaging_3d.azimuth_compensation(
                r_radar, -beam_angles[1], elev, N_syn_ante, lambda_i, window
            )
            heatmap = imaging_3d.static_imaging(signal_bf, azimuth_compen, N_beams)
            heatmaps.append(cp.asnumpy(heatmap))
        heatmaps = np.stack(heatmaps)

        # lidar
        lidar_frame = lidar_frames.get_a_frame_data(f)
        lidar_range_img = get_range_image_from_lidar(lidar_frame, lidar_transform)

        # further process
        heatmaps, lidar_range_img = process_data(heatmaps, lidar_range_img)

        # save
        np.save(os.path.join(target_parent_folder, f'lidar_npy/{f:05d}.npy'), lidar_range_img)
        np.save(os.path.join(target_parent_folder, f'rf_npy/{f:05d}.npy'), heatmaps)

    print()


def save_moving_data(traj_name: str, root_folder_name: str, out_folder: str):
    """
    Process the raw data of one specific *moving* trajectory

    Args:
        traj_name: the name of the trajectory to process
        root_folder_name: the name of the root folder that holds the signal data
        out_folder: the output folder for the processed data

    Saves for one trajectory:
        - rf: (#elev, #range_bin, #azimuth), float32
        - lidar: (#elev, #azimuth), float32
    """
    target_parent_folder = os.path.join(out_folder, traj_name)
    os.makedirs(os.path.join(target_parent_folder, 'lidar_npy'), exist_ok=True)
    os.makedirs(os.path.join(target_parent_folder, 'rf_npy'), exist_ok=True)

    print('loading data...')
    radar_frames, lidar_frames, _, params = get_all_from_dataset(traj_name, root_folder_name)
    r_radar = params["r_radar"]
    N_syn_ante = params["N_syn_ante"]
    N_rings = params["N_rings"]
    lambda_i = params["lambda_i"]
    lidar_transform = lidar_frames.lidar_transform

    # read motion estimation parameters
    motion = get_motion_params(traj_name, root_folder_name)
    delta_s0_esti = motion['delta_s0_esti']
    theta_v_esti = motion['theta_v_esti']

    # prepare for 3D imaging
    elevs = np.linspace(np.deg2rad(45), np.deg2rad(-45), 64)
    vbf_compens = [imaging_3d.vertical_beam_forming_compen(elev, lambda_i) for elev in elevs]
    static_refl = np.mean(radar_frames.radar_frames, axis=0, keepdims=True)
    window = imaging_3d.get_window(N_syn_ante, N_rings)

    print('imaging...')
    for f in tqdm(range(len(radar_frames))):
        # prepare the signal
        radar_frame, beam_angles = radar_frames.get_a_frame_data(f, pad=True)
        radar_frame = cp.array(radar_frame - static_refl, dtype=cp.complex64)

        # rf img
        heatmaps = []
        for elev, vbf_compen in zip(elevs, vbf_compens):
            signal_bf = imaging_3d.vertical_beam_forming(radar_frame, vbf_compen)
            azimuth_compen = imaging_3d.azimuth_compensation(
                r_radar, -beam_angles[1], elev, N_syn_ante, lambda_i, window
            )
            heatmap = imaging_3d.motion_imaging(
                signal_bf,
                azimuth_compen,
                elev,
                delta_s0_esti[f],
                theta_v_esti[f],
                beam_angles,
                lambda_i,
            )
            heatmaps.append(cp.asnumpy(heatmap))
        heatmaps = np.stack(heatmaps)

        # lidar
        lidar_frame = lidar_frames.get_a_frame_data(f)
        lidar_range_img = get_range_image_from_lidar(lidar_frame, lidar_transform)

        # further process
        heatmaps, lidar_range_img = process_data(heatmaps, lidar_range_img)

        # save
        np.save(os.path.join(target_parent_folder, f'lidar_npy/{f:05d}.npy'), lidar_range_img)
        np.save(os.path.join(target_parent_folder, f'rf_npy/{f:05d}.npy'), heatmaps)

    print()


def process_dataset(root_folder_name: str, out_folder: str):
    """
    Processes the complete signal dataset into format for neural network input

    Args:
        root_folder_name: the name of the root folder that holds the signal data
        out_folder: the output folder for the processed data

    Saves:
        - rf: (#elev, #range_bin, #azimuth), float32
        - lidar: (#elev, #azimuth), float32
    """
    traj_names = list_all_trajs_complete(root_folder_name)
    N_traj = len(traj_names)

    for i, traj_name in enumerate(traj_names):
        print(f'({i+1} / {N_traj}) {traj_name}:')

        # Check for static vs moving trajectory and call respective function
        if 'static' in traj_name:
            save_static_data(traj_name, root_folder_name, out_folder)
        else:
            save_moving_data(traj_name, root_folder_name, out_folder)

    print('done')
