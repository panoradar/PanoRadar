"""Visualization and interpretation of the imaging result."""

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_ani

from typing import List, Tuple, Callable


def enhance_imaging_reuslt(
    ori_heatmap: np.ndarray, mean_fac: float = 5, steep_fac: float = 70
) -> np.ndarray:
    """Enhance the imaging result with Sigmoid function
       Sigmoid (x) = 1/ (1+exp{-steep_fac * (x-mean*mean_fac)})
    Args:
        ori_heatmap: original imaging result from SAR
        mean_fac: mean factor
        steep_fac: steep factor
    Return:
        enh_heatmap: the enhanced one.
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    # normalize through bin, then use a sigmoid function to enhance it
    heatmaps_n = ori_heatmap / np.max(ori_heatmap, axis=0, keepdims=True)
    mean = np.mean(heatmaps_n, axis=0, keepdims=True)
    enh_heatmap = sigmoid(steep_fac * (heatmaps_n - mean * mean_fac))

    return enh_heatmap


def project_polar_to_cartesian_3d(
    heatmap: np.ndarray,
    elev: float,
    r_rings: np.ndarray,
    max_range: float,
    grid_size: float,
    beam_angles: np.ndarray,
    default_value: float = 0.0,
) -> np.ndarray:
    """Project the polar system imaging result to Cartesian system.
    This function is for 3D projection into different elevation slice.
    Args:
        heatmap: the polar system imaging result, (N_rings, N_beams)
        elev: the elevation angle, up positive down negative
        r_rings: radius of each ring, in meter, shape (N_rings, )
        max_range: maximum projection range, m x m image
        grid_size: the actual size of each grid/pixel
        beam_angles: the angle of each beam.
    Return:
        proj_heatmap: the Cartesian system imaging result
    """
    PROJ_MAP_SZ = int(2 * max_range / grid_size)  # size of the projected heatmap
    proj_heatmap = np.full((PROJ_MAP_SZ, PROJ_MAP_SZ), default_value, dtype=np.float32)
    N_rings, N_beams = heatmap.shape

    cos_a = np.cos(elev)
    cos_phi = np.cos(beam_angles)
    sin_phi = np.sin(beam_angles)

    # project polar to Cartesian
    for ring_id in range(0, N_rings):
        x_grid_id = (r_rings[ring_id] * cos_a * cos_phi + max_range) / grid_size
        y_grid_id = (r_rings[ring_id] * cos_a * sin_phi + max_range) / grid_size
        x_grid_id = np.round(x_grid_id).astype(np.int32)
        y_grid_id = np.round(y_grid_id).astype(np.int32)

        # bound to PROJ_MAP_SZ
        valid = np.logical_and(
            np.logical_and(x_grid_id >= 0, x_grid_id < PROJ_MAP_SZ),
            np.logical_and(y_grid_id >= 0, y_grid_id < PROJ_MAP_SZ),
        )

        proj_heatmap[y_grid_id[valid], x_grid_id[valid]] = heatmap[ring_id][valid]

    return proj_heatmap


def project_polar_to_cartesian(
    heatmap: np.ndarray,
    r_rings: np.ndarray,
    max_range: float,
    grid_size: float,
    beam_angles: np.ndarray,
    default_value: float = 0.0,
    rotation_offset: float = 0,
) -> np.ndarray:
    """Project the polar system imaging result to Cartesian system.
    Args:
        heatmap: the polar system imaging result, (N_rings, N_beams)
        r_rings: radius of each ring, in meter, shape (N_rings, )
        max_range: maximum projection range, m x m image
        grid_size: the actual size of each grid/pixel
        beam_angles: the facing angle of each beam
    Return:
        proj_heatmap: the Cartesian system imaging result
    """
    PROJ_MAP_SZ = int(2 * max_range / grid_size)  # size of the projected heatmap
    proj_heatmap = np.full((PROJ_MAP_SZ, PROJ_MAP_SZ), default_value, dtype=np.float32)
    N_rings, N_beams = heatmap.shape

    cos_phi = np.cos(beam_angles + rotation_offset)
    sin_phi = np.sin(beam_angles + rotation_offset)

    # project polar to Cartesian
    for ring_id in range(0, N_rings):
        x_grid_id = (r_rings[ring_id] * cos_phi + max_range) / grid_size
        y_grid_id = (r_rings[ring_id] * sin_phi + max_range) / grid_size
        x_grid_id = np.round(x_grid_id).astype(np.int32)
        y_grid_id = np.round(y_grid_id).astype(np.int32)

        # bound to PROJ_MAP_SZ
        valid = np.logical_and(
            np.logical_and(x_grid_id >= 0, x_grid_id < PROJ_MAP_SZ),
            np.logical_and(y_grid_id >= 0, y_grid_id < PROJ_MAP_SZ),
        )

        proj_heatmap[y_grid_id[valid], x_grid_id[valid]] = heatmap[ring_id][valid]

    return proj_heatmap


def show_2d_imaging_plane(
    heatmap_abs,
    lidar_frame,
    r_rings,
    beam_angles: np.ndarray,
    save_path: str
):
    """This is the function showing the large result figure"""
    # >>>>>>>>>>>>> Visualize the Imaging Result <<<<<<<<<<<<<<<<<<<<<
    # There are four plots:
    #   1. Aligned with Lidar;   2. Log10 (dB)

    MAX_RANGE = 10.0  # max imaging 10m*10m
    GRID_SIZE = 0.04  # grid size 4cm

    # ################# 1. Log  #################
    heatmap_n = np.log10(heatmap_abs + 0.001) / 3
    proj_heatmap_l = project_polar_to_cartesian(
        heatmap_n, r_rings, MAX_RANGE, GRID_SIZE, beam_angles, -0.2
    )
    snr_l = compute_snr(proj_heatmap_l, lidar_frame, MAX_RANGE, GRID_SIZE)

    # #################  visualize result  #################
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(lidar_frame[:, 0], lidar_frame[:, 1], s=1, c="C1")
    plt.imshow(
        proj_heatmap_l, origin="lower", extent=(-10, 10, -10, 10), cmap="jet", vmin=-0.2, vmax=0.5
    )
    plt.subplot(1, 2, 2)
    plt.imshow(
        proj_heatmap_l, origin="lower", extent=(-10, 10, -10, 10), cmap="jet", vmin=-0.2, vmax=0.5
    )
    plt.title(f"Log scale in dB, SNR:{snr_l:.2f} dB")

    if save_path:
        plt.savefig(f'{save_path}/imaging_result.png')
    else:
        plt.show()


def show_3d_imaging_plane(
    heatmaps: List[np.ndarray],
    lidar_frames: List[np.ndarray],
    elevs: np.ndarray,
    r_rings: np.ndarray,
    beam_angles: np.ndarray,
):
    """Show the 3D imaging results by each elevation angle slice by slice."""
    MAX_RANGE = 10.0  # max imaging 10m*10m
    GRID_SIZE = 0.04  # grid size 4cm

    num_row = np.ceil(len(heatmaps) / 3).astype(int)
    plt.figure(figsize=(24, 4 * num_row))

    i = 1
    for heatmap, lidar_frame, elev in zip(heatmaps, lidar_frames, elevs):
        # sigmoid enhanced
        heatmap_n = enhance_imaging_reuslt(heatmap, mean_fac=5, steep_fac=70)
        proj_heatmap_s = project_polar_to_cartesian_3d(
            heatmap_n, elev, r_rings, MAX_RANGE, GRID_SIZE, beam_angles
        )

        # log10, dB
        heatmap_n = np.log10(heatmap + 0.001) / 3
        proj_heatmap_l = project_polar_to_cartesian_3d(
            heatmap_n, elev, r_rings, MAX_RANGE, GRID_SIZE, beam_angles, -0.2
        )

        plt.subplot(num_row, 6, i)
        plt.scatter(lidar_frame[:, 0], lidar_frame[:, 1], s=1, c="C1")
        plt.imshow(proj_heatmap_s, origin="lower", extent=(-10, 10, -10, 10), cmap="jet")
        plt.title(f"elevation:{np.rad2deg(elev):.1f} deg")
        plt.subplot(num_row, 6, i + 1)
        plt.imshow(
            proj_heatmap_l,
            origin="lower",
            extent=(-10, 10, -10, 10),
            cmap="jet",
            vmin=-0.2,
            vmax=0.5,
        )
        plt.title("log scale in dB")

        i += 2


def heatmaps_to_pointcloud(
    heatmaps: List[np.ndarray],
    elevs: np.ndarray,
    r_rings: np.ndarray,
    threshold: float,
    beam_angles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert multiple heatmaps to point clouds using threshold.
    Args:
        heatmaps: multiple heatmap of different heights
        elevs: the elevation angle, in rad. The horizontal plane is zero,
            upper part is positive and lower part is negative.
        r_rings: radius of different rings
        threshold: the threshold to determine points
        beam_angles: the angle of each beam
    Returns:
        xyz: xyz point cloud location, shape (N, 3)
        reflection: the reflection strength, shape (N, )
    """
    xyz = []
    reflection = []

    # project polar to Cartesian
    N_rings, N_beams = heatmaps[0].shape
    cos_phi = np.cos(beam_angles)
    sin_phi = np.sin(beam_angles)
    #
    for heatmap, elev in zip(heatmaps, elevs):
        cos_a = np.cos(elev)
        sin_a = np.sin(elev)
        heatmap = enhance_imaging_reuslt(heatmap, mean_fac=5, steep_fac=70)
        for ring_id in range(N_rings):
            x = r_rings[ring_id] * cos_a * cos_phi
            y = r_rings[ring_id] * cos_a * sin_phi
            z = r_rings[ring_id] * sin_a * np.ones((1, N_beams))
            reflection.append(heatmap[ring_id])
            xyz.append(np.vstack((x, y, z)).T)

    xyz = np.vstack(xyz)
    reflection = np.hstack(reflection)
    indices = reflection > threshold

    return xyz[indices], reflection[indices]


def save_point_cloud(name: str, xyz_points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    o3d.io.write_point_cloud(f"{name}", pcd)


def compute_snr(
    proj_heatmap: np.ndarray,
    lidar_frame: np.ndarray,
    max_range: float,
    grid_size: float,
) -> float:
    """Compute the Signal to Noise Ration (SNR).
    Args:
        proj_heatmap: projected heatmap, in Catesian
        lidar_frame: (x,y) points, shape (N,2)
        max_range: the maximum range of the proj_heatmap, (-max_range, max_range)
        grid_size: the size of each grid
    Returns:
        snr: the Signal to Noise Ration (SNR) in decibel (dB)
    """
    MAX_IND = proj_heatmap.shape[1]
    x_ind = ((lidar_frame[:, 0] + max_range) / grid_size).astype(np.int32)
    y_ind = ((lidar_frame[:, 1] + max_range) / grid_size).astype(np.int32)

    # dilate the indice to consider more points as signal
    DW = 3  # dilate width
    x_ind = np.hstack([x_ind + i for i in range(-DW, DW + 1) for j in range(-DW, DW + 1)])
    y_ind = np.hstack([y_ind + j for i in range(-DW, DW + 1) for j in range(-DW, DW + 1)])

    # make sure indices are valid
    x_ind[x_ind < 0] = 0
    y_ind[y_ind < 0] = 0
    x_ind[x_ind >= MAX_IND] = MAX_IND - 1
    y_ind[y_ind >= MAX_IND] = MAX_IND - 1
    xy_ind = np.unique(np.vstack((x_ind, y_ind)), axis=1)  # shape (2, N)

    proj_heatmap = proj_heatmap - np.min(proj_heatmap)  # shift to starting from zero
    signal_count = xy_ind.shape[1]  # how many signal points

    # extact signal and noise
    signal = np.mean(proj_heatmap[xy_ind[1], xy_ind[0]])  # note x,y => col,row
    proj_heatmap = proj_heatmap.copy()
    proj_heatmap[xy_ind[1], xy_ind[0]] = 0
    proj_heatmap = np.sort(np.ravel(proj_heatmap))
    empty_count = int((1 - np.pi / 4) * MAX_IND**2)  # how many empty points
    noise = np.median(proj_heatmap[signal_count + empty_count :])

    snr = np.log10(signal**2 / noise**2) * 10
    return snr


def get_range_image_from_lidar(
    lidar_frame: np.ndarray, lidar_transform: np.ndarray, out_azi_size: int = None
) -> np.ndarray:
    """Get the range image from lidar point cloud.
    Args:
        lidar_frame: a single frame of point cloud, shape (channel, horizontal, xyz)
        out_azi_size: the output azimuth size
    Returns:
        range_img: each pixel is the range for the point. shape (elev, azimuth)
    """
    transform_3d = np.eye(3)
    transform_3d[:2, :2] = lidar_transform
    transform_3d = transform_3d[None]
    lidar_frame = lidar_frame @ transform_3d

    point_ranges = np.linalg.norm(lidar_frame, ord=2, axis=-1)
    point_azimuth = np.arctan2(lidar_frame[:, :, 1], lidar_frame[:, :, 0])
    point_azimuth[point_azimuth < 0] += 2 * np.pi  # shift -pi~pi to 0~2pi

    if out_azi_size is None:
        out_azi_size = lidar_frame.shape[1]
    out_elev_size = lidar_frame.shape[0]

    azimuth_min = 0
    azimuth_num = out_azi_size
    azimuth_bin = 2 * np.pi / azimuth_num
    azimuth_inds = np.round((point_azimuth - azimuth_min) / azimuth_bin).astype(np.int32)
    valid = np.logical_and(azimuth_inds >= 0, azimuth_inds < azimuth_num)

    polar = np.zeros((out_elev_size, out_azi_size), dtype=np.float32)
    for i in range(64):
        valid_i = valid[i]
        polar[i, azimuth_inds[i][valid_i]] = point_ranges[i][valid_i]

    polar = np.flip(polar, axis=1)  # from counter-clockwise to clockwise, align with radar
    return polar


def get_range_image_from_radar(
    heatmaps: List[np.ndarray], r_rings: np.ndarray, method='peaks'
) -> np.ndarray:

    heatmaps = np.stack(heatmaps)
    range_img_weighted_peaks = np.zeros((heatmaps.shape[0], heatmaps.shape[2]))

    if method == 'peaks':
        import scipy.signal

        for i in range(heatmaps.shape[0]):
            for j in range(heatmaps.shape[2]):
                peaks, _ = scipy.signal.find_peaks(
                    heatmaps[i, :, j], height=heatmaps[i, :, j].max() * 0.2, distance=10
                )
                # peaks = peaks[:10]
                weights = heatmaps[i, :, j][peaks]
                res = (r_rings[peaks] * weights).sum() / (weights.sum() + 1e-5)
                range_img_weighted_peaks[i, j] = res
    elif method == 'argsort':
        top_inds_all = np.argsort(heatmaps, axis=1)[:, -30:, :]
        for i in range(top_inds_all.shape[0]):
            for j in range(top_inds_all.shape[2]):
                top5_inds = top_inds_all[i, :, j]
                weights = heatmaps[i, :, j][top5_inds]
                res = (r_rings[top5_inds] * weights).sum() / (weights.sum() + 1e-5)
                range_img_weighted_peaks[i, j] = res
    else:
        raise NotImplementedError()

    return range_img_weighted_peaks


def get_panorama_image(
    video_frames: List[np.ndarray],
    res: Tuple[int, int],
    mid_offset: int,
    camera_orient="HOR",
) -> np.ndarray:
    """Get panorama image from video frames.
    Args:
        video_frames: A list video frames from beginning to end
        res: final image resolution, (width, height)
        mid_offset: mid_index offset, - left/up, + right/down
        camera_orient: the orientation of the camera, can be {"HOR", "VER"} for
            horizontal and vertical
    Returns:
        panorama: the desired panorama image
    """
    # determine orientation
    if camera_orient == "HOR":
        mid_ind = video_frames[0].shape[1] // 4 + mid_offset
        panorama = [frame[:, mid_ind].reshape(-1, 1) for frame in video_frames]
    else:
        mid_ind = video_frames[0].shape[0] // 2 + mid_offset
        span = video_frames[0].shape[1] // 2
        panorama = [frame[mid_ind, :span].reshape(-1, 1) for frame in video_frames]

    # resize
    panorama = cv2.resize(np.hstack(panorama), res)
    return panorama
