import json
import numpy as np

from typing import List, Dict
from pathlib import Path
from functools import partial
from collections import OrderedDict

from detectron2.data import MetadataCatalog, DatasetCatalog

obj_name2id = OrderedDict(
    [
        ('person', 0),
        ('non-person', 1),
    ]
)

seg_classes = [
    'person',
    'chair/table',
    'railing',
    'trashcan',
    'stairs',
    'elevator',
    'door',
    'window',
    'ceiling',
    'wall',
    'floor',
]

seg_colors = [
    (0, 106, 216),
    (202, 180, 34),
    (247, 238, 177),
    (255, 138, 244),
    (213, 219, 113),
    (121, 111, 173),
    (128, 125, 186),
    (188, 189, 220),
    (102, 69, 0),
    (158, 154, 200),
    (255, 113, 151),
]

obj_colors = [[0, 106, 216], [30, 200, 31]]

metadata = {
    'stuff_classes': seg_classes,
    'stuff_colors': seg_colors,
    'ignore_label': 255,
    'thing_classes': list(obj_name2id.keys()),
}

def register_dataset(cfg):
    """Register all the custom datasets that are used.
    Leave one trajectory out:
        loto_train,  loto_test

    Leave one building out
        lobo_train_building9, lobo_train_building5, lobo_train_building8, lobo_train_building4,
        lobo_train_building7, lobo_train_building3, lobo_train_building12, lobo_train_building10,
        lobo_train_building6, lobo_train_building2, lobo_train_building11, lobo_train_building1
        [And also change "train" to "test"]
    """
    base_path = Path(cfg.DATASETS.BASE_PATH)

    # excludes some files in the testing sets
    with open(base_path / Path('excludes.txt'), 'r') as f:
        excludes = [base_path / Path(line.rstrip('\n')) for line in f.readlines()]

    # define trajectories
    loto_train_building9 = list(base_path.glob('building9_[ms]*/exp*-00[!0]'))
    loto_test_building9 = list(base_path.glob('building9_[ms]*/exp*-000'))

    loto_train_building5 = list(base_path.glob('building5_[ms]*/exp*-00[!1]'))
    loto_test_building5 = list(base_path.glob('building5_[ms]*/exp*-001'))

    loto_train_building8 = list(base_path.glob('building8_[ms]*/exp*-00[!1]'))
    loto_test_building8 = list(base_path.glob('building8_[ms]*/exp*-001'))

    loto_train_building4 = list(base_path.glob('building4_[ms]*/exp*-00[!3]'))
    loto_test_building4 = list(base_path.glob('building4_[ms]*/exp*-003'))

    loto_train_building7 = list(base_path.glob('building7_[ms]*/exp*-00[!1]'))
    loto_test_building7 = list(base_path.glob('building7_[ms]*/exp*-001'))

    loto_train_building3 = list(base_path.glob('building3_[ms]*/exp*-00[!1]'))
    loto_test_building3 = list(base_path.glob('building3_[ms]*/exp*-001'))

    loto_train_building12 = list(base_path.glob('building12_[ms]*/exp*-00[!1]'))
    loto_test_building12 = list(base_path.glob('building12_[ms]*/exp*-001'))

    loto_train_building10 = list(base_path.glob('building10_[sm]*/exp*-00[!1]'))
    loto_test_building10 = list(base_path.glob('building10_[sm]*/exp*-001'))
    boost_building10 = list(base_path.glob('building10_extra_moving/exp*'))

    loto_train_building6 = list(base_path.glob('building6_[ms]*/exp*-00[!1]'))
    loto_test_building6 = list(base_path.glob('building6_[ms]*/exp*-001'))
    boost_building6 = list(base_path.glob('building6_extra_moving/exp*'))

    loto_train_building2 = list(base_path.glob('building2_[ms]*/exp*-00[!1]'))
    loto_test_building2 = list(base_path.glob('building2_[ms]*/exp*-001'))
    boost_building2 = list(base_path.glob('building2_extra_moving/exp*'))

    loto_train_building11 = list(base_path.glob('building11_[ms]*/exp*-00[!1]'))
    loto_test_building11 = list(base_path.glob('building11_[ms]*/exp*-001'))
    boost_building11 = list(base_path.glob('building11_extra_moving/exp*'))

    loto_train_building1 = list(base_path.glob('building1_[ms]*/exp*-00[!1]'))
    loto_test_building1 = list(base_path.glob('building1_[ms]*/exp*-001'))
    boost_building1 = list(base_path.glob('building1_extra_moving/exp*'))

    boost_building13 = list(base_path.glob('building13_moving/exp*'))
    boost_building14 = list(base_path.glob('building14_moving/exp*'))
    boost_building15 = list(base_path.glob('building15_moving/exp*'))

    # *********************  LOTO (Leave one trajectory out)  *********************
    # fmt: off
    loto_train_all_trajs = sorted(
        loto_train_building9 + loto_train_building5 + loto_train_building8 + loto_train_building4 + 
        loto_train_building7 + loto_train_building3 + loto_train_building12 + loto_train_building10 + 
        loto_train_building6 + loto_train_building2 + loto_train_building11 + loto_train_building1 +
        boost_building10 + boost_building6 + boost_building2 + boost_building11 + boost_building1 + 
        boost_building13 + boost_building14 + boost_building15
    )
    loto_test_all_trajs = sorted(
        loto_test_building9 + loto_test_building5 + loto_test_building8 + loto_test_building4 +
        loto_test_building7 + loto_test_building3 + loto_test_building12 + loto_test_building10 +
        loto_test_building6 + loto_test_building2 + loto_test_building11 + loto_test_building1
    )
    #
    DatasetCatalog.register('loto_train', partial(get_dataset_dicts, loto_train_all_trajs))
    MetadataCatalog.get('loto_train').set(**metadata)
    DatasetCatalog.register('loto_test', partial(get_dataset_dicts, loto_test_all_trajs))
    MetadataCatalog.get('loto_test').set(
        **metadata, vis_ind=get_vis_indices(loto_test_all_trajs, loto_test_all_trajs)
    )
    # *****************************************************************************


    # **********************  LOBO (Leave one building out)  **********************
    all_trajs = loto_train_all_trajs + loto_test_all_trajs
    lobo_train_building9 = sorted([t for t in all_trajs if t not in (loto_train_building9 + loto_test_building9)])
    lobo_train_building5 = sorted([t for t in all_trajs if t not in (loto_train_building5 + loto_test_building5)])
    lobo_train_building8 = sorted([t for t in all_trajs if t not in (loto_train_building8 + loto_test_building8)])
    lobo_train_building4 = sorted([t for t in all_trajs if t not in (loto_train_building4 + loto_test_building4)])
    lobo_train_building7 = sorted([t for t in all_trajs if t not in (loto_train_building7 + loto_test_building7)])
    lobo_train_building3 = sorted([t for t in all_trajs if t not in (loto_train_building3 + loto_test_building3)])
    lobo_train_building12 = sorted([t for t in all_trajs if t not in (loto_train_building12 + loto_test_building12)])
    lobo_train_building10 = sorted([t for t in all_trajs if t not in (loto_train_building10 + loto_test_building10 + boost_building10)])
    lobo_train_building6 = sorted([t for t in all_trajs if t not in (loto_train_building6 + loto_test_building6 + boost_building6)])
    lobo_train_building2 = sorted([t for t in all_trajs if t not in (loto_train_building2 + loto_test_building2 + boost_building2)])
    lobo_train_building11 = sorted([t for t in all_trajs if t not in (loto_train_building11 + loto_test_building11 + boost_building11)])
    lobo_train_building1 = sorted([t for t in all_trajs if t not in (loto_train_building1 + loto_test_building1 + boost_building1)])
    # 
    lobo_test_building9 = sorted(loto_train_building9 + loto_test_building9)
    lobo_test_building5 = sorted(loto_train_building5 + loto_test_building5)
    lobo_test_building8 = sorted(loto_train_building8 + loto_test_building8)
    lobo_test_building4 = sorted(loto_train_building4 + loto_test_building4)
    lobo_test_building7 = sorted(loto_train_building7 + loto_test_building7)
    lobo_test_building3 = sorted(loto_train_building3 + loto_test_building3)
    lobo_test_building12 = sorted(loto_train_building12 + loto_test_building12)
    lobo_test_building10 = sorted(loto_train_building10 + loto_test_building10)
    lobo_test_building6 = sorted(loto_train_building6 + loto_test_building6)
    lobo_test_building2 = sorted(loto_train_building2 + loto_test_building2)
    lobo_test_building11 = sorted(loto_train_building11 + loto_test_building11)
    lobo_test_building1 = sorted(loto_train_building1 + loto_test_building1)
    #
    DatasetCatalog.register('lobo_train_building9', partial(get_dataset_dicts, lobo_train_building9))
    MetadataCatalog.get('lobo_train_building9').set(**metadata)
    DatasetCatalog.register('lobo_test_building9', partial(get_dataset_dicts, lobo_test_building9, excludes))
    MetadataCatalog.get('lobo_test_building9').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building9, lobo_test_building9)
    )
    DatasetCatalog.register('lobo_train_building5', partial(get_dataset_dicts, lobo_train_building5))
    MetadataCatalog.get('lobo_train_building5').set(**metadata)
    DatasetCatalog.register('lobo_test_building5', partial(get_dataset_dicts, lobo_test_building5, excludes))
    MetadataCatalog.get('lobo_test_building5').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building5, lobo_test_building5)
    )
    DatasetCatalog.register('lobo_train_building8', partial(get_dataset_dicts, lobo_train_building8))
    MetadataCatalog.get('lobo_train_building8').set(**metadata)
    DatasetCatalog.register('lobo_test_building8', partial(get_dataset_dicts, lobo_test_building8, excludes))
    MetadataCatalog.get('lobo_test_building8').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building8, lobo_test_building8)
    )
    DatasetCatalog.register('lobo_train_building4', partial(get_dataset_dicts, lobo_train_building4))
    MetadataCatalog.get('lobo_train_building4').set(**metadata)
    DatasetCatalog.register('lobo_test_building4', partial(get_dataset_dicts, lobo_test_building4, excludes))
    MetadataCatalog.get('lobo_test_building4').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building4, lobo_test_building4)
    )
    DatasetCatalog.register('lobo_train_building7', partial(get_dataset_dicts, lobo_train_building7))
    MetadataCatalog.get('lobo_train_building7').set(**metadata)
    DatasetCatalog.register('lobo_test_building7', partial(get_dataset_dicts, lobo_test_building7, excludes))
    MetadataCatalog.get('lobo_test_building7').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building7, lobo_test_building7)
    )
    DatasetCatalog.register('lobo_train_building3', partial(get_dataset_dicts, lobo_train_building3))
    MetadataCatalog.get('lobo_train_building3').set(**metadata)
    DatasetCatalog.register('lobo_test_building3', partial(get_dataset_dicts, lobo_test_building3, excludes))
    MetadataCatalog.get('lobo_test_building3').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building3, lobo_test_building3)
    )
    DatasetCatalog.register('lobo_train_building12', partial(get_dataset_dicts, lobo_train_building12))
    MetadataCatalog.get('lobo_train_building12').set(**metadata)
    DatasetCatalog.register('lobo_test_building12', partial(get_dataset_dicts, lobo_test_building12, excludes))
    MetadataCatalog.get('lobo_test_building12').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building12, lobo_test_building12)
    )
    DatasetCatalog.register('lobo_train_building10', partial(get_dataset_dicts, lobo_train_building10))
    MetadataCatalog.get('lobo_train_building10').set(**metadata)
    DatasetCatalog.register('lobo_test_building10', partial(get_dataset_dicts, lobo_test_building10, excludes))
    MetadataCatalog.get('lobo_test_building10').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building10, lobo_test_building10)
    )
    DatasetCatalog.register('lobo_train_building6', partial(get_dataset_dicts, lobo_train_building6))
    MetadataCatalog.get('lobo_train_building6').set(**metadata)
    DatasetCatalog.register('lobo_test_building6', partial(get_dataset_dicts, lobo_test_building6, excludes))
    MetadataCatalog.get('lobo_test_building6').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building6, lobo_test_building6)
    )
    DatasetCatalog.register('lobo_train_building2', partial(get_dataset_dicts, lobo_train_building2))
    MetadataCatalog.get('lobo_train_building2').set(**metadata)
    DatasetCatalog.register('lobo_test_building2', partial(get_dataset_dicts, lobo_test_building2, excludes))
    MetadataCatalog.get('lobo_test_building2').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building2, lobo_test_building2)
    )
    DatasetCatalog.register('lobo_train_building11', partial(get_dataset_dicts, lobo_train_building11))
    MetadataCatalog.get('lobo_train_building11').set(**metadata)
    DatasetCatalog.register('lobo_test_building11', partial(get_dataset_dicts, lobo_test_building11, excludes))
    MetadataCatalog.get('lobo_test_building11').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building11, lobo_test_building11)
    )
    DatasetCatalog.register('lobo_train_building1', partial(get_dataset_dicts, lobo_train_building1))
    MetadataCatalog.get('lobo_train_building1').set(**metadata)
    DatasetCatalog.register('lobo_test_building1', partial(get_dataset_dicts, lobo_test_building1, excludes))
    MetadataCatalog.get('lobo_test_building1').set(
        **metadata, vis_ind=get_vis_indices(lobo_test_building1, lobo_test_building1)
    )
    # fmt: on
    # *******************************************************************************

def get_dataset_dicts(traj_paths: List[Path], excludes: List[Path] = None) -> List[Dict]:
    """Get the dataset dict from disk.

    NOTE: It only sets the file names. The dataset mapper in `mapper.py`
    will load the actual content and add them to the dict.

    Args:
        traj_paths: list of trajectory path base/building/trajectory
        excludes: will excludes those samples in this list
    Returns:
        Dataset Dict: [
           {'file_name', 'image_id', 'height', 'width',
            'depth_file_name', 'glass_file_name', 'sem_seg_file_name',
            'annotations': {'bbox', 'bbox_mode', 'segmentation', 'category_id'}
        }, ...]
    """
    dataset_dicts = []
    image_id = 0
    excludes = [] if excludes is None else excludes

    for traj_path in traj_paths:
        json_file_names = sorted((traj_path / Path('obj_json')).iterdir())
        rf_npy_names = sorted((traj_path / Path('rf_npy')).iterdir())
        seg_npy_names = sorted((traj_path / Path('seg_npy')).iterdir())
        lidar_npy_names = sorted((traj_path / Path('lidar_npy')).iterdir())
        glass_npy_names = sorted((traj_path / Path('glass_npy')).iterdir())

        for json_file_name, rf_npy_name, seg_npy_name, lidar_npy_name, glass_npy_name in zip(
            json_file_names, rf_npy_names, seg_npy_names, lidar_npy_names, glass_npy_names
        ):
            # skip excluded samples
            if rf_npy_name in excludes:
                continue

            record = {
                'file_name': str(rf_npy_name),
                'sem_seg_file_name': str(seg_npy_name),
                'depth_file_name': str(lidar_npy_name),
                'glass_file_name': str(glass_npy_name),
                'image_id': image_id,
                'height': 64,
                'width': 512,
            }
            image_id += 1

            # read json and get object bbox
            with open(json_file_name) as f:
                items = json.load(f)
            #
            objs = []
            for item in items:
                pts = np.array(item['points'])
                px = pts[:, 0]
                py = pts[:, 1]
                label = obj_name2id[item['label']]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": 0,  # =BoxMode.XYXY_ABS
                    "segmentation": [],  # [poly],
                    "category_id": label,
                }
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def get_vis_indices(val_trajs: List[str], static_1k_trajs: List[str]) -> List[int]:
    """Get the validation indices for logging images.
    Select the first and the middle one for each trajectory.

    NOTE: Only visualize the static 1K images, otherwise there will be too many.
    This function finds the correct static 1K indices in the `val_trajs`

    Args:
        val_trajs: the validation trajectories
        static_1k_trajs: the static 1K trajectories.
    Returns:
        vis_indices: the visualization indices for logging images
    """
    num_traj_files = [len(list((traj_path / Path('rf_npy')).iterdir())) for traj_path in val_trajs]
    num_traj_files.insert(0, 0)
    traj_start_ind = np.cumsum(num_traj_files)  # [0, num_traj1, num_traj1+num_traj2, ...]

    picks = [(True if traj_path in static_1k_trajs else False) for traj_path in val_trajs]

    # select the first one and the middle one
    vis_indices = []
    for i in range(1, len(traj_start_ind)):
        if not picks[i - 1]:
            continue
        vis_indices.append(traj_start_ind[i - 1])
        vis_indices.append(traj_start_ind[i - 1] + num_traj_files[i] // 2)

    return vis_indices
