import os
import sys
import pickle
import copy
import numpy as np
from skimage import io
from pathlib import Path
import torch
import spconv
from pcdet.utils import box_utils, object3d_utils, calibration, common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets.data_augmentation.dbsampler import DataBaseSampler
from pcdet.datasets import DatasetTemplate
import os
import torch
from tensorboardX import SummaryWriter
import time
import glob
import re
import datetime
import argparse
from pathlib import Path
import torch.distributed as dist
from pcdet.datasets import build_dataloader
from pcdet.models import build_bench_network
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from eval_utils import eval_utils

def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=80, required=False, help='Number of epochs to train for')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def bench_getitem(pc,test_set):
    sample_idx = '000000'
    points = pc
    input_dict = {
        'points': points,
        'sample_idx': sample_idx,
    }

    example = prepare_data(input_dict,test_set)

    example['sample_idx'] = sample_idx

    return example
def prepare_data(input_dict,test_set):
    voxel_generator = spconv.utils.VoxelGeneratorV2(
        voxel_size=[0.2, 0.2, 0.2],
        point_cloud_range=[-60, -60, -4, 61.6, 80.8, 4],
        max_num_points=50,
        max_voxels=30000)
    sample_idx = input_dict['sample_idx']
    points = input_dict['points']

    points = points[:, :4]

    voxel_grid = voxel_generator.generate(points)

    voxels = voxel_grid["voxels"]
    coordinates = voxel_grid["coordinates"]
    num_points = voxel_grid["num_points_per_voxel"]

    voxel_centers = (coordinates[:, ::-1] + 0.5) * voxel_generator.voxel_size \
                    + voxel_generator.point_cloud_range[0:3]
    points = common_utils.mask_points_by_range(points, [-60, -60, -4, 61.6, 80.8, 4])

    example = {}
    example.update({
        'batch_size':1,
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        'voxel_centers': voxel_centers,
        # 'calib': input_dict['calib'],
        'points': points
    })
    batch_list=[example]
    example = test_set.collate_batch(batch_list)
    return example
def example_convert_to_torch(example, dtype=torch.float32):
    device = torch.cuda.current_device()
    example_torch = {}
    float_names = [
        'voxels', 'anchors', 'box_reg_targets', 'reg_weights', 'part_labels',
        'gt_boxes', 'voxel_centers', 'reg_src_targets', 'points',
    ]

    for k, v in example.items():
        if k in float_names:
            try:
                example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(dtype)
            except RuntimeError:
                example_torch[k] = torch.zeros((v.shape[0], 1, 7), dtype=torch.float32, device=device).to(dtype)
        elif k in ['coordinates', 'box_cls_labels', 'num_points', 'seg_labels']:
            example_torch[k] = torch.tensor(v, dtype=torch.int32, device=device)
        else:
            example_torch[k] = v
    return example_torch
def statistics_info(ret_dict, metric, disp_dict):
    if cfg.MODEL.RCNN.ENABLED:
        for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
            metric['recall_roi_%s' % str(cur_thresh)] += ret_dict['roi_%s' % str(cur_thresh)]
            metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict['rcnn_%s' % str(cur_thresh)]
        metric['gt_num'] += ret_dict['gt']
        min_thresh = cfg.MODEL.TEST.RECALL_THRESH_LIST[0]
        disp_dict['recall_%s' % str(min_thresh)] = \
            '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])
# def collate_batch(batch_list, _unused=False):
#         example_merged = defaultdict(list)
#         for example in batch_list:
#             for k, v in example.items():
#                 example_merged[k].append(v)
#         ret = {}
#         for key, elems in example_merged.items():
#             if key in ['voxels', 'num_points', 'voxel_centers', 'seg_labels', 'part_labels', 'bbox_reg_labels']:
#                 ret[key] = np.concatenate(elems, axis=0)
#             elif key in ['coordinates', 'points']:
#                 coors = []
#                 for i, coor in enumerate(elems):
#                     coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
#                     coors.append(coor_pad)
#                 ret[key] = np.concatenate(coors, axis=0)
#             elif key in ['gt_boxes']:
#                 max_gt = 0
#                 batch_size = elems.__len__()
#                 for k in range(batch_size):
#                     max_gt = max(max_gt, elems[k].__len__())
#                 batch_gt_boxes3d = np.zeros((batch_size, max_gt, elems[0].shape[-1]), dtype=np.float32)
#                 for k in range(batch_size):
#                     batch_gt_boxes3d[k, :elems[k].__len__(), :] = elems[k]
#                 ret[key] = batch_gt_boxes3d
#             else:
#                 ret[key] = np.stack(elems, axis=0)
#         ret['batch_size'] = batch_list.__len__()
#         return ret
def PartA2(pc):
    args, cfg = parge_config()
    log_file = '/root/PCDet/bench_out'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    test_set = KittiDataset(
        class_names=['car','bigcar','ped','cyc'],
        training=False,)
    model=build_bench_network(test_set)
    model.load_params_from_file(filename=args.ckpt,logger=logger)
    model.cuda()
    model.eval()
    example = bench_getitem(pc,test_set)
    input_dict = example_convert_to_torch(example)
    pred_dicts, ret_dict = model(input_dict)
    disp_dict = {}
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.TEST.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
    statistics_info(ret_dict, metric, disp_dict)

    annos = test_set.generate_annotations(input_dict, pred_dicts, ['car','bigcar','ped','cyc'])
    return annos
class BaseKittiDataset(DatasetTemplate):
    def __init__(self):
        super().__init__()

    def set_split(self, split):
        self.__init__(self.root_path, split)

    def get_lidar(self, idx):
        if not cfg.DATA_CONFIG.TS_DATA:
            lidar_file = os.path.join(self.root_split_path, 'velodyne', '%s.bin' % idx)
            assert os.path.exists(lidar_file)
            return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        else:
            lidar_file = os.path.join(self.root_split_path, 'ts2kitti_lidar', '%s.npz' % idx)
            assert os.path.exists(lidar_file)
            pc = np.load(lidar_file, allow_pickle=True)['pc'].reshape(-1, 3)
            intensity = np.load(lidar_file, allow_pickle=True)['intensity'].reshape(-1, 1)
            return np.concatenate((pc,intensity), axis=1).reshape(-1,4)

    def get_image_shape(self, idx):
        img_file = os.path.join(self.root_split_path, 'image_2', '%s.png' % idx)
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        if not cfg.DATA_CONFIG.TS_DATA:
            label_file = os.path.join(self.root_split_path, 'label_2', '%s.txt' % idx)
            assert os.path.exists(label_file)
            return object3d_utils.get_objects_from_label(label_file)
        else:
            label_file = os.path.join(self.root_split_path, 'ts2kitti_label_new', '%s.txt' % idx)
            assert os.path.exists(label_file)
            return object3d_utils.get_objects_from_label(label_file)
    def get_calib(self, idx):
        calib_file = os.path.join(self.root_split_path, 'calib', '%s.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.root_split_path, 'planes', '%s.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def generate_prediction_dict(input_dict, index, record_dict):
        # finally generate predictions.
        sample_idx = input_dict['sample_idx'][index] if 'sample_idx' in input_dict else -1
        try:
            boxes3d_lidar_preds = record_dict['boxes'].cpu().numpy()
        except:
            boxes3d_lidar_preds = record_dict['boxes'].detach().cpu().numpy()
        if boxes3d_lidar_preds.shape[0] == 0:
            return {'sample_idx': sample_idx}

        if not cfg.DATA_CONFIG.TS_DATA:
            calib = input_dict['calib'][index]
            image_shape = input_dict['image_shape'][index]

            boxes3d_camera_preds = box_utils.boxes3d_lidar_to_camera(boxes3d_lidar_preds, calib)
            boxes2d_image_preds = box_utils.boxes3d_camera_to_imageboxes(boxes3d_camera_preds, calib,
                                                                     image_shape=image_shape)
        # predictions
            predictions_dict = {
                'bbox': boxes2d_image_preds,
                'box3d_camera': boxes3d_camera_preds,
                'box3d_lidar': boxes3d_lidar_preds,
                'scores': record_dict['scores'].cpu().numpy(),
                'label_preds': record_dict['labels'].cpu().numpy(),
                'sample_idx': sample_idx,
            }
        else:
            boxes3d_camera_preds = box_utils.ts_boxes3d_lidar_to_camera(boxes3d_lidar_preds)
            predictions_dict = {
                'bbox': np.zeros((boxes3d_lidar_preds.shape[0],4)),
                'box3d_camera': boxes3d_camera_preds,
                'box3d_lidar': boxes3d_lidar_preds,
                'scores': record_dict['scores'].detach().cpu().numpy(),
                'label_preds': record_dict['labels'].detach().cpu().numpy(),
                'sample_idx': sample_idx,
            }
        return predictions_dict

    @staticmethod
    def generate_annotations(input_dict, pred_dicts, class_names, save_to_file=False, output_dir=None):
        def get_empty_prediction():
            ret_dict = {
                'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([]),
                'boxes_lidar': np.zeros([0, 7])
            }
            return ret_dict

        def generate_single_anno(idx, box_dict):
            num_example = 0
            if 'bbox' not in box_dict:
                return get_empty_prediction(), num_example

            area_limit = image_shape = None
            if cfg.MODEL.TEST.BOX_FILTER['USE_IMAGE_AREA_FILTER']:
                image_shape = input_dict['image_shape'][idx]
                area_limit = image_shape[0] * image_shape[1] * 0.8

            sample_idx = box_dict['sample_idx']
            box_preds_image = box_dict['bbox']
            box_preds_camera = box_dict['box3d_camera']
            box_preds_lidar = box_dict['box3d_lidar']
            scores = box_dict['scores']
            label_preds = box_dict['label_preds']

            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': [], 'score': [], 'boxes_lidar': []}

            for box_camera, box_lidar, bbox, score, label in zip(box_preds_camera, box_preds_lidar, box_preds_image,
                                                                 scores, label_preds):
                if area_limit is not None:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0] or bbox[2] < 0 or bbox[3] < 0:
                        continue
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > area_limit:
                        continue

                if 'LIMIT_RANGE' in cfg.MODEL.TEST.BOX_FILTER:
                    limit_range = np.array(cfg.MODEL.TEST.BOX_FILTER['LIMIT_RANGE'])
                    if np.any(box_lidar[:3] < limit_range[:3]) or np.any(box_lidar[:3] > limit_range[3:]):
                        continue

                if not (np.all(box_lidar[3:6] > -0.1)):
                    print('Invalid size(sample %s): ' % str(sample_idx), box_lidar)
                    continue

                anno['name'].append(class_names[int(label - 1)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box_camera[6])
                anno['bbox'].append(bbox)
                anno['dimensions'].append(box_camera[3:6])
                anno['location'].append(box_camera[:3])
                # anno['location'].append(box_lidar[:3])
                anno['rotation_y'].append(box_camera[6])
                anno['score'].append(score)
                anno['boxes_lidar'].append(box_lidar)

                num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = get_empty_prediction()

            return anno, num_example

        annos = []
        for i, box_dict in enumerate(pred_dicts):
            sample_idx = box_dict['sample_idx']
            single_anno, num_example = generate_single_anno(i, box_dict)
            single_anno['num_example'] = num_example
            single_anno['sample_idx'] = np.array([sample_idx] * num_example, dtype=np.int64)
            annos.append(single_anno)
            if save_to_file:
                cur_det_file = os.path.join(output_dir, '%s.txt' % sample_idx)
                with open(cur_det_file, 'w') as f:
                    bbox = single_anno['bbox']
                    loc = single_anno['location']
                    dims = single_anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_anno['name'][idx], single_anno['alpha'][idx], bbox[idx][0], bbox[idx][1],
                                 bbox[idx][2], bbox[idx][3], dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_anno['rotation_y'][idx], single_anno['score'][idx]),
                              file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        assert 'annos' in self.kitti_infos[0].keys()
        import pcdet.datasets.kitti.kitti_object_eval_python.eval as kitti_eval

        if 'annos' not in self.kitti_infos[0]:
            return 'None', {}
        ignore_list = kwargs['lst']
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos if info['point_cloud']['lidar_idx'] not in ignore_list]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict


class KittiDataset(BaseKittiDataset):
    def __init__(self, class_names, training):
        super().__init__()

        self.class_names = class_names
        self.training = training

        self.mode = 'TRAIN' if self.training else 'TEST'

        self.kitti_infos = []
        # self.include_kitti_data(self.mode, logger)
        self.dataset_init(class_names)

    def include_kitti_data(self, mode):
        kitti_infos = []

        for info_path in cfg.DATA_CONFIG[mode].INFO_PATH:
            info_path = cfg.ROOT_DIR / info_path
            with open(info_path, 'rb') as f:
                # print("111111111111",cfg.DATA_CONFIG.DATA_SAMPLE_RANGE)
                # infos = pickle.load(f)[:cfg.DATA_CONFIG.DATA_SAMPLE_RANGE]
                infos = pickle.load(f)[:5000]
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if cfg.LOCAL_RANK == 0 and logger is not None:
            logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def dataset_init(self,class_names):
        self.voxel_generator= spconv.utils.VoxelGeneratorV2(
        voxel_size=[0.2, 0.2, 0.2],
        point_cloud_range=[-60, -60, -4, 61.6, 80.8, 4],
        max_num_points=50,
        max_voxels=30000)

def main():
    pc = np.load('/root/PCDet/ts_data/training/ts2kitti_lidar/000000.npz')['pc']
    intensity = np.load('/root/PCDet/ts_data/training/ts2kitti_lidar/000000.npz')['intensity']
    pc = np.hstack((pc,intensity)).reshape(-1,4)
    annos= PartA2(pc)

    return annos
main()