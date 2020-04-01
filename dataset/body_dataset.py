import os
import torch
import numpy as np
from glob import glob
import itertools

from dataset import data_util
import cv2

import util

_COLOR_MAP = np.asarray([[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]])
_DATA_ROOT = '/data/anpei/facial-data/seg_body'
_CALIB_ROOT = os.path.join(_DATA_ROOT, 'Calib_T_samba')

def _load_calib(data_root=_CALIB_ROOT):
    # Get camera intrinsics
    cameras = sorted(glob(os.path.join(data_root, '*/')))
    
    cam2worlds = {}
    cam_intrinsics = {}

    for cam_dir in cameras:
        cam_id = os.path.basename(cam_dir[:-1]).split('_')[0]
        # print(cam_id, cam_dir)

        cam_int = np.eye(4)
        fs= cv2.FileStorage(
            os.path.join(cam_dir, 'intrinsic.xml'), 
            cv2.FILE_STORAGE_READ)
        cam_int[:3, :3] = fs.getNode('M').mat()
        
        cam_RT = cv2.FileStorage(os.path.join(cam_dir, 'extrinsics.xml'), cv2.FILE_STORAGE_READ)
        cam_R = cam_RT.getNode('R').mat()
        cam_T = cam_RT.getNode('T').mat().squeeze()
        cam_RT = np.eye(4)
        cam_RT[:3, 3] = cam_T
        cam_RT[:3, :3] = cam_R

        cam2worlds[cam_id] = np.linalg.inv(cam_RT)
        cam_intrinsics[cam_id] = cam_int

    return cam2worlds, cam_intrinsics


def _load_seg_img(img_fp, trgt_sidelength=256, margin=10, crop=True, intrinsics=None):

    seg_img = cv2.imread(img_fp, cv2.IMREAD_UNCHANGED).astype(np.float32)
    seg_img /= 10.0
    H, W = seg_img.shape

    cx, cy = intrinsics[:2, 2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    # find mask region
    if crop:
        assert intrinsics is not None

        y_coord, x_coord = np.where(seg_img != 0)

        bbox = np.asarray([np.min(y_coord), np.min(x_coord), np.max(y_coord), np.max(x_coord)])
        sidelength = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        center = np.asarray([bbox[2] + bbox[0], bbox[3] + bbox[1]]) / 2.0
        bbox = np.asarray([
            max(center[0]-sidelength/2.0 - margin, 0), 
            max(center[1]-sidelength/2.0 - margin, 0), 
            min(center[0]+sidelength/2.0 + margin, H),
            min(center[1]+sidelength/2.0 + margin, W)]).astype(np.int64)
        
        shift_x = bbox[1]
        shift_y = H - bbox[2]

        cx -= shift_x
        cy -= shift_y

        seg_img = seg_img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        print('crop image: (%d,%d)->(%d,%d)'%(H, W, seg_img.shape[0], seg_img.shape[1]))

        H = W = sidelength
    
    if trgt_sidelength is not None:
        cx *= (trgt_sidelength / W)
        cy *= (trgt_sidelength / H)
        fx *= (trgt_sidelength / W)
        fy *= (trgt_sidelength / H)

        print('resice seg image: (%d, %d) -> (%d, %d)'%(H, W, trgt_sidelength, trgt_sidelength))

        seg_img = cv2.resize(seg_img, (trgt_sidelength, trgt_sidelength), interpolation=cv2.INTER_NEAREST)

    intrinsic = np.array([  [fx, 0., cx, 0.],
                            [0., fy, cy, 0],
                            [0., 0, 1, 0],
                            [0, 0, 0, 1]])

    seg_img = seg_img[None, :, :]
    seg_img = seg_img.reshape(seg_img.shape[0], -1).transpose(1, 0)
    print(seg_img.shape)

    return seg_img, intrinsics


def _campos2matrix(cam_pos, cam_center=None, cam_up=None):
    _cam_target = np.asarray([0,0.11,0.1]).reshape((1, 3)) if cam_center is None else cam_center
    _cam_up = np.asarray([0.0, 1.0, 0.0]) if cam_up is None else cam_up

    cam_dir = (_cam_target-cam_pos)
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    cam_right = np.cross(cam_dir, _cam_up)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_dir)

    cam_R = np.concatenate([cam_right, -cam_up, cam_dir], axis=0).T

    cam_P = np.eye(4)
    cam_P[:3, :3] = cam_R
    cam_P[:3, 3] = cam_pos

    return cam_P

def _rand_cam_spiral(R=1.0, num_samples=100):
    _v = 1.0
    _omega = 20

    t = np.linspace(0, R * 2, num=num_samples)
    z = -R + t * _v

    r = np.sqrt(R**2-z**2)
    x = r * np.cos(_omega * t)
    y = r * np.sin(_omega * t)
    cam_pos = (np.stack([x, z, y])).T

    return cam_pos

def _rand_cam_cube(Rx=0.5, Ry=0.5, Rz=1.0, num_samples=15):
    x = (np.random.rand(num_samples, 1) * 2.0 - 1.0) * Rx
    y = (np.random.rand(num_samples, 1) * 2.0 - 1.0) * Ry
    z = -(np.random.rand(num_samples, 1)) * Rz

    cam_pos = np.concatenate([x, z, y], axis=1)
    # print('*** cam_pos = ', cam_pos.shape, cam_pos)
    return cam_pos

def _rand_cam_sphere(R=2.0, num_samples=15, random=False):
    side_len = np.ceil(np.sqrt(num_samples)).astype(np.uint8)
    cam_pos = []

    _PHI_RANGE = [np.pi/2-0.6, np.pi/2+0.6]
    _THETA_RANGE = [np.pi/2-0.3, np.pi/2+0.3]

    if random:
        _theta = np.random.random_sample((side_len,)) * (_THETA_RANGE[1]-_THETA_RANGE[0]) + _THETA_RANGE[0]
        _phi = np.random.random_sample((side_len,)) * (_PHI_RANGE[1]-_PHI_RANGE[0]) + _PHI_RANGE[0]
    else:
        _theta = np.linspace(_THETA_RANGE[0], _THETA_RANGE[1], num=side_len)
        _phi = np.linspace(_PHI_RANGE[0], _PHI_RANGE[1], num=side_len)
    
    _p = 1
    _idx = 0

    for theta in _theta:
        for i in range(len(_phi)):
            _cur_idx = int(_idx+i*_p)
            phi = _phi[_cur_idx]
            x = R * np.sin(theta) * np.cos(phi)
            y = R * np.sin(theta) * np.sin(phi)
            z = R * np.cos(theta)
            
            cam_pos.append(np.array([x, z, y]))
        
        _p *= -1
        _idx = _cur_idx
    
    cam_pos = cam_pos[:num_samples]
    cam_pos = np.asarray(cam_pos)
    
    return cam_pos

def _rand_cam_plane(R=1.2, num_samples=15):
    side_len = np.ceil(np.sqrt(num_samples)).astype(np.uint8)
    cam2world = []

    _X_RANGE = [0.05, -0.05]
    _Y_RANGE = [0.13, 0.12]

    _x = np.linspace(_X_RANGE[0], _X_RANGE[1], side_len)
    _y = np.linspace(_Y_RANGE[0], _Y_RANGE[1], side_len)

    _p = 1
    _idx = 0

    for x in _x:
        for i in range(len(_y)):
            _cur_idx = int(_idx + i*_p)
            y = _y[_cur_idx]
            cam2world.append(_campos2matrix(
                np.array([x, y, R]), np.array([x, y, R-1.0]).reshape((1, 3))))
        _p *= -1
        _idx = _cur_idx

    cam2world = np.asarray(cam2world)
    
    return cam2world

def _get_random_poses(sample_radius, num_samples, mode):    
    if mode == 'spiral':
        cam_pos = _rand_cam_spiral(sample_radius, num_samples)
    elif mode == 'sphere':
        cam_pos = _rand_cam_sphere(sample_radius, num_samples)
    elif mode == 'load':
        _DEFAULT_POSE = '/data/anpei/facial-data/seg_face_8000/cam2world.npy'
        cam_pos = np.load(_DEFAULT_POSE)
        cam_pos = cam_pos[:num_samples, :3, 3].squeeze()
    elif mode == 'plane':
        return _rand_cam_plane(sample_radius, num_samples)
    else:
        cam_pos = _rand_cam_cube(num_samples=num_samples)
    
    cam2world = []

    for i in range(num_samples):
        cam2world.append(_campos2matrix(cam_pos[i]))

    cam2world = np.asarray(cam2world)

    return cam2world


class BPInstanceDataset():
    """Body part segmentation for a single person."""

    def __init__(self,
                 instance_idx,
                 instance_path,
                 cam2worlds,
                 intrinsics,
                 data_type='seg',
                 img_sidelength=None,
                 sample_observations=None,
                 shuffle=True):

        self.data_root = instance_path
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.data_type = data_type

        self.color_paths = sorted(glob(os.path.join(self.data_root, 'semantic_mask', '*.png')))

        self.intrinsics = intrinsics
        self.poses = cam2worlds

        if shuffle:
            idxs = np.random.permutation(len(self.color_paths))
            self.color_paths = [self.color_paths[x] for x in idxs]

        if sample_observations is not None:
            self.color_paths = [self.color_paths[idx] for idx in sample_observations]

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        
        img_fp = self.color_paths[idx]
        cam_id = os.path.basename(img_fp).split('.')[1].split('_')[0]

        # print(cam_id)

        img, intrinsics = _load_seg_img(
            img_fp, self.img_sidelength, crop=True, 
            intrinsics=self.intrinsics[cam_id])
        pose = self.poses[cam_id]

        uv = np.mgrid[0:self.img_sidelength, 0:self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            'instance_idx': torch.Tensor([self.instance_idx]).squeeze(),
            'observation_idx': idx,
            'rgb': torch.from_numpy(img).float(),
            'pose': torch.from_numpy(pose).float(),
            'uv': uv,
            'intrinsics': torch.Tensor(intrinsics).float(),
        }

        return sample


class BPInstanceRandomPose(BPInstanceDataset):
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(self,
                 instance_idx,
                 intrinsics,
                 num_observations=15,
                 sample_radius=2.0,
                 img_sidelength=128,
                 mode='cube'):

        self.instance_idx = instance_idx
        self.intrinsics = intrinsics
        self.img_sidelength = img_sidelength

        self.poses = _get_random_poses(
            sample_radius, num_samples=num_observations, mode=mode)
            


class BodyPartDataset(torch.utils.data.Dataset):
    """Dataset for 1000 face segs with 25 views each, where each datapoint is a FaceInstanceDataset.
    
    Body part labels :

    0: 'background'	1: 'skin'	2: 'nose'
    3: 'eye_g'	4: 'l_eye'	5: 'r_eye'
    6: 'l_brow'	7: 'r_brow'	8: 'l_ear'
    9: 'r_ear'	10: 'mouth'	11: 'u_lip'
    12: 'l_lip'	13: 'hair'	14: 'hat'
    
    """

    def __init__(self,
                 root_dir=_DATA_ROOT,
                 data_type='seg',
                 img_sidelength=128,
                 sample_instances=None,
                 sample_observations=None):

        # print('*** load_depth = ', load_depth)

        tot_instances = sorted(glob(os.path.join(root_dir, '[0-9]*/')))
        cam2world, intrinsics = _load_calib(_CALIB_ROOT)
        
        if sample_instances is not None:
            tot_instances=[tot_instances[idx] for idx in sample_instances if idx < len(tot_instances)]
        self.num_instances = len(tot_instances)

        self.all_instances = [BPInstanceDataset(    instance_idx=instance_idx,
                                                    instance_path=instance_fp,
                                                    cam2worlds=cam2world, 
                                                    intrinsics=intrinsics,
                                                    data_type=data_type,
                                                    img_sidelength=img_sidelength,
                                                    sample_observations=sample_observations)
                              for instance_idx, instance_fp in enumerate(tot_instances)]

        assert len(self.all_instances) == self.num_instances

        self.num_per_instance_observations =  [len(obj) for obj in self.all_instances]
        self.num_instances = len(self.all_instances)

        if data_type == 'seg':
            self.color_map = torch.tensor(_COLOR_MAP, dtype=torch.float32) / 255.0
        else:
            self.color_map = None

        print('> Load %d instances from %s.'%(self.num_instances, root_dir))

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        for instance in self.all_instances:
            instance.set_img_sidelength(new_img_sidelength)

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        # print('obj_idx = ', obj_idx, 'rel_idx = ', rel_idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        ground_truth = [{'rgb':ray_bundle['rgb']} for ray_bundle in observations]
        # print('** ground_truth = ', ground_truth[0]['rgb'].shape)

        return observations, ground_truth


class BPRandomPoseDataset(BodyPartDataset):

    def __init__(self,
                 sample_radius=1.1,
                 img_sidelength=128,
                 num_instances=100,
                 num_observations=15,
                 mode='load'
                 ):

        _DEFAULT_INT = '/data/anpei/facial-data/seg_face_2000/intrinsics.txt'

        if isinstance(num_instances, int):
            num_instances = list(range(num_instances))

        intrinsics = data_util.parse_intrinsics(_DEFAULT_INT, trgt_sidelength=img_sidelength)

        self.all_instances = [FaceInstanceRandomPose(   instance_idx=idx,
                                                        intrinsics=intrinsics,
                                                        num_observations=num_observations,
                                                        sample_radius=sample_radius,
                                                        img_sidelength=128,
                                                        mode=mode)

                                for idx in num_instances]

        self.color_map = torch.tensor(_COLOR_MAP, dtype=torch.float32) / 255.0
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]