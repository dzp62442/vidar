import os
import json
import fire
import warnings
import numpy as np
import torch
from tqdm import tqdm

from vidar.core.trainer import Trainer
from vidar.core.wrapper import Wrapper
from vidar.utils.config import cfg_has, dataset_prefix
from vidar.utils.data import make_batch
from vidar.utils.types import is_str
from vidar.utils.setup import setup_datasets, setup_dataloader
from vidar.utils.logging import pcolor

from camviz import BBox3D
from camviz import Camera as CameraCV
from camviz import Draw

from vidar.geometry.camera import Camera
from vidar.geometry.pose import Pose
from vidar.utils.data import make_batch, fold_batch, modrem
from vidar.utils.flip import flip_batch
from vidar.utils.viz import viz_depth, viz_optical_flow, viz_semantic

class Displayer:
    """
    Displayer class for dataset and result

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    ckpt : String
        Name of the model checkpoint to start from
    """
    def __init__(self, cfg, ckpt=None):
        super().__init__()

        self.cfg = cfg
        self.result_cfg = cfg.save
        self.ckpt_name = 'FSM_MR_6cams_DDAD'
        self.datasets, self.datasets_cfg = setup_datasets(cfg.datasets, verbose=True, stack=False)
        print('datasets_cfg:', self.datasets_cfg)
        self.dataloaders = {
            key: setup_dataloader(val, self.datasets_cfg[key][0].dataloader, key)
            for key, val in self.datasets.items() if key in self.datasets_cfg.keys()
        }
        self.prefixes = {
            key: [dataset_prefix(self.datasets_cfg[key][n], n) for n in range(len(val))]
            for key, val in self.datasets_cfg.items() if 'name' in self.datasets_cfg[key][0].__dict__.keys()
        }
        self.all_modes = ['train', 'mixed', 'validation', 'test']
        self.mode = 'validation'
        self.tasks = ['rgb', 'depth', 'fwd_optical_flow', 'bwd_optical_flow','semantic']
        self.cam_colors = ['red', 'blu', 'gre', 'yel', 'mag', 'cya'] * 100
        self.delete_1d_list = ['rgb', 'intrinsics', 'depth', 'pose']  # 去除多余的第一个维度，与 display_sample.py 保持一致
        self.num_cams = max(np.array(self.datasets_cfg[self.mode][0].cameras).shape)  # 相机数目
        self.auto_display = False  # 自动播放
        self.start = cfg.display.start if cfg_has(cfg, 'display') else 0
        self.stop = cfg.display.stop if cfg_has(cfg, 'display') else 1000000000
        print(self.start, self.stop)

        # 生成深度图结果保存路径
        self.result_depth_paths = []
        for c in self.datasets_cfg[self.mode][0].cameras[0]:
            prefix_name = self.prefixes[self.mode][0][:-1] + str(c)
            self.result_depth_paths.append(os.path.join(self.result_cfg.folder, prefix_name, self.ckpt_name))
        print('result_depth_paths:', self.result_depth_paths)
    
    
    def read_depth_results(self, batch_idx):
        """根据配置文件读取保存在文件夹中的深度估计结果"""
        depth_list = []
        for path in self.result_depth_paths:
            npz_file = '{:010d}_depth(0)_pred.npz'.format(batch_idx)
            
            try:  # 解压成功
                data = np.load(os.path.join(path, npz_file))
            except Exception as e:  # 文件无法打开或读取错误
                warnings.warn("{} occurred {}".format(npz_file, e), UserWarning)
                return str(e)
            
            try:  # 数据未损坏
                depth = torch.from_numpy(np.array([data['depth']]))
            except Exception as e:  # 数据损坏无法处理
                warnings.warn("{} data occurred {}".format(batch_idx, e), UserWarning)
                return str(e)
            
            depth_list.append(depth)
        depth_tensor = torch.stack(depth_list, dim=0)
        return {0: depth_tensor}
        

    def show(self):
        mode = self.mode
        num_cams = self.num_cams
        
        # 遍历所有数据集
        for dataset_idx, (dataset_cfg, dataloader, prefix) in \
                enumerate(zip(self.datasets_cfg[mode], self.dataloaders[mode], self.prefixes[mode])):
            progress_bar = self.val_progress_bar(dataloader, prefix, ncols=120)
            
            print('-----\n', dataset_cfg, '\n-----')
            wh = torch.Size(dataset_cfg.augmentation.resize[::-1])
            print('wh:', wh, type(wh))

            # 初始化该数据集的可视化窗口
            draw = Draw((wh[0] * 4, wh[1] * 3), width=3400)
            draw.add2DimageGrid('cam', (0.0, 0.0, 0.5, 1.0), n=(4, 2), res=wh)
            view_pose = torch.tensor([[-0.9997,  0.0233, -0.0089, -0.0083],
                                            [ 0.0088, -0.0069, -0.9999, -0.9769],
                                            [-0.0234, -0.9997,  0.0067, -1.1443],
                                            [ 0.0000,  0.0000,  0.0000,  1.0000]])
            draw.add3Dworld('wld', (0.5, 0.0, 1.0, 1.0), pose=view_pose)  # 初始化 3D 窗口的虚拟观察相机的位姿

            draw.addTexture('cam', n=num_cams)
            draw.addBuffer3f('lidar', 1000000, n=num_cams)
            draw.addBuffer3f('color', 1000000, n=num_cams)
            
            # 遍历该数据集的所有帧
            for batch_idx, batch in progress_bar:
                if batch_idx < self.start:
                    continue
                if batch_idx > self.stop:
                    break
                
                # 去除多余的第一个维度，与 display_sample.py 保持一致
                for key, val in batch.items():  
                    if key in self.delete_1d_list:
                        for k, v in batch[key].items():
                            batch[key][k] = v[0]
                
                # 加载数据
                rgb = batch['rgb']                  
                intrinsics = batch['intrinsics']
                depth = self.read_depth_results(batch_idx)
                if (is_str(depth)):  # 读取深度结果失败
                    continue
                batch['depth'] = depth
                pose = batch['pose']

                pose = Pose.from_dict(pose, to_global=True)
                cam = Camera.from_dict(intrinsics, rgb, pose)

                keys = [key for key in self.tasks if key in batch.keys()]

                points = {}
                for key, val in cam.items():
                    points[key] = cam[key].reconstruct_depth_map(
                        depth[key], to_world=True).reshape(num_cams, 3, -1).permute(0, 2, 1)
                
                # 可视化
                # draw.add3Dworld('wld', (0.5, 0.0, 1.0, 1.0), pose=cam[0].Tcw.T[2])

                camcv = []
                for i in range(num_cams):
                    camcv.append({key: CameraCV.from_vidar(val, i) for key, val in cam.items()})

                t, k = 0, 0
                key = keys[k]
                change = True
                color = True

                while draw.input():
                    if draw.SPACE:
                        color = not color
                        change = True
                    if draw.RETURN:
                        self.auto_display = not self.auto_display
                        change = True
                    if draw.RIGHT:
                        print("keys: {}, k: {}, key: {}".format(keys, k, key))
                        change = True
                        k = (k + 1) % len(keys)
                        while t not in batch[keys[k]].keys():
                            k = (k + 1) % len(keys)
                        key = keys[k]
                    if draw.LEFT:
                        print("keys: {}, k: {}, key: {}".format(keys, k, key))
                        change = True
                        k = (k - 1) % len(keys)
                        while t not in batch[keys[k]].keys():
                            k = (k - 1) % len(keys)
                        key = keys[k]
                    if draw.UP:
                        change = True
                        t = self.change_key(batch[key], t, 1)
                        while t not in batch[keys[k]].keys():
                            t = self.change_key(batch[key], t, 1)
                        break
                    if draw.DOWN:
                        change = True
                        t = self.change_key(batch[key], t, -1)
                        while t not in batch[keys[k]].keys():
                            t = self.change_key(batch[key], t, -1)
                    if change:
                        change = False
                        # print("key: {}, k: {}, t: {}".format(key, k, t))
                        for i in range(num_cams):
                            img = batch[key][t][i]
                            if key == 'depth':
                                img = viz_depth(img, filter_zeros=True)
                            elif key in ['fwd_optical_flow', 'bwd_optical_flow']:
                                img = viz_optical_flow(img)
                            elif key == 'semantic':
                                ontology = json.load(open('vidar/datasets/ontologies/%s.json' % batch['tag'][0]))
                                img = viz_semantic(img, ontology)
                            draw.updTexture('cam%d' % i, img)
                            draw.updBufferf('lidar%d' % i, points[t][i])
                            draw.updBufferf('color%d' % i, rgb[t][i])

                    draw.clear()
                    for i in range(num_cams):
                        draw['cam%d%d' % modrem(i, 2)].image('cam%d' % i)
                        draw['wld'].size(1).color(self.cam_colors[i]).points('lidar%d' % i, ('color%d' % i) if color else None)
                        for cam_key, cam_val in camcv[i].items():
                            clr = self.cam_colors[i] if cam_key == t else 'gra'
                            tex = 'cam%d' % i if cam_key == t else None
                            draw['wld'].object(cam_val, color=clr, tex=tex)
                        
                    # print(draw.screens['wld'].viewer.T)  # 3D 窗口的虚拟观察相机的位姿
                    draw.update(5)
                    if(self.auto_display):
                        break
    

    def val_progress_bar(self, dataloader, prefix, ncols=None):
        """Print validation progress bar on screen"""
        return tqdm(enumerate(dataloader, 0),
                    unit='im', unit_scale=dataloader.batch_size,
                    total=len(dataloader), smoothing=0,
                    ncols=ncols, desc=prefix)
    
    
    def change_key(self, dic, c, n):
        steps = sorted(dic.keys())
        return steps[(steps.index(c) + n) % len(steps)]