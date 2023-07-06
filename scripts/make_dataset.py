"""
将采集的数据转换为 DDAD 格式的数据集
预先准备：
    000000 序列文件夹：将每次实验/每个场景作为一个序列
    rgb 文件夹：存放图像
    dataset_json 文件：放置于数据集根目录下，划分训练集和测试集
    相机标定结果
程序输出：
    calibration_json 文件：放置于每个序列的 calibration 文件夹下的标定数据
    scene_json 文件：放置于每个序列文件夹下的数据描述文件
目前缺少：
    calibration_json 文件中的 extrinsics
    scene_json 文件中的 pose
"""

import os
import json
import shutil
from tqdm import tqdm 
from datetime import datetime


# 定义基础参数
DATASET = "GreenTractor"  # 数据集名称
ROOT_PATH = "/data/vidar/GreenTractor"  # 数据集根目录
CALIBRATION_PATH = os.path.join(ROOT_PATH, "Calibration", "calibration.json")  # 原始标定结果
CAM_NUMS = 8
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


# 处理序列中缺失的图像文件
def process_missing_img(rgb_dir, cam_name, img_idx):
    current_img = os.path.join(rgb_dir, cam_name, str(img_idx)+'.jpg')
    if not os.path.exists(current_img):
        print('Missing image: ', current_img)
        pre_img = os.path.join(rgb_dir, cam_name, str(img_idx-1)+'.jpg')
        shutil.copy2(pre_img, current_img)  # 复制上一帧图像

# 生成 scene_json 文件
def make_scene_json(scene_json, calibration_json, rgb_dir):
    # 读取相机名称
    with open(calibration_json, 'r') as f:
        calibration_json_data = json.load(f)
    cam_names = calibration_json_data['names']
    sequence_name = scene_json.split('/')[-2]  # 序列名称，例如 000000
    sequence_length = len(os.listdir(os.path.join(rgb_dir, cam_names[0])))  # 序列长度，即图像数目

    # 初始化 scene_json_data
    scene_json_data = {
        'data': [],
        'description': '',
        'log': '',
        'metadata': {}, 
        'name': '',
        'ontologies': {},
        'samples': []
    }

    for img_idx in tqdm(range(sequence_length)):
        # 补充时间戳
        current_time = datetime.utcnow()
        timestamp = current_time.isoformat() + "Z"
        # 添加 scene_json_data['data']
        for cam_name in cam_names:
            process_missing_img(rgb_dir, cam_name, img_idx)
            data = {
                'datum': {
                    'image': {
                        'annotations': {},
                        'channels': 3,
                        'filename': os.path.join('rgb', cam_name, str(img_idx)+'.jpg'),
                        'height': IMG_HEIGHT,
                        'metadata': {},
                        'pose': calibration_json_data['extrinsics'][0],
                        'width': IMG_WIDTH
                    }
                },
                'id': {
                    'index': '0',
                    'log': '',
                    'name': cam_name,
                    'timestamp': timestamp
                },
                'key': 's{}c{}i{:05d}'.format(sequence_name, cam_name, img_idx),  # 注意图像数据的 key 格式
                'next_key': '',
                'prev_key': 's{}c{}i{:05d}'.format(sequence_name, cam_name, img_idx-1)
            }
            scene_json_data['data'].append(data)
        # 添加 scene_json_data['samples']
        sample = {
            'calibration_key': 'calib{}'.format(sequence_name),
            'datum_keys': [],
            'id': {
                'index': '0',
                'log': '',
                'name': '',
                'timestamp': timestamp
            },
            'metadata': {}
        }
        for cam_name in cam_names:
            sample['datum_keys'].append('s{}c{}i{:05d}'.format(sequence_name, cam_name, img_idx))  # 注意图像数据的 key 格式
        scene_json_data['samples'].append(sample)
        # 控制序列长度
        if (img_idx>2000):
            break

    return scene_json_data


def main():
    # 加载 dataset_json
    dataset_json = os.path.join(ROOT_PATH, DATASET+'.json')
    with open(dataset_json, 'r') as f:
        dataset_json_data = json.load(f)
    train_splits = dataset_json_data['scene_splits']['0']['filenames']
    validation_splits = dataset_json_data['scene_splits']['1']['filenames']

    # 加载标定结果
    with open(CALIBRATION_PATH, 'r') as f:
        calibration_result = json.load(f)

    # 处理训练集
    for split_name in train_splits:
        sequence_name = split_name.split('/')[0]  # 序列名称，例如 000000
        sequence_dir = os.path.join(ROOT_PATH, sequence_name)  # 序列文件夹路径

        # 制作 calibration_json
        calibration_dir = os.path.join(sequence_dir, 'calibration')  # 序列文件夹下的 calibration 文件夹
        os.makedirs(calibration_dir, exist_ok=True)
        calibration_key = 'calib{}'.format(sequence_name)  # calibration_json 文件的命名规则
        calibration_json = os.path.join(calibration_dir, calibration_key+'.json')
        calibration_json_data = {key: calibration_result[key] for key in calibration_result.keys() if key!='distortions'}
        with open(calibration_json, 'w') as f:
            json.dump(calibration_json_data, f, indent=4)

        # 制作 scene_json
        scene_json = os.path.join(ROOT_PATH, split_name)
        print(scene_json)
        rgb_dir = os.path.join(sequence_dir, 'rgb')  # 序列文件夹下的 rgb 文件夹
        scene_json_data = make_scene_json(scene_json, calibration_json, rgb_dir)
        with open(scene_json, 'w') as f:
            json.dump(scene_json_data, f, indent=4)

    # 处理测试集，同上
    for split_name in validation_splits:
        sequence_name = split_name.split('/')[0]  # 序列名称，例如 000000
        sequence_dir = os.path.join(ROOT_PATH, sequence_name)  # 序列文件夹路径

        # 制作 calibration_json
        calibration_dir = os.path.join(sequence_dir, 'calibration')  # 序列文件夹下的 calibration 文件夹
        os.makedirs(calibration_dir, exist_ok=True)
        calibration_key = 'calib{}'.format(sequence_name)  # calibration_json 文件的命名规则
        calibration_json = os.path.join(calibration_dir, calibration_key+'.json')
        calibration_json_data = {key: calibration_result[key] for key in calibration_result.keys() if key!='distortions'}
        with open(calibration_json, 'w') as f:
            json.dump(calibration_json_data, f, indent=4)

        # 制作 scene_json
        scene_json = os.path.join(ROOT_PATH, split_name)
        print(scene_json)
        rgb_dir = os.path.join(sequence_dir, 'rgb')  # 序列文件夹下的 rgb 文件夹
        scene_json_data = make_scene_json(scene_json, calibration_json, rgb_dir)
        with open(scene_json, 'w') as f:
            json.dump(scene_json_data, f, indent=4)


if __name__ == '__main__':
    main()