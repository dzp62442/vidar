import os
import json
import shutil
from tqdm import tqdm 
from datetime import datetime
import numpy as np
import cv2 as cv


# 定义基础参数
DATASET = "GreenTractor"  # 数据集名称
ROOT_PATH = "/data/vidar/GreenTractor"  # 数据集根目录
CALIBRATION_PATH = os.path.join(ROOT_PATH, "Calibration", "calibration.json")  # 原始标定结果
CAM_NUMS = 8
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
STOP_NUM = 2000


# 对每个序列进行去畸变
def undistort(sequence_dir, calibration_result):
    rgb_dir = os.path.join(sequence_dir, 'rgb')
    undistort_rgb_dir = os.path.join(sequence_dir, 'undistort_rgb')
    cam_names = calibration_result['names']
    sequence_length = len(os.listdir(os.path.join(rgb_dir, cam_names[0])))  # 序列长度，即图像数目
    
    # 遍历每个相机进行去畸变
    for cam_idx, cam_name in enumerate(cam_names):
        intrinsic = calibration_result['intrinsics'][cam_idx]
        intrinsic = np.array([[intrinsic['fx'], 0., intrinsic['cx']], [0., intrinsic['fy'], intrinsic['cy']], [0., 0., 1.]])
        distortion = np.array(calibration_result['distortions'][cam_idx])
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(intrinsic, distortion, (IMG_WIDTH, IMG_HEIGHT), 0, (IMG_WIDTH, IMG_HEIGHT))
        mapx, mapy = cv.initUndistortRectifyMap(intrinsic, distortion, None, new_camera_matrix, (IMG_WIDTH, IMG_HEIGHT), 5)
        os.makedirs(os.path.join(undistort_rgb_dir, cam_name), exist_ok=True)
        print(sequence_dir, cam_name)
        print('intrinsic:\n', intrinsic)
        print('new_camera_matrix:\n', new_camera_matrix)
        # 遍历每张图像进行去畸变
        for img_idx in tqdm(range(sequence_length)):
            img = cv.imread(os.path.join(rgb_dir, cam_name, str(img_idx)+'.jpg'))
            dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
            cv.imwrite(os.path.join(undistort_rgb_dir, cam_name, str(img_idx)+'.jpg'), dst)
            if img_idx >= STOP_NUM:
                break


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
        undistort(sequence_dir, calibration_result)

    # 处理测试集
    for split_name in validation_splits:
        sequence_name = split_name.split('/')[0]  # 序列名称，例如 000000
        sequence_dir = os.path.join(ROOT_PATH, sequence_name)  # 序列文件夹路径
        undistort(sequence_dir, calibration_result)


if __name__ == '__main__':
    main()