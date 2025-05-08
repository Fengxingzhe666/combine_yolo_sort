import argparse
import sys
import numpy as np
import os
import threading  # 导入threading模块用于多线程编程

sys.path.append('D:/code/deep_sort-master')
from zzk_test import gather_sequence_info
from zzk_test import run as run_sort

sys.path.append('D:/code/yolov5-7.0')
from detect import run as run_yolov5
from detect import parse_opt


def txt_to_matrix(txt_file_path, frame_idx, px=640, py=480):
    """
    本函数将txt文本文件中记录的YOLO格式数据转化为MOT格式的npy二维数组
    :param txt_file_path: str
    文本文件的路径
    :param frame_idx: int
    当前帧索引，用于赋值给输出的npy二维数组的第一列
    :param px: int
    图像的长（像素值）
    :param py: int
    图像的宽（像素值）
    :return: npy二维数组
    返回MOT格式数组
    """
    matrix = []  # matrix是一个列表
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_data = line.strip().split(' ')  # 按空格分割每行数据
            line_data = [float(x) for x in line_data]  # 将每个元素转换为浮点数
            matrix.append(line_data)

    matrix_yolo_format = np.array(matrix)
    # 创建一个初始值全为0的138列的二维数组，行数与matrix_yolo_format保持一致
    num_rows = matrix_yolo_format.shape[0]
    matrix_mot_format = np.ones((num_rows, 138))

    # for i in range(0, num_rows):
    #     matrix_mot_format[i, 0] = i + 1

    matrix_mot_format[:, 0] = frame_idx  # 将第一列所有元素赋值为帧索引
    matrix_mot_format[:, 1] = -1  # 将第二列所有元素赋值为-1
    matrix_mot_format[:, 2] = matrix_yolo_format[:, 1] * px - matrix_yolo_format[:, 3] * px / 2
    matrix_mot_format[:, 3] = matrix_yolo_format[:, 2] * py - matrix_yolo_format[:, 4] * py / 2
    matrix_mot_format[:, 4] = matrix_yolo_format[:, 3] * px
    matrix_mot_format[:, 5] = matrix_yolo_format[:, 4] * py
    matrix_mot_format[:, 6] = matrix_yolo_format[:, 5]

    return np.array(matrix_mot_format)

def run_yolov5_in_thread(i, seq_info):
    run_yolov5(
        weights='d:/code/yolov5-7.0/runs/train/custom_model05/weights/best.pt',
        source=seq_info["image_filenames"][i],
        data='d:/code/yolov5-7.0/data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project='d:/code/combine_yolo_sort/yolov5',
        name='exp',
        exist_ok=True,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1
    )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sequence_dir = "D:/code/drone_dataset/zzk01"
    detection_file = "D:/code/drone_dataset/zzk00/result_matrix.npy"

    seq_info = gather_sequence_info(sequence_dir, np.ones((1, 138)), 0)
    threads = []  # 用于存储创建的线程对象
    for i in range(1, seq_info["max_frame_idx"] + 1):
        # 创建线程对象，指定目标函数为run_yolov5_in_thread，并传入相应参数
        thread = threading.Thread(target=run_yolov5_in_thread, args=(i, seq_info))
        thread.start()  # 启动线程
        threads.append(thread)  # 将线程对象添加到列表中

    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()

    detection_npy = None  # 初始化detection_npy
    for i in range(1, seq_info["max_frame_idx"] + 1):
        txt_file_path = f"D:\\code\\combine_yolo_sort\\yolov5\\exp\\labels\\{str(i).zfill(6)}.txt"
        if os.path.exists(txt_file_path):
            detection_npy_last = txt_to_matrix(txt_file_path, i, 1920, 1080)
            if i == 1:
                detection_npy = detection_npy_last.reshape(1, -1)
            else:
                detection_npy = np.vstack((detection_npy, detection_npy_last))

    if i == seq_info["max_frame_idx"]:
        run_sort(
            sequence_dir="D:/code/drone_dataset/zzk01",
            detection_npy=detection_npy,
            output_file="D:/code/combine_yolo_sort/tmp/hypotheses.txt",
            min_confidence=0.3,
            nms_max_overlap=1,
            min_detection_height=0,
            max_cosine_distance=0.2,
            nn_budget=None,
            display=True,
            max_frame_idx=i
        )
