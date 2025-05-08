import argparse
import sys
import numpy as np
import os

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sequence_dir = "D:/code/drone_dataset/zzk01"
    detection_file = "D:/code/drone_dataset/zzk00/result_matrix.npy"
    # output_file = "./tmp/hypotheses.txt"
    # min_confidence = 0.3
    # nms_max_overlap = 1
    # min_detection_height = 0
    # max_cosine_distance = 0.2
    # nn_budget = None
    # display = True

    # 初始化一个1行138列全为0的二维矩阵
    # detection_npy = np.zeros((1, 138))

    seq_info = gather_sequence_info(sequence_dir, np.ones((1,138)), 0)
    for i in range(1, seq_info["max_frame_idx"]+1):
        run_yolov5(
            weights='d:/code/yolov5-7.0/runs/train/custom_model05/weights/best.pt',
            source=seq_info["image_filenames"][i],
            data='d:/code/yolov5-7.0/data/coco128.yaml',
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=True,  # save results to *.txt
            save_conf=True,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='d:/code/combine_yolo_sort/yolov5',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=True,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
        )
        # str(i)将整数/浮点数转化为字符串，zfill(6)为字符串定义长度，不足部分补0
        txt_file_path = f"D:\\code\\combine_yolo_sort\\yolov5\\exp\\labels\\{str(i).zfill(6)}.txt"
        if os.path.exists(txt_file_path):
            # detection_npy_last表示该次检测
            detection_npy_last = txt_to_matrix(txt_file_path, i,1920,1080)
            if i == 1:
                # 当第一次循环时，将获取到的detection_npy_last作为第一行数据赋值给detection_npy
                detection_npy = detection_npy_last.reshape(1, -1)
            else:
                # 将detection_npy_last添加到detection_npy的末行（非第一次循环时执行此操作）
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
