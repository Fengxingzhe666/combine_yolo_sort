from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

import sys
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

def gather_sequence_info(sequence_dir, detection_npy, max_frame_idx=0):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_npy : npy
        Replace the Path to the detection file to a npy array directly.
    max_frame_idx : int,optional
        用于限制image_filenames只包含前max_frame_idx项的最大帧数参数，默认值为0，表示不设置限制，直接读取全部内容
    """
    image_dir = os.path.join(sequence_dir, "img1")
    all_image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if max_frame_idx == 0:
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}
    else:
        # 对获取到的所有图像文件名字典的键（帧数）进行排序，方便后续截取前max_frame_idx项
        sorted_frame_indices = sorted(all_image_filenames.keys())
        # 根据max_frame_idx截取前max_frame_idx项的帧数
        selected_frame_indices = sorted_frame_indices[:max_frame_idx]

        # 重新构建只包含前max_frame_idx项的image_filenames字典
        image_filenames = {
            idx: all_image_filenames[idx] for idx in selected_frame_indices}

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detection_npy[:, 0].min())
        max_frame_idx = int(detection_npy[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detection_npy.shape[1] - 10 if detection_npy is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detection_npy,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def run(sequence_dir, detection_npy, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, max_frame_idx):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_npy : npy
        Replace the Path to the detection file to a npy array directly.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_npy, max_frame_idx)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    global detection_mat
    detection_mat = np.empty((0, 138))  # Initialize as an empty 2D array with 0 rows and 138 columns
    # detection_mat = np.empty((0,))

    def create_detections(frame_idx, min_height=0):
        """Create detections for given frame index from the raw detection matrix.

        Parameters
        ----------
        frame_idx : int
            The frame index.
        min_height : Optional[int]
            A minimum detection bounding box height. Detections that are smaller
            than this value are disregarded.

        Returns
        -------
        List[tracker.Detection]
            Returns detection responses at given frame index.

        """

        run_yolov5(
            weights='d:/code/yolov5-7.0/runs/train/custom_model05/weights/best.pt',
            source=seq_info["image_filenames"][frame_idx],
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
        txt_file_path = f"D:\\code\\combine_yolo_sort\\yolov5\\exp\\labels\\{str(frame_idx).zfill(6)}.txt"
        if os.path.exists(txt_file_path):
            # detection_npy_last表示该次检测
            detection_npy_last = txt_to_matrix(txt_file_path, frame_idx,seq_info["image_size"][1],seq_info["image_size"][0])
            # 将detection_npy_last添加到detection_npy的末行
            global detection_mat
            detection_mat = np.vstack((detection_mat, detection_npy_last))

        frame_indices = detection_mat[:, 0].astype(np.int)
        mask = frame_indices == frame_idx

        detection_list = []  # 初始化一个列表，用于存储检测列表
        for row in detection_mat[mask]:
            bbox, confidence, feature = row[2:6], row[6], row[10:]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))  # 这一行是检测的核心代码
        return detection_list

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]  # 过滤掉置信度小于0.3的检测目标

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            # vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=100)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)


if __name__ == '__main__':
    # sequence_dir = "./MOT16/test/MOT16-06"
    # detection_file = "./resources/detections/MOT16_POI_test/MOT16-06.npy"
    # detection_file = "D:/code/drone_dataset/zzk00/result_matrix.npy"
    run(
        sequence_dir="D:/code/drone_dataset/multiple_drone_track01",
        detection_npy=np.load("D:/code/drone_dataset/zzk00/result_matrix.npy"),
        output_file="./tmp/hypotheses.txt",
        min_confidence=0.3,
        nms_max_overlap=1,
        min_detection_height=0,
        max_cosine_distance=0.2,
        nn_budget=None,
        display=True,
        max_frame_idx=0
    )