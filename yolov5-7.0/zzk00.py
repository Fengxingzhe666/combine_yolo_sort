import numpy as np


def txt_to_matrix(txt_file_path, px=640, py=480):
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

    for i in range(0, 100):
        matrix_mot_format[i, 0] = i + 1

    # matrix_mot_format[:, 0] = 1  # 将第一列所有元素赋值为1
    matrix_mot_format[:, 1] = -1  # 将第二列所有元素赋值为-1
    matrix_mot_format[:, 2] = matrix_yolo_format[:, 1] * px - matrix_yolo_format[:, 3] * px / 2
    matrix_mot_format[:, 3] = matrix_yolo_format[:, 2] * py - matrix_yolo_format[:, 4] * py / 2
    matrix_mot_format[:, 4] = matrix_yolo_format[:, 3] * px
    matrix_mot_format[:, 5] = matrix_yolo_format[:, 4] * py
    matrix_mot_format[:, 6] = matrix_yolo_format[:, 5]

    return np.array(matrix_mot_format)


if __name__ == '__main__':
    path = r"D:/code/yolov5-7.0/runs/detect/exp39/labels/屏幕截图(8).txt"
    result_matrix = txt_to_matrix(path, 1920, 1080)
    np.save("result_matrix.npy", result_matrix)
