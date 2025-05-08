import cv2
import numpy as np
import math
import os

def overlay_drone(background, drone_img, center_x, center_y):
    """
    将 drone_img 贴到 background 指定中心 (center_x, center_y) 位置上。
    若 drone_img 没有透明通道，则直接覆盖。
    """
    bg_h, bg_w = background.shape[:2]
    dh, dw = drone_img.shape[:2]

    # 计算无人机左上角坐标
    top_left_x = center_x - dw // 2
    top_left_y = center_y - dh // 2

    # 防止越界，裁剪到背景范围
    x1 = max(0, top_left_x)
    y1 = max(0, top_left_y)
    x2 = min(bg_w, top_left_x + dw)
    y2 = min(bg_h, top_left_y + dh)

    # 计算在无人机图片上的有效区域
    drone_x1 = x1 - top_left_x
    drone_y1 = y1 - top_left_y
    drone_x2 = drone_x1 + (x2 - x1)
    drone_y2 = drone_y1 + (y2 - y1)

    if x1 < x2 and y1 < y2:
        background[y1:y2, x1:x2] = drone_img[drone_y1:drone_y2, drone_x1:drone_x2]

def main():
    # 如果脚本和 my_drone01.jpg / my_drone02.jpg / my_drone03.jpg 不在同一目录
    # 请把下面的文件名换成正确的绝对或相对路径
    drone_files = [
        "my_drone01.jpg",
        "my_drone04.jpeg",
        "my_drone03.jpg"
    ]

    # 检查文件是否存在，读取图像
    drone_imgs = []
    for f in drone_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"未找到无人机图片文件：{f}")
        img = cv2.imread(f)
        if img is None:
            raise IOError(f"无法读取图片文件：{f}")
        drone_imgs.append(img)

    # 根据“视差越小 => 距离越远 => 图像越小”的要求，
    # 事先定义三个无人机的参数（缩放倍数、视差偏移、运动轨迹函数）
    # 你可以根据需要自行修改这些参数
    # lower offset => smaller scale => "further" drone
    # higher offset => bigger scale => "closer" drone
    drones_info = [
        {
            "filename": "my_drone01.jpg",
            "scale": 0.25,        # 缩放比例(越小说明图像更小)
            "stereo_offset": 10, # 视差偏移(越小说明距离越远)
            # 轨迹函数：返回 (x, y) 给左目坐标
            "trajectory": lambda t, w, h: (
                int(w * 0.3 + 100 * math.sin(t * 0.02)),
                int(h * 0.4 +  50 * math.sin(t * 0.03))
            )
        },
        {
            "filename": "my_drone04.jpeg",
            "scale": 0.4,
            "stereo_offset": 20,
            "trajectory": lambda t, w, h: (
                int(w * 0.6 + 120 * math.cos(t * 0.03)),
                int(h * 0.3 +  80 * math.sin(t * 0.02))
            )
        },
        {
            "filename": "my_drone03.jpg",
            "scale": 0.6,
            "stereo_offset": 30,
            "trajectory": lambda t, w, h: (
                int(w * 0.5 +  80 * math.cos(t * 0.02)),
                int(h * 0.7 +  40 * math.sin(t * 0.05))
            )
        }
    ]

    # 将实际读取的图像按顺序赋值
    for i, info in enumerate(drones_info):
        # 缩放无人机图片
        scaled = cv2.resize(
            drone_imgs[i],
            None,
            fx=info["scale"],
            fy=info["scale"],
            interpolation=cv2.INTER_AREA
        )
        drones_info[i]["img"] = scaled

    # ---------- 双目视频参数 ----------
    width_single = 760  # 左/右画面宽度
    height_single = 760 # 左/右画面高度
    out_width = width_single * 2   # 1520
    out_height = height_single     # 760

    fps = 30
    duration = 6  # 秒数，可自行调整
    frame_count = fps * duration

    # 定义背景颜色（天空蓝：BGR=(235,206,135))
    sky_blue_bgr = (235, 206, 135)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('synthetic_drone_stereo.mp4', fourcc, fps, (out_width, out_height))

    # ---------- 逐帧生成 ----------
    for f_idx in range(frame_count):
        # 初始化左右画面
        left_frame = np.full((height_single, width_single, 3), sky_blue_bgr, dtype=np.uint8)
        right_frame = np.full((height_single, width_single, 3), sky_blue_bgr, dtype=np.uint8)

        # 依次绘制三架无人机
        for drone_info in drones_info:
            # 计算无人机在左目画面中的位置 (drone_x, drone_y)
            drone_x, drone_y = drone_info["trajectory"](f_idx, width_single, height_single)

            # 贴图到左画面
            overlay_drone(left_frame, drone_info["img"], drone_x, drone_y)

            # 在右画面中加上视差偏移
            offset = drone_info["stereo_offset"]
            overlay_drone(right_frame, drone_info["img"], drone_x + offset, drone_y)

        # 拼接 [Left | Right]
        stereo_frame = np.hstack((left_frame, right_frame))

        out_video.write(stereo_frame)

    out_video.release()
    print("视频已生成：synthetic_drone_stereo.mp4 (分辨率: 1520×760)")

if __name__ == "__main__":
    main()
