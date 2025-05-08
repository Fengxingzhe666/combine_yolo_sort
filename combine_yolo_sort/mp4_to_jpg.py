import cv2
import os

# 视频文件路径
video_path = r'D:\学习\20250107周汇报会\多个无人机跟随02.mp4'
# 保存图片的目标文件夹路径
output_folder = r'D:\code\drone_dataset\multiple_drone_track02\img1'

# 检查保存图片的文件夹是否存在，不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 帧计数器，用于给保存的图片命名
frame_count = 0
# 实际保存的图片计数器
save_count = 1

while True:
    # 读取视频的每一帧，ret为是否成功读取帧的标志，frame是读取到的帧数据（以numpy数组形式）
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # 每隔几帧保存一次
    if frame_count % 1 == 0:
        # 构建保存图片的文件名，使用zfill(6)来确保文件名是6位数字，前面用0填充
        frame_name = str(save_count).zfill(6) + '.jpg'
        # 构建保存图片的完整路径
        frame_path = os.path.join(output_folder, frame_name)
        # 将当前帧保存为图片
        cv2.imwrite(frame_path, frame)
        save_count += 1

# 释放视频文件资源
cap.release()
