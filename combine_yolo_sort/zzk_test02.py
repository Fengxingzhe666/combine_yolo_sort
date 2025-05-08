import cv2
import os
import time
import zzk_test

# 图片文件夹路径
image_folder = './MOT16/test/MOT16-06'
detection_file='./resources/detections/MOT16_POI_test/MOT16-06.npy'

seq_info = zzk_test.gather_sequence_info(image_folder, detection_file,100) # 这里需要更改detection_file为detection_npy

# 获取文件夹中的所有图片文件
images = [img for img in seq_info["image_filenames"] if img.endswith((".jpg", ".png", ".jpeg"))]
images.sort()  # 按文件名排序

# 创建一个窗口
cv2.namedWindow("Image Slideshow", cv2.WINDOW_NORMAL)
frame_idx = 1
# 遍历所有图片并显示
for image in images:
    img_path = os.path.join(image_folder, image)
    for row in seq_info["detections"]:
        if row[0] == frame_idx and row[6] >= 0.3:
            a=1
    img = cv2.imread(img_path)
    # 添加红色矩形框
    cv2.rectangle(img,)
    cv2.rectangle(img, (100, 100), (150, 150), (0, 0, 255), 2)
    cv2.imshow("Image Slideshow", img)
    cv2.waitKey(33)  # 等待

    # 按下任意键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭窗口
cv2.destroyAllWindows()
