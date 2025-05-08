import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot receive frame (stream end?). Exiting ...")
        break
    # 显示帧
    cv2.imshow('Camera Feed', frame)
    # 按下'q'键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
