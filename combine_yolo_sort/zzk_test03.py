import time
import deep_sort_app
from deep_sort_app import run


if __name__ == '__main__':
    start_time = time.time()  # 记录程序开始时间
    # sequence_dir = "./MOT16/test/MOT16-06"
    # detection_file = "./resources/detections/MOT16_POI_test/MOT16-06.npy"
    run(
        sequence_dir="D:/code/drone_dataset/zzk00",
        detection_file="D:/code/drone_dataset/zzk00/result_matrix.npy",
        output_file="./tmp/hypotheses.txt",
        min_confidence=0.3,
        nms_max_overlap=1,
        min_detection_height=0,
        max_cosine_distance=0.2,
        nn_budget=None,
        display=False
    )
    end_time = time.time()  # 记录程序结束时间
    elapsed_time = end_time - start_time  # 计算运行所用时间
    print(f"程序运行所用时间为: {elapsed_time} 秒")
