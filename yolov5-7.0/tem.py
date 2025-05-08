import numpy as np
import shutil
import os

def generate_lines():
    base_line = "0 0.288542 0.528704 0.18125 0.205556 0.88721"
    lines = []
    for i in range(100):
        parts = base_line.split()
        parts[1] = "{:.6f}".format(float(parts[1]) + 0.005 * i)
        parts[2] = "{:.6f}".format(float(parts[2]) + 0.003 * i)
        new_line = " ".join(parts)
        lines.append(new_line)
    return lines


# result = generate_lines()
# for line in result:
#     print(line)


def copy_files():
    source_file = r"D:\code\drone_dataset\zzk00\000001.png"
    folder = os.path.dirname(source_file)
    for i in range(2, 100):
        new_file_name = os.path.join(folder, str(i).zfill(6) + ".png")
        shutil.copy(source_file, new_file_name)


if __name__ == "__main__":
    # copy_files()
    detection_file = 'result_matrix.npy'
    detections = np.load(detection_file)
    a = 1
