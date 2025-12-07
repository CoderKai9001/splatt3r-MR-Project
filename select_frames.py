import os
from natsort import natsorted

def select_frames(path, stride: int):
    file_list = natsorted(os.listdir(path))
    path_list = [f"{path}/{file}" for file in file_list]
    sampled_paths = []
    for i in range(len(path_list)):
        if i % stride == 0:
            sampled_paths.append(path_list[i])
        else:
            continue

    return sampled_paths

