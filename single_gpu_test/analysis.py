import os
import pandas
main_dir = '/home/jovyan/jupyter/thermal_throttling/single_gpu_test/data'
dir_list = os.listdir(main_dir)

for directory in dir_list:
    file_list = os.listdir(os.path.join(main_dir, directory))
    for file in file_list:
        if file.startswith('latency') and file.endswith('.csv'):
            file_path = os.path.join(main_dir, directory, file)
            if file_path