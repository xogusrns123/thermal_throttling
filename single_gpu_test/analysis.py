import os
import pandas as pd
main_dir = './data'
dir_list = os.listdir(main_dir)

for directory in dir_list:
    file_list = os.listdir(os.path.join(main_dir, directory))
    for file in file_list:
        if file.startswith('latency') and file.endswith('.csv'):
            file_path = os.path.join(main_dir, directory, file)
            df = pd.read_csv(file_path)
            iter = df['iter']
            total_latency = df['total_latency']
            flops = 1 / total_latency
            flops_true_values = flops.iloc[100:1001]
            average_flops = flops_true_values.mean()
            print(average_flops)