import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
import os
import csv
import argparse
import multiprocessing
from train import train
from gpu_monitor import launch
import time
from plot import draw

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max_iter', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_label', type=str, default='1080Ti-0')

    return parser

def get_filepath(gpu_idx, gpu_label):
    gpu_names = os.popen("nvidia-smi --query-gpu=name,index --format=csv,noheader").read().strip().split('\n')

    for gpu_name in gpu_names:
        name = gpu_name.split(',')[0]
        idx = int(gpu_name.split(',')[1])

        if idx == gpu_idx:
            latency_file_name = f"latency_{name}_idx{str(gpu_idx)}.csv"
            info_file_name = f"info_{name}_idx{str(gpu_idx)}.csv"
            break

    dir_name = f"/app/data/{gpu_label}"
    # 디렉토리가 존재하지 않는 경우 디렉토리를 생성합니다.
    os.makedirs(dir_name, exist_ok=True)

    latency_file_path = os.path.join(dir_name, latency_file_name)
    with open(latency_file_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'total_latency', 'forward_time', 'backward_time', 'step_time', 'zero_grad_time'])

    info_file_path = os.path.join(dir_name, info_file_name)
    with open(info_file_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Temperature(°C)', 'Utilization(%)', 'Fan Speed(%)', 'Power Usage(W)', 'Power Max(W)', 'Memory Usage(MiB)', 'Memory Max(MiB)'])
    
    return latency_file_path, info_file_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser('resnet152 training', parents=[get_args_parser()])
    opts = parser.parse_args()

    latency_file_path, info_file_path = get_filepath(opts.gpu, opts.gpu_label)

    pcs1 = multiprocessing.Process(target=launch, args=(info_file_path, opts.gpu,))
    pcs2 = multiprocessing.Process(target=train, args=(opts, latency_file_path,))

    print("start process 1 gpu_monitoring")
    pcs1.start()

    time.sleep(1)

    print("start process 2 training")
    pcs2.start()

    # wait until pcs2 terminated
    print("wait until process 2 terminated")
    pcs2.join()

    # if pcs terminated, terminate pcs1
    if pcs1.is_alive():
        print("Terminating process 1...")
        pcs1.terminate()
        pcs1.join()

    print("All process terminated")
    print("run plot.py")
    datadirectory = f'/app/data/{opts.gpu_label}'
    draw(datadirectory)