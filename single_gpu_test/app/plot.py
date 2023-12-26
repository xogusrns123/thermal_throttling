import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
DataDirectory = '/home/jovyan/jupyter/thermal_throttling/single_gpu_test/data/1080Ti-0'
def get_file_path(dir_path):
    file_list = os.listdir(dir_path)

    for file in file_list:
        if file.startswith('info'):
            info_file_path = os.path.join(dir_path,file)

        if file.startswith('latency'):
            latency_file_path = os.path.join(dir_path, file)

    return info_file_path, latency_file_path

if __name__ == '__main__':
    info_file_path, latency_file_path = get_file_path(DataDirectory)

    df = pd.read_csv(info_file_path)

    # 시간을 datetime 형식으로 변환
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 단위 제거
    df['Power Usage(W)'] = df['Power Usage(W)'].str.replace(' W', '')
    df['Power Max(W)'] = df['Power Max(W)'].str.replace(' W', '')
    df['Memory Usage(MiB)'] = df['Memory Usage(MiB)'].str.replace(' MiB', '')
    df['Memory Max(MiB)'] = df['Memory Max(MiB)'].str.replace(' MiB', '')
    df['Utilization(%)'] = df['Utilization(%)'].str.replace(' %', '')
    
    df['Power Usage(W)'] = pd.to_numeric(df['Power Usage(W)'])
    df['Power Max(W)'] = pd.to_numeric(df['Power Max(W)'])
    df['Memory Usage(MiB)'] = pd.to_numeric(df['Memory Usage(MiB)'])
    df['Memory Max(MiB)'] = pd.to_numeric(df['Memory Max(MiB)'])
    df['Utilization(%)'] = pd.to_numeric(df['Utilization(%)'])
    
    # 전력 사용량과 메모리 사용량을 퍼센트로 변환
    df['Power Usage(%)'] = df['Power Usage(W)'] / df['Power Max(W)'] * 100
    df['Memory Usage(%)'] = df['Memory Usage(MiB)'] / df['Memory Max(MiB)'] * 100

    average = len(df['Time']) / 2
    start = int(average - len(df['Time']) / 3)
    end = int(average + len(df['Time']) / 3)

    # 그래프 그리기
    fig, axs = plt.subplots(4, 2, figsize=(20, 24))
    
    # 온도 그래프
    axs[0,0].plot(df['Time'], df['Temperature(°C)'], color='tab:blue')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Temperature(°C)', color='tab:blue')
    axs[0,0].tick_params(axis='y', labelcolor='tab:blue')
    
    # 전력 사용량 그래프
    axs[0,1].plot(df['Time'], df['Power Usage(%)'], color='tab:red')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Power Usage(%)', color='tab:red')
    axs[0,1].tick_params(axis='y', labelcolor='tab:red')
    
    # 메모리 사용량 그래프
    axs[1,0].plot(df['Time'], df['Memory Usage(%)'], color='tab:purple')
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('Memory Usage(%)', color='tab:purple')
    axs[1,0].tick_params(axis='y', labelcolor='tab:purple')
    
    # Utilization 그래프
    axs[1,1].plot(df['Time'], df['Utilization(%)'], color='tab:green')
    axs[1,1].set_xlabel('Time')
    axs[1,1].set_ylabel('Utilization(%)', color='tab:green')
    axs[1,1].tick_params(axis='y', labelcolor='tab:green')

    # saturation 부분
    # 온도 그래프
    axs[2,0].plot(df['Time'][start:end], df['Temperature(°C)'][start:end], color='tab:blue')
    axs[2,0].set_xlabel('Time')
    axs[2,0].set_ylabel('Temperature(°C)', color='tab:blue')
    axs[2,0].tick_params(axis='y', labelcolor='tab:blue')
    
    # 전력 사용량 그래프
    axs[2,1].plot(df['Time'][start:end], df['Power Usage(%)'][start:end], color='tab:red')
    axs[2,1].set_xlabel('Time')
    axs[2,1].set_ylabel('Power Usage(%)', color='tab:red')
    axs[2,1].tick_params(axis='y', labelcolor='tab:red')
    
    # 메모리 사용량 그래프
    axs[3,0].plot(df['Time'][start:end], df['Memory Usage(%)'][start:end], color='tab:purple')
    axs[3,0].set_xlabel('Time')
    axs[3,0].set_ylabel('Memory Usage(%)', color='tab:purple')
    axs[3,0].tick_params(axis='y', labelcolor='tab:purple')
    
    # Utilization 그래프
    axs[3,1].plot(df['Time'][start:end], df['Utilization(%)'][start:end], color='tab:green')
    axs[3,1].set_xlabel('Time')
    axs[3,1].set_ylabel('Utilization(%)', color='tab:green')
    axs[3,1].tick_params(axis='y', labelcolor='tab:green')
    
    # X축 레이블 겹치지 않게 설정
    axs = axs.flatten()
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45)
        
    fig.tight_layout()
    plt.savefig(os.path.join(DataDirectory, 'gpu_info.png'))

    df = pd.read_csv(latency_file_path)
    iter = df['iter']
    total_latency = df['total_latency']
    forward_time = df['forward_time']
    backward_time = df['backward_time']
    step_time = df['step_time']
    zero_grad_time = df['zero_grad_time']
    flops = 1 / total_latency

    # 그래프 그리기
    fig, axs = plt.subplots(2, 1, figsize=(15, 17))
    # total_latency 그래프
    axs[0].plot(iter[5:], flops[5:], label='1/total_latency')
    axs[1].plot(iter[200:1000], flops[200:1000], label='1/total_latency')

    axs = axs.flatten()
    for ax in axs:
        ax.set_xlabel('iter')
        ax.set_ylabel('1/total_latency')
    
    plt.savefig(os.path.join(DataDirectory, 'latency.png'))
