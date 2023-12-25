import os
import csv
import time
from datetime import datetime

# GPU 이름을 얻기 위한 nvidia-smi 명령어 실행
gpu_names = os.popen("nvidia-smi --query-gpu=name,index --format=csv,noheader").read().strip().split('\n')
print(gpu_names)

# 각 GPU에 대해
for gpu_name in gpu_names:
    name = gpu_name.split(',')[0]
    idx = int(gpu_name.split(',')[1])
    # CSV 파일을 'gpu_info_{GPU 이름}.csv'로 생성하거나 열기
    with open(f'/app/data/{name}_idx{idx}.csv', mode='w') as file:
        writer = csv.writer(file)

        # CSV 헤더 작성
        # writer.writerow([str(idx), name]*4)
        writer.writerow(['Time', 'Temperature(°C)', 'Utilization(%)', 'Fan Speed(%)', 'Power Usage(W)', 'Power Max(W)', 'Memory Usage(MiB)', 'Memory Max(MiB)'])


# monitoring 시작!
print(f'start monitoring: {datetime.now()}')
# 24시간 동안 1초 간격으로 GPU 정보를 수집
for i in range(3600):
    
    # 각 GPU에 대해
    for gpu_name in gpu_names:
        name = gpu_name.split(',')[0]
        idx = int(gpu_name.split(',')[1])
        # CSV 파일을 'gpu_info_{GPU 이름}.csv'로 생성하거나 열기
        with open(f'/app/data/{name}_idx{idx}.csv', mode='a') as file:
            writer = csv.writer(file)
            
            # nvidia-smi 명령어를 실행해 GPU 정보를 얻음
            gpu_stats = os.popen(f"nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,fan.speed,power.draw,power.limit,memory.used,memory.total --id={idx} --format=csv,noheader").read().split(',')

            # GPU 정보를 CSV 파일에 쓰기
            writer.writerow([stat.strip() for stat in gpu_stats])
            print(gpu_stats)

    # 1초 대기
    time.sleep(1)
