import os
import csv
from datetime import datetime
import time
import traceback
import sys



def launch(file_path, idx):
    try:
        print(f'start monitoring: {datetime.now()}')

        for i in range(1800):
            with open(file_path, mode='a') as f:
                writer = csv.writer(f)
                # nvidia-smi 명령어를 실행해 GPU 정보를 얻음
                gpu_stats = os.popen(f"nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,fan.speed,power.draw,power.limit,memory.used,memory.total --id={idx} --format=csv,noheader").read().split(',')

                # GPU 정보를 CSV 파일에 쓰기
                writer.writerow([stat.strip() for stat in gpu_stats])

            # sleep 1s
            time.sleep(1)
    except Exception as e:
        print("An error occurred in the gpu_monitor.py")
        print(traceback.format_exc())
        sys.exit(1)



