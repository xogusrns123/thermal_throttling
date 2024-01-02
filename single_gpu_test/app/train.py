import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
import os
import csv
import traceback
import sys

def train(opts, file_path):
    try:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{str(opts.gpu)}"
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise Exception("Can't not use GPU")
        print('Current Device:', torch.cuda.current_device())
        # ResNet 모델 로드
        model = resnet152()
        model = model.to(device)
        # 손실함수 및 최적화기 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        inputs = torch.randn(opts.batch_size, 3, 224, 224).to(device)  # 큰 배치 크기 사용
        labels = torch.randint(0, 1000, (opts.batch_size,)).long().to(device)
        
        model.train()

        print("start record event")
        # puts a time stamp in the stream of gpu kernel execution
        before_forward_event = torch.cuda.Event(enable_timing=True)
        after_forward_event = torch.cuda.Event(enable_timing=True)
        after_backward_event = torch.cuda.Event(enable_timing=True)
        after_step_event = torch.cuda.Event(enable_timing=True)
        after_zero_grad_event = torch.cuda.Event(enable_timing=True)

        print("start train")

        for i in range(0, opts.max_iter):
            before_forward_event.record()

            outputs = model(inputs)
            after_forward_event.record()

            loss = criterion(outputs, labels)
            loss.backward()
            after_backward_event.record()

            optimizer.step()
            after_step_event.record()

            optimizer.zero_grad()
            after_zero_grad_event.record()
            torch.cuda.synchronize()

            total_latency = before_forward_event.elapsed_time(after_zero_grad_event) / 1000
            forward_time = before_forward_event.elapsed_time(after_forward_event) / 1000
            backward_time = after_forward_event.elapsed_time(after_backward_event) / 1000
            step_time = after_backward_event.elapsed_time(after_step_event) / 1000
            zero_grad_time = after_step_event.elapsed_time(after_zero_grad_event) / 1000

            with open(file_path, 'a') as f:
                writer = csv.writer(f)

                writer.writerow([str(i), total_latency, forward_time, backward_time, step_time, zero_grad_time])

            if i % 10 == 0:
                print(f"training iter:{i}")
                
    except Exception as e:
        print("An eror occured in the train.py")
        print(traceback.format_exc())
        sys.exit(1)