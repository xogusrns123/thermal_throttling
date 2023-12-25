#! /bin/bash

docker run -t --rm --name single_gpu_test --gpus all --ipc=host -v ./:/app pytorch_ddp
