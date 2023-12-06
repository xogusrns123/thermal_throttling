#! /bin/bash

docker run -t --rm --name thermal_throttling --gpus all --ipc=host -v ./:/app pytorch_ddp
