#! /bin/bash

docker run -t --rm --name thermal_throttling --gpus all -v ./:/app pytorch_ddp