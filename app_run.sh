#! /bin/bash

torchrun --nproc_per_node=2 main.py --epoch=20 --batch_size=270 --gpu_ids 0 1
