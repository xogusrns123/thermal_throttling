#! /bin/bash

torchrun --nproc_per_node=2 main.py --epoch=20 --batch_size=270
