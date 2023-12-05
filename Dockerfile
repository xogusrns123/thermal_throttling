# docker build -t pytorch_ddp .

# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# install some libraries
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         cmake \
#         git \
#         curl \
#         ca-certificates \
#         libjpeg-dev \
#         libpng-dev && \
#     rm -rf /var/lib/apt/lists/*

# Copy the local code to the container
COPY . /app

# requirements.txt install
RUN pip install -r requirements.txt

# Set environment variables for distributed training
# ENV NCCL_SOCKET_IFNAME=eth0

# Define the command to run your distributed training script
# Adjust the script name and parameters accordingly
CMD ["torchrun", "--nproc_per_node=2", "main.py", "--epoch=20"]