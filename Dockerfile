# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any additional dependencies you may need
# For example, if you use torchvision, you can uncomment the following line
# RUN pip install torchvision

# Copy the local code to the container
COPY . .

# Set environment variables for distributed training
ENV NCCL_SOCKET_IFNAME=eth0

# Define the command to run your distributed training script
# Adjust the script name and parameters accordingly
CMD ["python", "your_distributed_training_script.py"]
