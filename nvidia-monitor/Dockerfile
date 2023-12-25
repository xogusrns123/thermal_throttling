# docker build -t nvidia_monitor .

# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

RUN mkdir -p /app/data

# Run the command to start the GPU monitoring script
CMD ["python", "./gpu_monitor.py"]
