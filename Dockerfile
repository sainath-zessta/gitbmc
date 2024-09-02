# Use an appropriate base image with Python 3.12 and CUDA support for amd64 architecture
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies for OpenCV and other libraries
RUN apt update && apt upgrade -y && \
    apt install -y software-properties-common build-essential libffi-dev libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev \
    libbz2-dev libexpat1-dev liblzma-dev libffi-dev libssl-dev libgl1-mesa-glx libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install -y python3.12 python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 312 && \
    update-alternatives --config python3 && \
    apt install python3.12-distutils -y && \
    apt install wget -y && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    apt-get install -y git

# Set the working directory inside the container
WORKDIR /app
ENV OPENAI_API_KEY=

# Copy your application files into the container
COPY . /app

# Install Python dependencies
RUN pip3 install -r requirements.txt


# Default command to pass to the entry point
CMD ["ENTRYPOINT", "<AppName>", "<IP_JSON>", "<OP_JSON>"]