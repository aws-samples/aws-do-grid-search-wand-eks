ARG BASE_IMAGE=pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
FROM $BASE_IMAGE

LABEL description="Base container for Torch Elastic using transformers"

RUN apt-get update && apt-get install -y htop vim wget curl software-properties-common debconf-utils python3-distutils dnsutils bc

# Install python3.9
RUN DEBIAN_FRONTEND=noninteractive; add-apt-repository -y ppa:deadsnakes/ppa; apt install -y python3.9; update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py; python get-pip.py; rm -f get-pip.py

# install utilities and dependencies
RUN pip install awscli --upgrade

RUN pip uninstall -y torch
RUN pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

# Install other python libraries
RUN pip install datasets transformers configparser pathlib sklearn pandas wandb pyyaml

WORKDIR /workspace

COPY ./examples /workspace/examples
RUN chmod -R u+x /workspace/examples
ENV PATH=/workspace/examples/:${PATH}

USER root
ENTRYPOINT ["python", "-m", "torch.distributed.run"]
