FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

RUN apt-get update 

RUN apt-get install -y curl

#git required to pip install from git repo's
RUN apt-get install -y git

# fix for setting timezones
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install python3
RUN apt-get install -y python3.9
RUN apt-get install -y python3-distutils

RUN ln -sf /usr/bin/python3.9 /usr/bin/python 

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py

RUN pip install --upgrade pip

# install jax and jaxlib

RUN mkdir -p /home
RUN mkdir -p /home/app
WORKDIR /home/app

# Install ML Packages built with CUDA11 support
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#run requirements in order
#required because scipy needs numpy to already be installed. see https://stackoverflow.com/questions/51399515/docker-cannot-build-scipy.
COPY requirements.txt .

RUN while read module; do pip3 install $module; done < requirements.txt
