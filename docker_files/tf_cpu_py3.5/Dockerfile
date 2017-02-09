#FROM ubuntu:16.04
# This means you derive your docker image from the tensorflow docker image
FROM gcr.io/tensorflow/tensorflow:latest-devel-py3

# note this is needed to be able to install pip3
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev
RUN apt-get install -y vim

# get wget
RUN apt-get install wget

# install python 3.5
RUN add-apt-repository -y ppa:fkrull/deadsnakes
RUN apt-get -y update
RUN apt-get -y install python3.5

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.5 get-pip.py
RUN python3.5 -m pip install -U numpy

#Install some stuff my lib needs
RUN python3.5 -m pip install -U numpy
RUN python3.5 -m pip install -U namespaces
RUN python3.5 -m pip install -U scikit-learn
RUN python3.5 -m pip install -U scipy
RUN python3.5 -m pip install -U pdb
RUN python3.5 -m pip install -U keras

# install my library (only when the a container is spun)
# bash script that configures the running container
ENTRYPOINT ["sh", "/home_simulation_research/hbf_tensorflow_code/docker_files/tf_cpu/container_setup.sh"]

# enviromnet
ENV OMP_NUM_THREADS=2
