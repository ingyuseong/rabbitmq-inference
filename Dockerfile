FROM continuumio/miniconda3:latest

LABEL maintainer="YOUR_NAME"
LABEL version="1.0.0"
LABEL description="A docker image for RabbitMQ script to run starGAN v2 inference"

WORKDIR ./API
COPY ./API .

RUN conda config --add channels conda-forge
RUN conda create --name stargan_mq --file requirements_conda.txt
RUN rm -rf /opt/conda/pkgs/*

ENV PATH /opt/conda/envs/bidi_gan/bin:$PATH

RUN pip install -r requirements_pip.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

CMD ["bash", "-c", "source activate stargan_mq && python rabbitMQ.py"]