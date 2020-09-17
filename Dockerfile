FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	build-essential \
	python3.6 \
	python3.6-dev \
	python3-pip \
	&& apt-get clean

RUN pip3 install --no-cache\
	numpy==1.18.4 \
	Pillow==7.2.0 \
	torch==1.6.0 \
	torchvision==0.7.0 \
	streamlit==0.66.0

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /streamlit_app/
EXPOSE 7500
WORKDIR /streamlit_app/
