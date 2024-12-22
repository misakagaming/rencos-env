#FROM ubuntu:focal
quangbd/dc-python:3.6-cuda9.0-cudnn7-ubuntu16.04


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Python Version x.y
ARG PYTHON_VERSION=3.6
ARG PYLUCENE_VERSION=8.11.0


# ADD Python PPA Repository
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get remove -y software-properties-common && \
    apt-get purge --auto-remove -y && \
    apt-get clean

RUN which gpg-agent

# Install Python
RUN apt-get install -y --no-install-recommends \
        "python$PYTHON_VERSION-dev" \
        python3-setuptools \
        python3-pip && \
    apt-get remove -y gpg-agent && \
    apt-get purge --auto-remove -y && \
    apt-get clean

# ======================== START OF ADDITIONAL INSTALLATION ========================

# Install Java
RUN apt-get install -y --no-install-recommends \
    build-essential \
    ant \
    jcc \
    curl \
    git \
    default-jdk 
    
RUN ls /usr/bin/ | grep "python"
RUN ln -s $(which python$PYTHON_VERSION) /usr/bin/python

RUN which python$PYTHON_VERSION && which python && python --version

WORKDIR /usr/lib/jvm/default-java/jre/lib
RUN ln -s ../../lib amd64

# Java 11
RUN java --version && javac --version


# Installing PyLucene
RUN which ant && ant -version

RUN apt-get install -y --no-install-recommends \
    libffi-dev \
    zlib1g-dev

WORKDIR /usr/src/pylucene
RUN curl https://dlcdn.apache.org/lucene/pylucene/pylucene-$PYLUCENE_VERSION-src.tar.gz | tar -xz


ENV PREFIX_PYTHON=/usr \
    JCC_JDK=/usr/lib/jvm/default-java \
    ANT=ant \
    JCC='python -m jcc' \
    NUM_FILES=10 \
    PYTHON=python \
    NO_SHARED=1

RUN cd "pylucene-$PYLUCENE_VERSION/lucene-java-$PYLUCENE_VERSION/lucene" && \
    ant ivy-bootstrap && \
    ant && \
    cd ../../../

RUN cd "pylucene-$PYLUCENE_VERSION/jcc" && \
    ls -la && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python setup.py build && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python setup.py install && \
    cd .. && \
    make JCC="python -m jcc" ANT=ant PYTHON=python NUM_FILES=8&& \
    make install JCC="python -m jcc" ANT=ant PYTHON=python NUM_FILES=8 && \
    cd ../../

RUN apt-get remove -y gpg-agent ant jcc build-essential && \
    apt-get purge --auto-remove -y && \
    apt-get clean

WORKDIR /usr/src
RUN rm -rf pylucene

RUN python -c "import lucene; lucene.initVM()"

# ======================== END OF ADDITIONAL INSTALLATION ========================

WORKDIR /app
COPY . .

WORKDIR /

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
		checkinstall \
		software-properties-common \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 


RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
		python3-distutils \
        python3-venv \
		python-is-python3 \
		libstdc++6 \
		gcc-multilib\
		libc6 \
		 && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 
	

RUN python -m pip install --upgrade pip \
    && python -m venv /opt/python/venv/base
	
#RUN ln -s $(which python$PYTHON_VERSION) /usr/bin/python
	


COPY requirements.txt /opt/python/venv/base/
RUN /opt/python/venv/base/bin/python -m pip install --no-cache-dir -r /opt/python/venv/base/requirements.txt

COPY rencos-master .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]


