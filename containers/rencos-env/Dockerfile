FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
#FROM nvidia/cuda:12.2.0-devel-ubuntu22.04


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Python Version x.y
ARG PYTHON_VERSION=3
ARG PYLUCENE_VERSION=7.4.0

RUN nvcc --version

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
	
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
		python3-distutils \
        python3-dev \
        python3-pip \
        python3-venv \
		python-is-python3 \
		libstdc++6 \
		gcc-multilib\
		libc6 \
		 && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 


RUN python3 --version
#RUN python3.6 -m pip install --upgrade Pillow


# ======================== START OF ADDITIONAL INSTALLATION ========================

# Install Java
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
    build-essential \
    ant \
    curl \
    git \
    default-jdk
    
#RUN ls /usr/bin/ | grep "python"
#RUN ln -sf $(which python$PYTHON_VERSION) /usr/bin/python

#RUN python --version

RUN python3 -m pip install --upgrade pip

#RUN which python$PYTHON_VERSION && which python && python --version

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
RUN curl https://archive.apache.org/dist/lucene/pylucene/pylucene-$PYLUCENE_VERSION-src.tar.gz | tar -xz



ENV PREFIX_PYTHON=/usr \
    JCC_JDK=/usr/lib/jvm/default-java \
    ANT=ant \
    JCC='python3 -m jcc' \
    NUM_FILES=10 \
    PYTHON=python3 \
    NO_SHARED=1
	




RUN mkdir -p "/root/.ant/lib"

WORKDIR /root/.ant/lib

COPY ivy-2.4.0.jar .

WORKDIR /usr/src/pylucene

RUN cd "pylucene-$PYLUCENE_VERSION/lucene-java-$PYLUCENE_VERSION/lucene" && \
    ant ivy-bootstrap && \
    ant && \
    cd ../../../

RUN cd "pylucene-$PYLUCENE_VERSION/jcc" && \
    ls -la && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python3 setup.py build && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python3 setup.py install && \
    cd .. && \
    make JCC="python3 -m jcc" ANT=ant PYTHON=python3 NUM_FILES=8&& \
    make install JCC="python3 -m jcc" ANT=ant PYTHON=python3 NUM_FILES=8 && \
    cd ../../


WORKDIR /usr/src
RUN rm -rf pylucene


# ======================== END OF ADDITIONAL INSTALLATION ========================

WORKDIR /app
COPY . .

WORKDIR /



RUN python3 -m pip install --upgrade pip \
    && python3 -m venv --system-site-packages /opt/python3/venv/base


ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
	
RUN /opt/python3/venv/base/bin/python3 -m pip install --upgrade pip	

COPY requirements.txt /opt/python3/venv/base/


COPY rencos-master .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]