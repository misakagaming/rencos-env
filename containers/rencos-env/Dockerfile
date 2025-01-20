#FROM aaii/cuda:9.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
#FROM nvidia/cuda:12.2.0-devel-ubuntu22.04


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Python Version x.y
ARG PYTHON_VERSION=3.8
ARG PYLUCENE_VERSION=7.4.0

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
		checkinstall \
		software-properties-common \
		dpkg \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 
	
#RUN add-apt-repository ppa:deadsnakes/ppa

#RUN export DEBIAN_FRONTEND=noninteractive && apt-get --purge remove "*cublas*" "cuda*" "nsight*" 

#RUN rm -rf /usr/local/cuda*

#COPY cuda-repo-ubuntu1604_9.2.148-1_amd64.deb .
#COPY 7fa2af80.pub .


#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
#RUN mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN apt-key add 7fa2af80.pub
#RUN dpkg -i cuda-repo-ubuntu1604_9.2.148-1_amd64.deb
#RUN apt-get update
#RUN apt-get -y install cuda --allow-unauthenticated

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        "python$PYTHON_VERSION-dev" \
		"python$PYTHON_VERSION-distutils" \
		python3-setuptools \
        python3-pip \
        "python$PYTHON_VERSION-venv" \
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
RUN ln -sf $(which python$PYTHON_VERSION) /usr/bin/python

#RUN python --version

RUN python -m pip install --upgrade pip

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
    JCC='python -m jcc' \
    NUM_FILES=10 \
    PYTHON=python \
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
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python setup.py build && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python setup.py install && \
    cd .. && \
    make JCC="python -m jcc" ANT=ant PYTHON=python NUM_FILES=8&& \
    make install JCC="python -m jcc" ANT=ant PYTHON=python NUM_FILES=8 && \
    cd ../../


WORKDIR /usr/src
RUN rm -rf pylucene


# ======================== END OF ADDITIONAL INSTALLATION ========================

WORKDIR /app
COPY . .

WORKDIR /



RUN python -m pip install --upgrade pip \
    && python -m venv --system-site-packages /opt/python/venv/base
	
COPY requirements.txt /opt/python/venv/base/
	
RUN /opt/python/venv/base/bin/python -m pip install --no-cache-dir -r /opt/python/venv/base/requirements.txt


ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
	
RUN /usr/python-2.7/bin/python -m pip install virtualenv

RUN virtualenv -p /opt/bin/python2.7 --distribute temp-python


COPY rencos-master .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]