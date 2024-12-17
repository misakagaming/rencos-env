FROM ubuntu:focal

# Python Version x.y
ARG PYTHON_VERSION=3.6
ARG PYLUCENE_VERSION=8.11.0

# Uncomment to install specific version of poetry
ENV LANG=C.UTF-8

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
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
        "python$PYTHON_VERSION-dev" \
		"python$PYTHON_VERSION-distutils" \
		python3-setuptools \
        python3-pip \
        python3-venv \
        python-is-python3 \
		libstdc++6 \
		gcc-multilib\
		libc6 &&\
    apt-get remove -y gpg-agent && \
    apt-get purge --auto-remove -y && \
    apt-get clean

RUN python3.6 --version
RUN python3.6 -m pip install --upgrade pip


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
RUN which python$PYTHON_VERSION
RUN python3 --version
#RUN ln -s $(which python$PYTHON_VERSION) /usr/bin/python 



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
    JCC='python3.6 -m jcc' \
    NUM_FILES=10 \
    PYTHON=python3 \
    NO_SHARED=1

RUN cd "pylucene-$PYLUCENE_VERSION/lucene-java-$PYLUCENE_VERSION/lucene" && \
    ant ivy-bootstrap && \
    ant && \
    cd ../../../

RUN cd "pylucene-$PYLUCENE_VERSION/jcc" && \
    ls -la && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python3.6 setup.py build && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python3.6 setup.py install && \
    cd .. && \
    make JCC="python3.6 -m jcc" ANT=ant PYTHON=python3.6 NUM_FILES=8&& \
    make install JCC="python3.6 -m jcc" ANT=ant PYTHON=python3.6 NUM_FILES=8 && \
    cd ../../

RUN apt-get remove -y gpg-agent ant jcc build-essential && \
    apt-get purge --auto-remove -y && \
    apt-get clean

WORKDIR /usr/src
RUN rm -rf pylucene

RUN python3.6 -m venv /opt/python3/venv/base
RUN python3.6 --version


COPY requirements.txt /opt/python3/venv/base/
RUN /opt/python3/venv/base/bin/python3.6 -m pip install --no-cache-dir torch==0.4.1 torchvision==0.2.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN /opt/python3/venv/base/bin/python3.6 -m pip install --no-cache-dir -r /opt/python3/venv/base/requirements.txt

COPY rencos-master .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]