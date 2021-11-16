# inspired by https://sourcery.ai/blog/python-docker/
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as base
ARG CUDA_SHORT=112

# Setup locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# no .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# traceback on segfau8t
ENV PYTHONFAULTHANDLER 1

# use ipdb for breakpoints
ENV PYTHONBREAKPOINT=ipdb.set_trace

# common dependencies
RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \
      # primary interpreter
      python3.9 \

      # required by transformers package
      python3.9-distutils \

      # redis-python
      redis \

      # git-state
      git \

      # for Atari Roms and redis
      wget \

 && apt-get clean

FROM base AS python-deps

# build dependencies
RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \

      # required by poetry
      python \
      python3-pip \

      # required for Arari Roms
      unrar \
      unzip \

      # for opencv-python (https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html)
      cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \

 && apt-get clean

WORKDIR "/deps"

COPY pyproject.toml poetry.lock /deps/

RUN pip install poetry \
 && poetry install

RUN wget "http://www.atarimania.com/roms/Roms.rar" \
 && unrar e Roms.rar \
 && unzip ROMS.zip \
 && /root/.cache/pypoetry/virtualenvs/torchbeast-K3BlsyQa-py3.9/bin/python -m atari_py.import_roms ROMS/


FROM base AS runtime

WORKDIR "/project"
ENV VIRTUAL_ENV=/root/.cache/pypoetry/virtualenvs/torchbeast-K3BlsyQa-py3.9/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=python-deps $VIRTUAL_ENV $VIRTUAL_ENV
COPY . .

ENTRYPOINT ["python"]
