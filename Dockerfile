FROM python:3.7

# Configuration
RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

# Install a few dependencies that are not automatically installed, plus nnU-net
RUN conda create -n hsnet python=3.7
RUN conda activate hsnet

RUN conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
RUN conda install -c conda-forge tensorflow
RUN pip3 install tensorboardX

RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/NTAvanHoeffelen/hsnet.git /home/user/hsnet && \
    git -C /home/user/hsnet checkout umc && \
    chown -R user /home/user/hsnet && \

COPY HSnet_wrapper.py /home/user/hsnet/HSnet_wrapper.py


# Configure entrypoint
USER root

COPY run.sh /root/

# Configure entrypoint
ENTRYPOINT ["/bin/bash","/root/run.sh"]

# docker build "/mnt/netcache/bodyct/temp/fewshot_Niels/hsnet" --no-cache --tag doduo1.umcn.nl/nielsvanhoeffelen/fs_model:1.0
# docker push doduo1.umcn.nl/nielsvanhoeffelen/fs_model:1.0