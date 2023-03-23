FROM doduo1.umcn.nl/uokbaseimage/base:tf2.10-pt1.12

# Configuration
RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

# Install miniconda
RUN wget --progress=dot:mega https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
        && chmod +x miniconda.sh && ./miniconda.sh -b -p /home/user/conda \
        && rm -f miniconda.sh

# add conda to the path
ENV PATH /home/user/conda/bin:$PATH

# Update configure files
RUN echo 'export PATH=/home/user/conda/bin:$PATH' >> /etc/profile.d/pynn.sh \
        && ln -sf /home/user/conda/etc/profile.d/conda.sh /etc/profile.d/

# Get HSnet git
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/NTAvanHoeffelen/hsnet.git /home/user/hsnet && \
    git -C /home/user/hsnet checkout umc && \
    chown -R user /home/user/hsnet

RUN conda env create -f /home/user/hsnet/hsnet.yaml
RUN echo "source activate hsnet" > ~/.bashrc
ENV PATH /opt/conda/envs/hsnet/bin:$PATH

#USER root

#COPY run.sh /root/

# Configure entrypoint
ENTRYPOINT ["/bin/bash","/home/user/hsnet/run.sh"]

# docker build "/mnt/netcache/bodyct/temp/fewshot_Niels/hsnet" --no-cache --tag doduo1.umcn.nl/nielsvanhoeffelen/fs_model:1.0
# docker push doduo1.umcn.nl/nielsvanhoeffelen/fs_model:1.0