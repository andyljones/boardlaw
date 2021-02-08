# Build image
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04 as build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Needed for conda
        curl ca-certificates bzip2 procps \
        # Needed for profiling
        nsight-compute-2020.2.1 nsight-systems-2020.3.4 \
        # Needed for vast work
        openssh-server rsync

# Set up conda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p/opt/conda && \
    rm ~/miniconda.sh 
ENV PATH /opt/conda/bin:/opt/nvidia/nsight-compute/2020.2.1/:/opt/nvidia/nsight-systems/2020.3.4/bin/:$PATH

# Set up SSH server
ADD boardlaw_rsa.pub /root/.ssh/boardlaw_rsa.pub
RUN mkdir -p ~/.ssh && cat ~/.ssh/boardlaw_rsa.pub > ~/.ssh/authorized_keys && \
    echo PermitUserEnvironment yes >> /etc/ssh/sshd_config && \
    echo MaxStartups 100 >> /etc/ssh/sshd_config && \
    echo LoginGraceTime 120 >> /etc/ssh/sshd_config
EXPOSE 22

WORKDIR /code

# Dev image
FROM build AS dev

# Install the Python packages I usually use
RUN apt-get install -y --no-install-recommends \
        # Needed for sanity
        neovim gdb wget man-db tree silversearcher-ag build-essential htop \
        # Needed for git installs
        git ssh-client 

# Grab tini so that Jupyter doesn't spray zombies everywhere
ADD https://github.com/krallin/tini/releases/download/v0.18.0/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Set up git
RUN git config --global user.name "Andrew Jones" && \
    git config --global user.email "andyjones.ed@gmail.com"

# Install the Python packages I usually use
# Install Jupyter 7.5 because 7.6.1 has a bunch of lag with autoreload 
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install  \
        # Core reqs
        numpy tqdm matplotlib beautifulsoup4 psutil ninja pytest b2sdk sympy portalocker fabric patchwork \ 
        # Things I find useful
        pandas jupyter seaborn bokeh setproctitle wurlitzer ipython av flake8 gitpython networkx einops \ 
        # Docs
        sphinx

# Geotorch
RUN pip install git+https://github.com/Lezcano/geotorch/

# Copy the Jupyter config into place. 
ADD .jupyter /root/.jupyter
ADD .ipython /root/.ipython

# Install my backend Jupyter extensions
# aljpy needs to be before noterminal
RUN pip install git+https://github.com/andyljones/aljpy.git && \ 
    pip install git+https://github.com/andyljones/snakeviz@custom && \
    pip install git+https://github.com/andyljones/noterminal && \
    pip install git+https://github.com/andyljones/pytorch_memlab && \
    rm -rf ~/.cache

# Install my frontend Jupyter extensions 
RUN pip install jupyter_contrib_nbextensions && \ 
    jupyter contrib nbextension install --user && \
    cd /root && mkdir nbextensions && cd nbextensions && \
    git clone https://github.com/andyljones/nosearch && \
    cd nosearch && \
    jupyter nbextension install nosearch && \
    jupyter nbextension enable nosearch/main && \
    cd .. && \
    git clone https://github.com/andyljones/noterminal && \
    cd noterminal && \
    jupyter nbextension install noterminal && \
    jupyter nbextension enable noterminal/main && \
    cd .. && \
    git clone https://github.com/andyljones/stripcommon && \
    cd stripcommon && \
    jupyter nbextension install stripcommon && \
    jupyter nbextension enable stripcommon/main && \
    jupyter nbextension enable autoscroll/main 

RUN pip install cmake && \
    apt-get install clang-10 -y --no-install-recommends && \
    ln -s $(which clang-10) /usr/bin/clang++ && \
    cd / && \
    git clone https://github.com/andyljones/open_spiel.git && \
    # TODO: The install.sh bit will fail here after getting the subrepos we're after.
    cd open_spiel && ./install.sh || true && pip install -e .

# Set up MoHex
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/London \
    apt-get install libdb-dev libboost-all-dev -y --no-install-recommends && \
    cd / && \
    git clone https://github.com/cgao3/benzene-vanilla-cmake.git mohex && \
    mkdir mohex/build && \
    cd mohex/build && \
    cmake ../ && \
    make -j4 && \
    ln -s /mohex/build/src/mohex/mohex /bin/mohex

# Something upstream of here breaks Jupyter. 
RUN pip install --upgrade ipython

# Only commit the input cells of notebooks
# https://stackoverflow.com/questions/18734739/using-ipython-notebooks-under-version-control
RUN curl https://raw.githubusercontent.com/toobaz/ipynb_output_filter/master/ipynb_output_filter.py > /bin/ipynb_output_filter.py && \
    echo "*.ipynb   filter=ipynb_output_filter" > /root/.gitattributes && \
    git config --global core.attributesfile /root/.gitattributes && \
    git config --global filter.ipynb_output_filter.clean "python /bin/ipynb_output_filter.py" && \
    git config --global filter.ipynb_output_filter.smudge cat

# Install the vast CLI
RUN cd /usr/bin/ && \
    wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast && \
    chmod +x vast

ADD rsyncd.conf /etc/

# Set up the entrypoint script
ADD dev.sh /usr/bin/

ENTRYPOINT ["tini", "--", "dev.sh"]
