###############
Developer Guide
###############

So far boardlaw's a `one-man project <https://andyljones.com>`_, but I'm keen to change that. If you think it's an
interesting research direction and you'd like to help out, drop by the `RL <https://discord.gg/xhfNqQv>`_ or `EAI
<https://discord.gg/K8xcydxcka>`_ Discords and give me a ping! I'm usually active London daytime. Slightly more formally,
you can post an issue on the tracker, or `give me an email <me@andyljones.com>`_.

One thing I'd advise against is putting together a PR: this is a research repo and large chunks of it can change 
dramatically and unexpectedly. As open-source software though, you are of course welcome to fork the repo and do
 something entirely different with it! 
 
Prelude
*******
My workflow involves a Jupyter instance running inside a Docker container that sits on a GPU-equipped server. To 
replicate my workflow, you'll need 

* a server with a GPU that `has compute capability 7.5 or above <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ 
  and at least 6GB of memory
* NVIDIA drivers 460.32.03 or above
* both Docker and the `NVIDIA container toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_

To test that you've got all of this right, you should be able to run

.. code::

    docker run pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel python -c "import torch; print(torch.tensor(1).cuda())"

and have it print out ``torch.tensor(1)``.

Setup
*****
First, clone the repo into a local dir.

.. code::

    git clone git@github.com:andyljones/boardlaw.git

Then pull and run the boardlaw image. Run this command, substituting the dir you just cloned the repo into as $CODE_DIR 

.. code::

    docker pull andyljones/boardlaw
    docker run --shm-size="16g" -v CODE_DIR:/code andyljones/boardlaw:latest

For completeness, the 

* ``--shm-size`` ups the shared memory size, as the default 64MB can upset PyTorch's IPC
* ``-v$`` mounts the dir you cloned the repo into as ``/code`` inside the Docker container.

The above is the short version of the command, which will keep the container in the foreground of your terminal 
session so you can easily see if anything goes wrong. Typically I run the container with the much longer command

.. code::

    docker run -d --name boardlaw --shm-size="16g" \
        -v CODE_DIR:/code \
        -p 35022:22 \ 
        --cap-add SYS_ADMIN \
        --cap-add=SYS_PTRACE \
        --security-opt apparmor=unconfined \
        --security-opt seccomp=unconfined \
        andyljones/boardlaw:latest

which additionally has

TODO: Add details

Run
***
Anyway, to run whatever I've got it set to right this second

.. code::

    from boardlaw.main import *
    run()

Once it's declared the run started, you can watch its progress from a second Jupyter instance with

.. code::

    from pavlov import *
    monitor(-1)

for logging and the latest stats and  

.. code::

    from pavlov import *
    stats.review(-1)

for charts.

There's a :github:`Docker image <docker>` if you're having trouble with dependencies.

