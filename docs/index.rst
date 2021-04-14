#####################################
Scaling Scaling Laws with Board Games
#####################################


Below you can find the code, models and data from our `Scaling Scaling Laws <https://arxiv.org/abs/2104.03113>`_ paper.

Abstract
********

*The largest experiments in machine learning now require resources far beyond the budget of all but a few institutions. 
Fortunately, it has recently been shown that the results of these huge experiments can often be extrapolated from the 
results of a sequence of far smaller, cheaper experiments. In this work, we show that not only can the extrapolation be 
done based on the size of the model, but on the size of the problem as well. By conducting a sequence of experiments 
using AlphaZero and Hex, we show that the performance achievable with a fixed amount of compute degrades predictably 
as the game gets larger and harder. Along with our main result, we further show that increasing the test-time compute 
available to an agent can substitute for reduced train-time compute, and vice versa.*

.. image:: flops_curves.svg
    :alt: A replication of the compute-performance curves
    :width: 640

Code
****
Our code is `on Github <https://github.com/andyljones/boardlaw>`_. After cloning the repo, you can install the dependencies with:: 
    
    pip install -r requirements.txt

We recommend you do this in a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_. Or, better yet, a Docker container. You can find our Dockerfile `here <https://github.com/andyljones/boardlaw/tree/master/docker>`_. 

With the requirements installed and the database (see below) downloaded, you'll be able to reproduce all the plots from the paper using the `paper module <https://github.com/andyljones/boardlaw/blob/master/analysis/paper.py>`_. 

If you want to train your own models, take a look in the `main module <https://github.com/andyljones/boardlaw/blob/master/boardlaw/main.py#L132-L184>`_. 

If you want to evaluate your own models, take a look in the `arena package <https://github.com/andyljones/boardlaw/blob/master/boardlaw/arena/neural.py#L315-L322>`_.

Data 
****
Our data is held in a `SQLite database <https://f002.backblazeb2.com/file/boardlaw/output/experiments/eval/database.sql>`_. Once you've downloaded it, you can query it with::

    import pandas as pd
    pd.read_sql('select * from agents', 'sqlite:///path_to_database.sql')

You can find the schema for the database in `this module <https://github.com/andyljones/boardlaw/blob/master/boardlaw/sql.py#L24-L146>`_, along with 
documentation of the fields and some utility functions for querying it. 

To download the files for a specific training run, the best option is to use backblaze's sync tool. ::

    import b2sdk.v1 as b2
    import sys
    import time 

    run = '2021-03-26 15-30-17 harsh-wait'
    dest = 'local_storage'

    bucket = 'boardlaw'
    api = b2.B2Api()

    syncer = b2.Synchronizer(4)
    with b2.SyncReport(sys.stdout, False) as reporter:
        syncer.sync_folders(
            source_folder=b2.parse_sync_folder(f'b2://boardlaw/output/pavlov/{run}', api),
            dest_folder=b2.parse_sync_folder(f'dest/{run}', api),
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter)

When synced into the ``output/pavlov`` subdirectory, you can load the files using functions from `pavlov <https://github.com/andyljones/boardlaw/tree/master/pavlov>`_, a small 
monitoring library built alongside this project::  

    from pavlov import stats, storage, runs, files

    run = '2021-03-26 15-30-17 harsh-wait'

    # To list the runs you've downloaded 
    runs.pandas()

    # To list the files downloaded for a specific run
    files.pandas(run)

    # To view the residual variance from the run
    stats.pandas(run, 'corr.resid-var')

The state dicts from the snapshots themselves can also be accessed through pavlov, but if you've downloaded the database too then an easier option is ::

    from boardlaw.arena import common
    from boardlaw import analysis

    ag = common.agent(run)
    worlds = common.worlds(run, 1)

    analysis.record(worlds, [ag, ag], n_trajs=1).notebook()

which will play a game between the loaded agents and display it in your notebook. 