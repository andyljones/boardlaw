# Build and push image to DockerHub (might have to do this manually)
# Find an appropriate vast machine
# Create a machine using their commandline
# Pass onstart script to start the job
# Use vast, ssh, rsync to monitor things (using fabric and patchwork?)
# Manual destroy
import pandas as pd
import json
from subprocess import check_output
from pathlib import Path

STORAGE = 10

def set_key():
    target = Path('~/.vast_api_key').expanduser()
    if not target.exists():
        key = json.loads(Path('credentials.yml').read_text())['vast']
        target.write_text(key)

def invoke(command):
    set_key()
    return check_output(f'vast {command}', shell=True)

def offers():
    """
    compute_cap:            int       cuda compute capability*100  (ie:  650 for 6.5, 700 for 7.0)
    cpu_cores:              int       # virtual cpus
    cpu_cores_effective:    float     # virtual cpus you get
    cpu_ram:                float     system RAM in gigabytes
    cuda_vers:              float     cuda version
    disk_bw:                float     disk read bandwidth, in MB/s
    disk_space:             float     disk storage space, in GB
    dlperf:                 float     DL-perf score  (see FAQ for explanation)
    dlperf_usd:             float     DL-perf/$
    dph:                    float     $/hour rental cost
    duration:               float     max rental duration in days
    external:               bool      show external offers
    flops_usd:              float     TFLOPs/$
    gpu_mem_bw:             float     GPU memory bandwidth in GB/s
    gpu_ram:                float     GPU RAM in GB
    gpu_frac:               float     Ratio of GPUs in the offer to gpus in the system
    has_avx:                bool      CPU supports AVX instruction set.
    id:                     int       instance unique ID
    inet_down:              float     internet download speed in Mb/s
    inet_down_cost:         float     internet download bandwidth cost in $/GB
    inet_up:                float     internet upload speed in Mb/s
    inet_up_cost:           float     internet upload bandwidth cost in $/GB
    min_bid:                float     current minimum bid price in $/hr for interruptible
    num_gpus:               int       # of GPUs
    pci_gen:                float     PCIE generation
    pcie_bw:                float     PCIE bandwidth (CPU to GPU)
    reliability:            float     machine reliability score (see FAQ for explanation)
    rentable:               bool      is the instance currently rentable
    rented:                 bool      is the instance currently rented
    storage_cost:           float     storage cost in $/GB/month
    total_flops:            float     total TFLOPs from all GPUs
    verified:               bool      is the machine verified
    """
    js = json.loads(invoke(f'search offers --raw --storage {STORAGE}').decode())
    return pd.DataFrame.from_dict(js)

def suggest():
    o = offers()
    viable = o.query('gpu_name == "RTX 2080 Ti" & num_gpus == 1')
    return viable.sort_values('dph_total').iloc[0]

def launch():
    s = suggest()
