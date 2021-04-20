import sys
import time
import numpy as np
from pathlib import Path
import json
import b2sdk.v1 as b2
import aljpy
import multiprocessing
from pavlov import runs, files, storage
from tqdm.auto import tqdm
from logging import getLogger

log = getLogger(__name__)

BUCKET = 'boardlaw'

OPEN_CREDENTIALS = dict(
    application_key_id='0025bc00838182c0000000006',
    application_key='K0023q+mw4bVM5Qx/6keRDn8MMv6J/U')

@aljpy.memcache()
def api(bucket):
    # Keys are in 1Password
    api = b2.B2Api()
    path = Path('credentials.json')
    if path.exists():
        keys = json.loads(path.read_text())['backblaze'][bucket]
    else:
        keys = OPEN_CREDENTIALS
        log.warn('No credentials file found; using public Backblaze access')
    api.authorize_account('production', **keys)
    return api

def sync_up(local, remote):
    bucket, path = remote.split(':')
    workers = multiprocessing.cpu_count()
    syncer = b2.Synchronizer(workers)
    with b2.SyncReport(sys.stdout, False) as reporter:
        syncer.sync_folders(
            source_folder=b2.parse_sync_folder(local, api(bucket)),
            dest_folder=b2.parse_sync_folder(f'b2://{bucket}/{path}', api(bucket)),
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter)

def sync_down(local, remote, **kwargs):
    bucket, path = str(remote).split(':')
    workers = multiprocessing.cpu_count()
    syncer = b2.Synchronizer(workers, **kwargs)
    with b2.SyncReport(sys.stdout, False) as reporter:
        syncer.sync_folders(
            source_folder=b2.parse_sync_folder(f'b2://{bucket}/{path}', api(bucket)),
            dest_folder=b2.parse_sync_folder(str(local), api(bucket)),
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter)

def upload(local, remote):
    bucket, path = remote.split(':')
    bucket = api(bucket).get_bucket_by_name(remote)
    bucket.upload_local_file(
        local_file=local,
        file_name=path)

def ablate_run_snapshots(run):
    for i, info in tqdm(storage.snapshots(run).items()):
        if (i == 0) or (np.log2(i) % 1 == 0):
            pass
        else:
            print('Removing', run, info['path'].name)
            files.remove(run, info['path'].name)

def ablate_snapshots():
    for run, info in runs.runs().items():
        ablate_run_snapshots(run)

def download(local, remote):
    local = str(local)
    remote = str(remote)
    Path(local).parent.mkdir(exist_ok=True, parents=True)

    bucket = api(BUCKET).get_bucket_by_name(BUCKET)
    dest = b2.DownloadDestLocalFile(local)
    return bucket.download_file_by_name(remote, dest)

def download_run_info(run):
    local = runs.infopath(run, res=False)
    if local.exists():
        log.info(f'Run info for "{run}" already exists')
        return
    remote = Path('output/pavlov') / local.relative_to(runs.ROOT)
    download(local, remote)

def download_agent(run, idx):
    download_run_info(run)
    filename = storage.SNAPSHOT.format(n=idx)
    local_path = f'{runs.ROOT}/{run}/{filename}'
    if Path(local_path).exists():
        log.info(f'Snapshot "{run}" #{idx} already exists')
    remote_path = f'output/pavlov/{run}/{filename}'
    download(local_path, remote_path)

def backup():
    sync_up('./output/pavlov', f'{BUCKET}:output/pavlov')
    sync_up('./output/experiments/eval', f'{BUCKET}:output/experiments/eval')
    sync_up('./output/experiments/bee', f'{BUCKET}:output/experiments/bee')
    sync_up('./output/experiments/architecture/results', f'{BUCKET}:output/experiments/architecture/results')

def fetch():
    sync_down('./output/pavlov', 'boardlaw:output/pavlov')

if __name__ == '__main__':
    backup()