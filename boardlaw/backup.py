import sys
import time
from pathlib import Path
import json
import b2sdk.v1 as b2
import aljpy
import multiprocessing

@aljpy.memcache()
def api(bucket):
    # Keys are in 1Password
    api = b2.B2Api()
    keys = json.loads(Path('credentials.json').read_text())['backblaze'][bucket]
    api.authorize_account('production', **keys)
    return api

def sync(source, dest):
    bucket, path = dest.split(':')
    workers = multiprocessing.cpu_count()
    syncer = b2.Synchronizer(workers)
    with b2.SyncReport(sys.stdout, False) as reporter:
        syncer.sync_folders(
            source_folder=b2.parse_sync_folder(source, api(bucket)),
            dest_folder=b2.parse_sync_folder(f'b2://{bucket}/{path}', api(bucket)),
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter
        )

def upload(source, dest):
    bucket, path = dest.split(':')
    bucket = api(bucket).get_bucket_by_name(dest)
    bucket.upload_local_file(
        local_file=source,
        file_name=path)

def cleanup():
    from tqdm.auto import tqdm
    from pavlov import files, runs, storage
    for run, _ in tqdm(runs.runs().items()):
        for _, fi in storage.snapshots(run).items():
            files.remove(run, fi['path'].name)

def backup():
    cleanup()
    sync('./output/pavlov', 'boardlaw:output/pavlov')
    sync('./output/experiments/architecture/results', 'boardlaw:output/experiments/architecture/results')

if __name__ == '__main__':
    backup()