import sys
import time
from pathlib import Path
import json
import b2sdk.v1 as b2
import aljpy
import multiprocessing

@aljpy.memcache()
def api():
    # Keys are in 1Password
    api = b2.B2Api()
    keys = json.loads(Path('credentials.json').read_text())['backblaze']
    api.authorize_account('production', **keys)
    return api

def sync(source, dest):
    workers = multiprocessing.cpu_count()
    syncer = b2.Synchronizer(workers)
    with b2.SyncReport(sys.stdout, False) as reporter:
        syncer.sync_folders(
            source_folder=b2.parse_sync_folder(source, api()),
            dest_folder=b2.parse_sync_folder(f'b2://boardlaw/{dest}', api()),
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter
        )

def upload(source, dest):
    bucket = api().get_bucket_by_name('boardlaw')
    bucket.upload_local_file(
        local_file=source,
        file_name=dest)

def backup():
    sync('./output/pavlov', 'output/pavlov')

if __name__ == '__main__':
    backup()