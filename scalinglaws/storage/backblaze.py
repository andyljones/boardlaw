import sys
import time
from pathlib import Path
import json
import b2sdk.v1 as b2
import aljpy

@aljpy.memcache()
def api():
    api = b2.B2Api()
    keys = json.loads(Path('/credentials/backblaze.json').read_text())
    api.authorize_account('production', **keys)
    return api

def sync(source, dest, workers=4):
    syncer = b2.Synchronizer(workers)
    with b2.SyncReport(sys.stdout, False) as reporter:
        syncer.sync_folders(
            source_folder=b2.parse_sync_folder(source, api()),
            dest_folder=b2.parse_sync_folder(f'b2://alj-drones/{dest}', api()),
            now_millis=int(round(time.time() * 1000)),
            reporter=reporter
        )

def sync_traces():
    from .common import compression
    compression.compress_traces() 
    sync('./output/traces', 'output/traces')%     