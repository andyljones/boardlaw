# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/compute/api/create_instance.py
from fabric import Connection
from pathlib import Path
import json
import googleapiclient.discovery

PROJECT = 'andyljones'
ZONE = 'us-west1-a'

_api = None
def api():
    # `gcloud init`
    # `gcloud auth application-default login`
    global _api
    if _api is None:
        _api = googleapiclient.discovery.build('compute', 'v1')
    return _api

def create(name='boardlaw', template='boardlaw-refine'):
    # Templates: https://console.cloud.google.com/compute/instanceTemplates
    return (api().instances().insert(
        project=PROJECT, 
        zone=ZONE, 
        sourceInstanceTemplate=f'projects/{PROJECT}/global/instanceTemplates/{template}', 
        body={'name': name}).execute())

def instances():
    instances = api().instances().list(project=PROJECT, zone=ZONE).execute().get('items', [])
    return {i['name']: i for i in instances}

def connection(name):
    # gcloud beta compute ssh --zone "us-west1-a" "andyjones_ed@boardlaw-1" --project "andyljones" --container "klt-boardlaw-refine-1-ncbb"
    # Fixing container name doesn't seem possible atm: https://stackoverflow.com/questions/55998957/how-to-specify-the-container-name-running-on-google-compute-engine
    # Fix SSH user name
    pass
    info = instances()[name]
    external_ip = info['networkInterfaces'][0]['accessConfigs'][0]['natIP']
    return Connection( 
        host=external_ip, 
        user='andyjones_ed', 
        port=22, 
        connect_kwargs={
            'allow_agent': False, 
            'look_for_keys': False, 
            'key_filename': ['/root/.ssh/google_compute_engine']})

def run(name, command):
    conn = connection(name)
    container = conn.run('docker ps -aqf "ancestor=andyljones/boardlaw"', hide=True).stdout.strip()
    conn.run(f'docker exec {container} {command}')

# localhost ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDBpbJwqfPafs6Ve8Q9gLvxPUURFKY6bbxLGCWX1VtRLo8DAfzN4BF6OYdwCwGSEmoUB/fUXz+UGnQQkVpcnyMJXP2SJEYPu735/HH6n7AjMIQFX9zEMRcTly7KpT/8sTOckV/Nh0D0ADIiPMbLTXFjCqg6o842TbC3R/FH7aYBFt055INcEBclrIdnSZM2GymDsoG7F2JUsqWNlZs37xnw+PBd/IdF6hdgQnVMWvyCGTjvA1j386hwhJcctgaOk6lkNqMD9hTjuW5BxBGwKrWqgwbXyrFybPYVyrShoXNIEov7oMLaGhK5e99T5KH9ZfxuJXV6edF+kwgJa1FzVtHJqwGKXN06CqhC9+/ItIeRhTqvW/5BDUloUsDFi6l+hzJRzGR0mUndnYCKcSdOT1E/tHQ0mnydP2CM5U2xXQWaq2ZaXxINN5ZqqH1W3E0ozra8muAlAgbXCFb7PMAg44DopSGr6s7UIb957wtuh6fc0pFSfqWRKvxn0Two23+r0hU=