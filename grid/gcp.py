# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/compute/api/create_instance.py
from fabric import Connection
from pathlib import Path
import json
import googleapiclient.discovery
import jittens

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

def host_connection(name):
    # gcloud beta compute ssh --zone "us-west1-a" "andyjones_ed@boardlaw-1" --project "andyljones" --container "klt-boardlaw-refine-1-ncbb"
    # Fixing container name doesn't seem possible atm: https://stackoverflow.com/questions/55998957/how-to-specify-the-container-name-running-on-google-compute-engine
    # Fix SSH user name
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

def container_connection(name):
    info = instances()[name]
    external_ip = info['networkInterfaces'][0]['accessConfigs'][0]['natIP']
    return Connection( 
        host=external_ip, 
        user='root', 
        # Make sure to add a firewall rule permitting 36022
        port=36022, 
        connect_kwargs={
            'allow_agent': False, 
            'look_for_keys': False, 
            'key_filename': ['/root/.ssh/boardlaw_rsa']})

def container_command(name):
    info = instances()[name]
    external_ip = info['networkInterfaces'][0]['accessConfigs'][0]['natIP']
    print(f'SSH_AUTH_SOCK="" ssh root@{external_ip} -p 36022 -o StrictHostKeyChecking=no -i /root/.ssh/boardlaw_rsa')

def resources(name):
    machine = instances()[name]['machineType'].split('/')[-1]
    res = api().machineTypes().get(project=PROJECT, zone=ZONE, machineType=machine).execute()
    return {'cpu': res['guestCpus'], 'memory': res['memoryMb']/1024}

def jittenate():
    jittens.machines.clear()

    for name, info in instances().items():
        external_ip = info['networkInterfaces'][0]['accessConfigs'][0]['natIP']
        jittens.ssh.add(name,
            resources=resources(name),
            root='/code',
            connection_kwargs={
                'host': external_ip, 
                'user': 'root', 
                'port': 36022, 
                'connect_kwargs': {
                    'allow_agent': False,
                    'look_for_keys': False,
                    'key_filename': ['/root/.ssh/boardlaw_rsa']}})

