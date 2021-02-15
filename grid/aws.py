
# * Rent an EC2 instance
# * Install docker: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html
# ```
# sudo amazon-linux-extras install docker -y
# sudo service docker start
# sudo usermod -a -G docker ec2-user
# ```
# Log out and in, then `docker info` to verify.
# ```
# docker run andyljones/boardlaw --network host --name boardlaw
# ```
import json
import invoke
from pathlib import Path
from IPython import display
from fabric import Connection
from invoke.exceptions import UnexpectedExit
import jittens

def client():
    import boto3
    creds = json.loads(Path('credentials.json').read_text())['aws']
    return boto3.client('ec2', 'us-east-1', **creds)
    
def instances(id=None):
    if isinstance(id, str):
        return instances()[id]
    if isinstance(id, int):
        i = instances()
        return i[sorted(i)[id]]
    assert id is None
    resp = client().describe_instances(Filters=[{'Name': 'tag:Name', 'Values': ['boardlaw']}])
    states = {}
    for r in resp['Reservations']:
        for i in r['Instances']:
            if i['State']['Name'] in ('pending', 'running',):
                states[i['InstanceId']] = i
    return states

def launch():
    return client().run_instances(
        LaunchTemplate={'LaunchTemplateName': 'boardlaw'},
        MinCount=1,
        MaxCount=1)

def terminate(id=None):
    if id is None:
        return [terminate(id) for id in instances()] 
    return client().terminate_instances(InstanceIds=[id])


_connections = {}
def machine_connection(id=-1):
    if id not in _connections:
        info = instances(id)
        _connections[id] = Connection( 
            host=info['PublicDnsName'], 
            user='ec2-user', 
            port=22, 
            connect_kwargs={
                'allow_agent': False, 
                'look_for_keys': False, 
                'key_filename': ['/root/.ssh/andyljones-useast.pem']})
    return _connections[id]

def wait():
    while True:
        status = {}
        for id in instances():
            try:
                machine_connection(id).run('ls /var/lib/cloud/instance/boot-finished', hide=True)
                status[id] = 'ready'
            except UnexpectedExit:
                status[id] = 'initializing'
                pass

        display.clear_output(wait=True)
        for id, v in status.items():
            print(f'{id:25} {v}')
        
        if all(v in ('ready',) for v in status.values()):
            break

def container_connection(id=-1):
    info = instances(id)
    return Connection( 
        host=info['PublicDnsName'], 
        user='root', 
        # Make sure to add a firewall rule permitting 36022
        port=36022, 
        connect_kwargs={
            'allow_agent': False, 
            'look_for_keys': False, 
            'key_filename': ['/root/.ssh/boardlaw_rsa']})

def container_command(id):
    info = instances(id)
    print(f'SSH_AUTH_SOCK="" ssh root@{info["PublicDnsName"]} -p 36022 -o StrictHostKeyChecking=no -i /root/.ssh/boardlaw_rsa')

def resources(id):
    itype = instances(id)['InstanceType']
    [desc] = client().describe_instance_types(InstanceTypes=[itype])['InstanceTypes']
    return {'cpu': desc['VCpuInfo']['DefaultVCpus'], 'memory': desc['MemoryInfo']['SizeInMiB']/1024}

def jittenate():
    jittens.machines.clear()

    for id, info in instances().items():
        jittens.ssh.add(id,
            resources=resources(id),
            root='/code',
            connection_kwargs={
                'host': info['PublicDnsName'], 
                'user': 'root', 
                'port': 36022, 
                'connect_kwargs': {
                    'allow_agent': False,
                    'look_for_keys': False,
                    'key_filename': ['/root/.ssh/boardlaw_rsa']}})