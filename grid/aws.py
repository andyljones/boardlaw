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
from pathlib import Path
import boto3
from IPython import display

def client():
    creds = json.loads(Path('credentials.json').read_text())['aws']
    return boto3.client('ec2', 'us-east-1', **creds)

def states():
    resp = client().describe_instances(Filters=[{'Name': 'tag:Name', 'Values': ['boardlaw']}])
    states = {}
    for r in resp['Reservations']:
        for i in r['Instances']:
            states[i['InstanceId']] = i['State']['Name']
    return states

def launch():
    return client().run_instances(
        LaunchTemplate={'LaunchTemplateName': 'boardlaw'},
        MinCount=1,
        MaxCount=1)

def wait():
    while True:
        s = states()
        display.clear_output(wait=True)
        for k, v in s.items():
            print(f'{k:15s}    {v}')
        if all(v in ('running',) for v in s.values()):
            break

    

