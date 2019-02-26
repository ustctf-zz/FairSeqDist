import os
import requests
from requests_ntlm import HttpNtlmAuth
import json
import subprocess
import time
import threading
import random

user = 'fetia'


with open(r'C:\Users\fetia\pwd','r') as f:
    pwd=f.read()

hdfs_mapping={'eu1':'gfs',
            'eu2':'gfs',
            'sc1':'gfs',
            'wu2':'gfs',
            'rr1':'hdfs',
            }


def post(vc, name, nprocs, cluster, nnodes, docker_old = False, nccl = False, log_interval = 50, max_toks = 3277, uf = 40, lr = 0.0005, warm_updates = 4000):

    ngpus = nprocs * nnodes

    job={
    "ClusterId": cluster,
    "VcId": vc,
    "JobName": name,
    "UserName": user,
    "BuildId": 0,
    "ToolType": None,
    "ConfigFile": "fetia/Src/fairseq_latest/run/train_enfi_transformer_philly_dist{}.sh".format("_tcp" if not docker_old else ""),
    "Inputs": [
    {
    "Name": "dataDir",
    "Path": "/hdfs/{}/fetia/Src/".format(vc)
    }
    ],
    "Outputs": [],
    "IsDebug": False,
    "RackId": "anyConnected",
    "MinGPUs": ngpus,
    "PrevModelPath": None,
    'ExtraParams':"--dataset wmt19.tokenized.fi-en --warm-update {} -M {} --uf {} -E Dist{}x{}_uf4 --nodes {} --port 1326 --nproc {} -A transformer_wmt_en_de_big  -LR {} -SI 1 --max-update 100000 -SIU 0 --enc 6 --dec 6 -LI {} {}".
        format(warm_updates, max_toks, uf, nnodes, nprocs, nnodes, nprocs, lr, log_interval, "--nccl" if nccl else ""),
    "SubmitCode": "p",
    "IsMemCheck": False,
    "IsCrossRack": False,
    "Registry": "phillyregistry.azurecr.io",
    "Repository": "philly/jobs/custom/pytorch",
    "Tag": "fairseq-0.6.0.0.4.1" if docker_old else "fairseq-0.6.0",
    "OneProcessPerContainer": True,
    "DynamicContainerSize": False,
    "NumOfContainers": nnodes,
    "CustomMPIArgs": 'env OMPI_MCA_BTL=self,sm,tcp,openib',
    "Timeout": None
    }

    job=json.dumps(job)

    url='https://philly/api/v2/submit'
    headers = {'Content-Type':'application/json'}
    requests.post(url, headers=headers, data=job, auth=HttpNtlmAuth(user,pwd), verify=False)

def submit():

    '''Distributed config'''
    world_size = 2 #number of machines you need
    ngpupernode = 4 #number of gpus you need on each machine
    old_docker = False #better not change. Changing to true will be in-stable. But if you are running 2*4 jobs, it is fairly stable and might even be 15% faster than setting it to False.
    nccl = False #better not change
    vc = "msrmt" #vc you run your jobs
    cluster = "wu2" #cluster you run your jobs

    '''Training config'''
    max_toks = 4096 if vc == "msrmt" else 1536
    uf = 32 if vc == "msrmt" else 86
    lr = 0.001
    warm_updates = 8000
    log_interval = 5

    expname = 'Df{}x{}e_1.0_warm8'.format(world_size, ngpupernode)

    post(vc=vc, cluster=cluster, name = expname, nprocs= ngpupernode, nnodes= world_size, docker_old = old_docker, nccl= nccl,
         log_interval= log_interval, max_toks= max_toks, uf= uf, lr = lr, warm_updates= warm_updates)


if __name__=='__main__':
    submit()
    print('Submitted.')