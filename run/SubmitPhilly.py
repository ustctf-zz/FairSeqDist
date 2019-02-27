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


def post(dataset, vc, name, nprocs, cluster, nnodes, docker_old = False, nccl = False, log_interval = 50, max_toks = 4096,
         uf = 32, lr = 0.0005, warm_updates = 4000, arch = "transformer_wmt_en_de_big", layers = 6):

    ngpus = nprocs * nnodes
    seed = random.randint(1, 2000)
    port = random.randint(1000, 9999)
    print('Using seed {} and port {}'.format(seed, port))

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
    'ExtraParams':"--dataset {} --warm-update {} -M {} --uf {} -E Dist{}x{} --nodes {} --port {} -s {} --nproc {} "
                  "-A {}  -LR {} -SI 1 --max-update 100000 -SIU 0 --enc {} --dec {} -LI {} {}".
        format(dataset, warm_updates, max_toks, uf, nnodes, nprocs, nnodes, port, seed, nprocs, arch, lr, layers, layers, log_interval, "--nccl" if nccl else ""),
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
    world_size = 4 #number of machines you need
    ngpupernode = 4 #number of gpus you need on each machine
    old_docker = False #better not change. Changing to true will be in-stable. But if you are running 2*4 jobs, it is fairly stable and might even be 15% faster than setting it to False.
    nccl = False #better not change
    vc = "msrmt" #vc you run your jobs
    cluster = "wu2" #cluster you run your jobs

    '''Training config'''
    max_toks = 4096 if vc == "msrmt" else 1536
    uf = 32 if vc == "msrmt" else 86
    lr = 0.0005
    warm_updates = 4000
    log_interval = 50
    dataset = "wmt19.bt1.tokenized.fi-en.joined"
    arch = "Transformer_wmt_en_de_t2t"
    layers = 10

    expname = 'beft_20_lb'.format(world_size, ngpupernode)

    post(dataset=dataset, vc=vc, cluster=cluster, name = expname, nprocs= ngpupernode, nnodes= world_size, docker_old = old_docker, nccl= nccl,
         log_interval= log_interval, max_toks= max_toks, uf= uf, lr = lr, warm_updates= warm_updates, arch= arch, layers = layers)


if __name__=='__main__':
    submit()
    print('Submitted.')