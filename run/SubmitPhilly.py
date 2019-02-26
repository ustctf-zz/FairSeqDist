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


def post(vc, name, nprocs, cluster, nnodes, docker_old = False, nccl = False, log_interval = 50, max_toks = 3277, uf = 40):

    ngpus = nprocs * nnodes
    lr = 0.0005 * ngpus / 4
    lr = 0.0005

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
    'ExtraParams':"--dataset wmt19.tokenized.fi-en -M {} --uf {} -E Dist{}x{}_uf4 --nodes {} --port 1678 --nproc {} -A transformer_wmt_en_de_big  -LR {} -SI 1 --max-update 100000 -SIU 0 --enc 6 --dec 6 -LI {} {}".
        format(max_toks, uf, nnodes, nprocs, nnodes, nprocs, lr, log_interval, "--nccl" if nccl else ""),
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


    # upload_code(main_script, cluster, env)
    # upload_code('utils.py', cluster, env)
    # upload_code('-r models', cluster, env)

    world_size = 2
    ngpupernode = 4
    old_docker = False
    nccl = False
    log_interval = 5
    vc = "msrmt"
    cluster = "wu2"

    max_toks = 4096 if vc == "msrmt" else 1536
    uf = 32 if vc == "msrmt" else 86

    uf = 4
    expname = 'Vf{}x{}e_1.0_0.25uf'.format(world_size, ngpupernode)

    post(vc=vc, cluster=cluster, name = expname, nprocs= ngpupernode, nnodes= world_size, docker_old = old_docker, nccl= nccl, log_interval= log_interval, max_toks= max_toks, uf= uf)


if __name__=='__main__':
    submit()
    print('Submitted.')