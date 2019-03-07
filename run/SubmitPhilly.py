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
         uf = 32, lr = 0.0005, max_lr = 0.0005, warm_updates = 4000, arch = "transformer_wmt_en_de_big", layers = 6, dropout = 0.3, reload_dir = "",
         lr_scheduler = "inverse_sqrt", cosine_period = 40000, extra = "", save_interval_updates = 0):

    ngpus = nprocs * nnodes
    seed = random.randint(1, 5000)
    #seed = 1
    port = random.randint(1000, 9999)
    #port = 1678
    is_cosine = lr_scheduler == "cosine"
    cosine_command = "-MLR {} -CP {}".format(max_lr, cosine_period)

    print('Using seed {} and port {}'.format(seed, port))

    job={
    "ClusterId": cluster,
    "VcId": vc,
    "JobName": name,
    "UserName": user,
    "BuildId": 0,
    "ToolType": None,
    "ConfigFile": "fetia/Src/fairseq_latest/run/train_enfi_transformer_philly_dist_tcp{}.sh".format("_cosine" if is_cosine else ""),
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
    'ExtraParams':"-d {} --dataset {} --warm-update {} -M {} --uf {} -E {} --nodes {} --port {} -s {} --nproc {} "
                  "-A {}  -LR {} -LRS {} -SI 1 --max-update 130000 -SIU {} --enc {} --dec {} -LI {} {} "
                  "{} {}".
        format(dropout, dataset, warm_updates, max_toks, uf, extra, nnodes, port, seed, nprocs,
               arch, lr, lr_scheduler, save_interval_updates, layers, layers, log_interval, "--nccl" if nccl else "",
               "-RD {}".format(reload_dir) if reload_dir != "" else "", cosine_command if is_cosine else ""),
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
    requests.post(url, headers=headers, data=job, auth=HttpNtlmAuth(user, pwd), verify=False)

def submit():

    '''Distributed config'''
    world_size = 2 #number of machines you need
    ngpupernode = 2 #number of gpus you need on each machine
    old_docker = False #better not change. Changing to true will be in-stable. But if you are running 2*4 jobs, it is fairly stable and might even be 15% faster than setting it to False.
    nccl = False #better not change
    vc = "msrmt" #vc you run your jobs
    cluster = "wu2" #cluster you run your jobs

    '''Training config'''
    max_toks = 4096 if vc == "msrmt" else 1536
    #max_toks = 3277
    #uf = 32 if vc == "msrmt" else 86
    uf = 32
    #uf = 20
    lr = 0.0005
    max_lr = 0.0005
    lr_scheduler = "inverse_sqrt"
    cosine_period = 35000
    warm_updates = 4000
    save_updates = 0
    log_interval = 200
    dataset = "wmt19.bt1.tokenized.en-fi.joined"
    arch = "transformer_vaswani_wmt_en_de_big "
    layers = 6
    dropout = 0.3
    reloaddir = "wmt19.bt1.tokenized.en-fi.joined_transformer_vaswani_wmt_en_de_big_dp0.3_seed2305_maxtok4096_lr0.0005_SI1_enc6_dec6_Dist2x4_uf4_1.0"
    #reloaddir = "wmt19.bt1.tokenized.fi-en.joined_transformer_wmt_en_de_big_t2t_dp0.3_seed1792_maxtok2048_uf16_lr0.0005_SI1_enc10_dec10_Dist4x4_1.0"
    #reloaddir = "wmt19.bt1.tokenized.en-fi.joined_transformer_vaswani_wmt_en_de_big_dp0.3_seed2305_maxtok4096_lr0.0005_SI1_enc6_dec6_Dist2x4_uf4_1.0"

    expname = '22beft_1'
    extra = ""

    post(dataset=dataset, vc=vc, cluster=cluster, name = expname, nprocs= ngpupernode, nnodes= world_size, docker_old = old_docker, nccl= nccl,
         log_interval= log_interval, max_toks= max_toks, uf= uf, lr = lr, max_lr = max_lr, lr_scheduler= lr_scheduler, warm_updates= warm_updates,
         arch= arch, layers = layers, dropout= dropout, reload_dir = reloaddir, cosine_period= cosine_period, save_interval_updates= save_updates, extra= extra)


if __name__=='__main__':
    submit()
    print('Submitted.')