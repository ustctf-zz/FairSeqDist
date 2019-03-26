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

vc_maps={'eu1':'nextmsra',
            'eu2':'gfs',
            'sc3':'resrchprojvc3',
            'wu2':'msrmt',
            'rr1':'sdrgvc',
            }

def post(dataset, vc, name, nprocs, cluster, nnodes, docker_old = False, nccl = False, log_interval = 50, max_toks = 4096,
         uf = 32, lr = 0.0005, max_lr = 0.0005, warm_updates = 4000, arch = "transformer_wmt_en_de_big", layers = 6, dropout = 0.3, reload_dir = "",
         lr_scheduler = "inverse_sqrt", cosine_period = 40000, extra = "", save_interval_updates = 0, src = "en", tgt = "fi", r2l = False, c10d = False,
         blob = False):

    ngpus = nprocs * nnodes
    seed = random.randint(500, 9999)
    #seed = 9337
    port = random.randint(1000, 9999)
    #port = 1678
    is_cosine = lr_scheduler == "cosine"
    cosine_command = "-MLR {} -CP {}".format(max_lr, cosine_period)

    print('Using seed {} and port {}'.format(seed, port))

    def getConfigFile():
        blob_ = "/blob/" if blob else ""
        prefix = "fetia/Src/{}/run/".format("fairseq_latest" if cluster == "wu2" else "FairSeqDist")
        #config_name = "train_enfi_transformer_philly_{}.sh".format("dist_tcp" if nnodes > 1 else "single")
        config_name = "train_enfi_transformer_philly_dist_tcp.sh"
        return blob_ + os.path.join(prefix, config_name)

    def CustomMPIArgs():
        return ""

    job={
    "ClusterId": cluster,
    "VcId": vc,
    "JobName": name,
    "UserName": user,
    "BuildId": 0,
    "ToolType": None,
    "ConfigFile": getConfigFile(),
    "Inputs": [
    {
    "Name": "dataDir",
    "Path": "/hdfs/{}/fetia/Src/".format(vc) if not blob else "/blob/fetia/Src/"
    }
    ],
    "Outputs": [],
    "IsDebug": False,
    "RackId": "anyConnected",
    "MinGPUs": ngpus,
    "PrevModelPath": None,
    'ExtraParams':"-d {} --dataset {} --warm-update {} -M {} --uf {} -E {} --nodes {} --port {} -s {} --nproc {} "
                  "-A {}  -LR {} -LRS {} -SI 1 --max-update 300000 -SIU {} --enc {} --dec {} -LI {} {} "
                  "{} {} --src {} --tgt {} {} -UC "
                  "{} {}".
        format(dropout, dataset, warm_updates, max_toks, uf, extra, nnodes, port, seed, nprocs,
               arch, lr, lr_scheduler, save_interval_updates, layers, layers, log_interval, "--nccl" if nccl else "",
               "-RD {}".format(reload_dir) if reload_dir != "" else "", cosine_command if is_cosine else "", src, tgt, "--r2l" if r2l else "",
               "--c10d" if c10d else "", "-BLOB" if blob else ""),
    "SubmitCode": "p",
    "IsMemCheck": False,
    "IsCrossRack": False,
    "Registry": "phillyregistry.azurecr.io",
    "Repository": "philly/jobs/custom/pytorch",
    "Tag": "fairseq-0.6.0.0.4.1" if docker_old else "fairseq-0.6.0",
    "OneProcessPerContainer": True,
    "DynamicContainerSize": False,
    "NumOfContainers": nnodes,
    #"CustomMPIArgs": 'env NCCL_DEBUG=INFO NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=^docker0',
    #"CustomMPIArgs":  'env NCCL_DEBUG=INFO',
    "CustomMPIArgs": 'env OMPI_MCA_BTL=self,sm,tcp,openib',
    "Timeout": None
    }

    if blob:
        job['volumes'] = {
            "myblob": {
                "_comment": "This will mount testcontainer in the storage account pavermatest inside the container at path '/blob'. The credentials required for accessing storage account pavermatest are below, in the 'credentials' section.",
                "type": "blobfuseVolume",
                "storageAccount": "msralaphilly",
                "containerName": "ml-la",
                "path": "/blob"
            }
        }
        job['credentials'] = {
            "storageAccounts": {
                "msralaphilly": {
                    "_comment": "Credentials for accessing 'pavermatest' storage account. Secrets can be saved with Philly from your Philly profile page at https://philly/#/userView/. With this the secret doesn't have to be maintained in the user's workspace.",
                    "key": "NJL+oSaTpZTWOOc0jM304H8tpSvydwoPKcQr5goPF8sXwzFb9c6NuWDng8vwemaIZnXc2AYVc6+LyYYNZAiAYw=="
                }
            }
        }

    job=json.dumps(job)

    url='https://philly/api/v2/submit'
    headers = {'Content-Type':'application/json'}
    requests.post(url, headers=headers, data=job, auth=HttpNtlmAuth(user, pwd), verify=False)

def submit():

    '''Distributed config'''
    world_size = 1 #number of machines you need
    ngpupernode = 8 #number of gpus you need on each machine

    nccl = True #better not change
    old_docker = False # better not change. Changing to true will be in-stable. But if you are running 2*4 jobs, it is fairly stable and might even be 15% faster than setting it to False.
    #vc = "msrmt" #vc you run your jobs
    #cluster = "wu2" #cluster you run your jobs

    cluster = "sc3"  # cluster you run your jobs
    vc = vc_maps[cluster]  # vc you run your jobs
    c10d = False

    '''Training config'''
    max_toks = 2560
    uf = 4 * 4096 * 32 // (max_toks * world_size * ngpupernode)
    print('uf', uf)

    lr = 0.0005
    max_lr = 0.0005
    lr_scheduler = "inverse_sqrt"
    cosine_period = 35000
    warm_updates = 4000
    save_updates = 1500
    log_interval = 200
    dataset = "wmt19.Round3ef4_rd2kdfe5_rd2bt.tokenized.en-fi.joined"
    #dataset = "wmt19.Round2ef2kdfe5bt.tokenized.en-fi.joined"
    arch = "transformer_wmt_en_de_big"
    #arch = "transformer_vaswani_wmt_en_de_big"
    layers = 6
    dropout = 0.3
    reloaddir = ""
    src = "en"
    tgt = 'fi'
    r2l = False
    #reloaddir = "wmt19.tokenized.en-fi.joined_transformer_wmt_en_de_big_dp0.3_seed4726_maxtok4096_uf16_lr0.00052_enc6_dec6_erlbsef2--r2l"
    #reloaddir = "wmt19.tokenized.en-fi.joined_transformer_wmt_en_de_big_dp0.3_seed9128_maxtok4096_uf11_lr0.0005_enc6_dec6_erlbsef--r2l"
    reloaddir = "3rd_ef_start_ef2rd2"
    #reloaddir = "FiEnR2L"
    blob = True
    expname = '3rdef_bl_wm'
    extra = expname

    post(dataset=dataset, vc=vc, cluster=cluster, name = expname, nprocs= ngpupernode, nnodes= world_size, docker_old = old_docker, nccl= nccl,
         log_interval= log_interval, max_toks= max_toks, uf= uf, lr = lr, max_lr = max_lr, lr_scheduler= lr_scheduler, warm_updates= warm_updates,
         arch= arch, layers = layers, dropout = dropout, reload_dir = reloaddir, cosine_period= cosine_period, save_interval_updates= save_updates,
         extra= extra, src= src, tgt= tgt , r2l=r2l, c10d=c10d, blob=blob)


if __name__=='__main__':
    submit()
    print('Submitted.')