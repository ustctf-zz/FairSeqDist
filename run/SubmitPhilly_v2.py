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

base_job_args = {
    'cluster': 'sc3',
    'blob': True,
    'nprocs': 8,
    'nnodes': 1,
    'nccl': True,
    'c10d': False,

    'dataset': 'wmt19.tokenized.en-fi.joined',
    'src': 'en',
    'tgt': 'fi',
    'r2l': False,

    'save_interval_updates': 0,
    'log_interval': 150,
    'max_toks': 2560,
    'dropout': 0.3,

    'lr': 0.0005,
    'lr_scheduler': 'inverse_sqrt',
    'warm_updates': 4000,

    'net_code': "",
    'arch': 'transformer_wmt_en_de_big',
    'layers': 6,

    'update_code': False,
    'is_gen': False,
    'gen_alpha': 1.2,
}

def post(**kwargs):

    cluster = kwargs.get('cluster', 'wu2')
    vc = kwargs.get('vc', vc_maps[cluster])
    blob = kwargs.get('blob', True)
    name = kwargs.get('name')
    nprocs = kwargs.get('nprocs', 4)
    nnodes = kwargs.get('nnodes', 1)
    nccl = kwargs.get('nccl', True)
    c10d = kwargs.get('c10d', False)

    dataset = kwargs.get('dataset')
    src = kwargs.get('src', "en")
    tgt = kwargs.get('tgt', "fi")
    r2l = kwargs.get('r2l', False)

    reload_dir = kwargs.get('reload_dir', "")
    save_interval_updates = kwargs.get('save_interval_updates', 0)
    log_interval = kwargs.get('log_interval', 50)
    max_toks = kwargs.get('max_toks', 4096)
    uf = kwargs.get('uf', 32)
    dropout = kwargs.get('dropout', 0.3)

    lr = kwargs.get('lr', 0.0005)
    lr_scheduler = kwargs.get('lr_scheduler', "inverse_sqrt")
    cosine_period = kwargs.get('cosine_period', 40000)
    max_lr = kwargs.get('max_lr', 0.0005)
    warm_updates = kwargs.get('warm_updates', 4000)

    net_code = kwargs.get('net_code', "")
    arch = kwargs.get('arch',"transformer_wmt_en_de_big")
    layers = kwargs.get('layers', 6)

    extra = kwargs.get('extra', "")
    update_code = kwargs.get('update_code', False)

    is_gen = kwargs.get('is_gen', False)
    gen_alpha = kwargs.get('gen_alpha', 1.2)

    assert not is_gen or reload_dir != ""
    nas = net_code != ""
    if nas:
        update_code = False
    if is_gen:
        nnodes = 1
        nprocs = 1

    ngpus = nprocs * nnodes
    seed = random.randint(500, 9999)
    port = random.randint(1000, 9999)
    is_cosine = lr_scheduler == "cosine"
    cosine_command = "-MLR {} -CP {}".format(max_lr, cosine_period)

    def getConfigFile():
        blob_ = "/blob/" if blob else ""

        if is_gen: # inf
            config_name = "gen_enfi_transformer_philly.sh"
            if nas:
                prefix = "fetia/Src/FairSeqDistFlex/run/"
            else:
                prefix = "fetia/Src/{}/run/".format("fairseq_latest" if cluster == "wu2" else "FairSeqDist")
        else: # train
            if nas:
                prefix = "fetia/Src/FairSeqDistFlex/run/"
                config_name = "train_enfi_transformer_philly_dist_nas.sh"
            else:
                prefix = "fetia/Src/{}/run/".format("fairseq_latest" if cluster == "wu2" else "FairSeqDist")
                config_name = "train_enfi_transformer_philly_dist_tcp.sh"

        return blob_ + prefix + config_name

    print('Using seed {} and port {}, config file {}'.format(seed, port, getConfigFile()))

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
                  "{} {} --src {} --tgt {} {} {}"
                  "{} {} {} "
                  "{} ".
        format(dropout, dataset, warm_updates, max_toks, uf, extra, nnodes, port, seed, nprocs,
               arch, lr, lr_scheduler, save_interval_updates, layers, layers, log_interval, "--nccl" if nccl else "",
               "-RD {}".format(reload_dir) if reload_dir != " " else "", cosine_command if is_cosine else "", src, tgt, "--r2l" if r2l else "",
               "--c10d" if c10d else "", "-BLOB" if blob else "", "-UC" if update_code else " ", "--alpha {}".format(gen_alpha) if is_gen else "",
               "-N usr_net_code/{}".format(net_code) if nas else ""),
    "SubmitCode": "p",
    "IsMemCheck": False,
    "IsCrossRack": False,
    "Registry": "phillyregistry.azurecr.io",
    "Repository": "philly/jobs/custom/pytorch",
    "Tag": "fairseq-0.6.0",
    "OneProcessPerContainer": True,
    "DynamicContainerSize": False,
    "NumOfContainers": nnodes,
    "CustomMPIArgs": 'env OMPI_MCA_BTL=self,sm,tcp,openib',
    "Timeout": None
    }

    print(job)

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

job_pools = {
    'fe_r2l_on_2ndbt':
        {
          'dataset': 'wmt19.Round2ef2kdfe5bt.tokenized.en-fi.joined',
            'reload_dir': 'wmt19.Round2ef2kdfe5bt.tokenized.en-fi.joined_transformer_wmt_en_de_big_dp0.3_seed5334_maxtok2560_uf25_lr0.0005_enc6_dec6_rl_rd2_fe--r2l',
            'r2l': True,
            'src': 'fi', 'tgt': 'en',
            'blob': False,
        },
    'ef_ls_transf_combo_base':
        {'net_code': 'e6_lstm3_d6_lstm.json',
         'reload_dir': 'wmt19.tokenized.en-fi.joined_nas_transformer_wmt_en_de_big_nc_e6_lstm3_d6_lstm_dp0.3_seed884_maxtok3072_lr0.001_SI1_ef_ls_transf_nc_1.0',
        'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.001, 'warm_updates': 8000, 'max_toks': 3072},
    'fe_ls_transf_combo_base':
        {'net_code': 'e6_lstm3_d6_lstm.json',
         'reload_dir': 'wmt19_fien_lstm_combo_transf',
          'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.001, 'warm_updates': 8000, 'max_toks': 2560, 'src': 'fi', 'tgt': 'en',
         'gen_alpha': 1.6,
         },

    'ef_rl_bt1':
        {
            'reload_dir': 'EnFiR2LBT1', 'dataset': 'wmt19.r2l.bt1.tokenized.en-fi.joined', 'log_interval': 200, 'r2l': True,
        },
}

def getJobConfigs(name = "", train = True, update_code = False):
    if name not in job_pools:
        raise Exception('name {} not in job pools'.format(name))
    job_config = {**base_job_args, **job_pools[name]}
    job_config['is_gen'] = not train
    job_config['name'] = '{}.{}'.format('tr' if train else 'ge', name)
    if not train:
        assert 'reload_dir' in job_config and job_config['reload_dir'] != ''
    else:
        job_config['uf'] = 4 * 4096 * 32 // (job_config['max_toks'] * job_config['nnodes'] * job_config['nprocs'])
    job_config['extra'] = name
    job_config['update_code'] = update_code
    return job_config

def submit():

    job_name = "fe_ls_transf_combo_base"
    train = False
    job_args = getJobConfigs(job_name, train, update_code= False)
    post(**job_args)


if __name__=='__main__':
    submit()
    print('Submitted.')