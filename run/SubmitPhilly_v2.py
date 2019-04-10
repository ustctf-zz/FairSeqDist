import os
import requests
from requests_ntlm import HttpNtlmAuth
import json
import subprocess
import time
import threading
import random

from enum import Enum

user = 'fetia'

class JobMode(Enum):
    TRAIN = 1
    TEST = 2
    SAMPLE = 3

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
    'fp16': False,

    'dataset': 'wmt19.tokenized.en-fi.joined',
    'src': 'en',
    'tgt': 'fi',
    'r2l': False,

    'save_interval_updates': 0,
    'max_updates': 300000,
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
    'job_mode': JobMode.TRAIN,
    'gen_alpha': 1.2,
    'gen_subset': 'wmt18',
    'initial_model': "",
    'sample_batch_size': 1024,
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
    fp16 = kwargs.get('fp16', False)

    dataset = kwargs.get('dataset')
    src = kwargs.get('src', "en")
    tgt = kwargs.get('tgt', "fi")
    r2l = kwargs.get('r2l', False)

    reload_dir = kwargs.get('reload_dir', "")
    save_interval_updates = kwargs.get('save_interval_updates', 0)
    max_updates = kwargs.get('max_updates', 300000)

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

    job_mode = kwargs.get('job_mode', JobMode.TRAIN)
    gen_alpha = kwargs.get('gen_alpha', 1.2)
    gen_subset = kwargs.get('gen_subset', 'wmt18')
    initial_model = kwargs.get('initial_model', None)
    sdp = kwargs.get('sdp', None)
    source_file = kwargs.get('sample_source_file', None)
    sample_bs = kwargs.get('sample_batch_size', 1024)

    assert not job_mode or reload_dir != ""
    nas = net_code != ""
    if nas:
        update_code = False
    if job_mode is not JobMode.TRAIN:
        nnodes = 1
        nprocs = 1

    ngpus = nprocs * nnodes
    seed = random.randint(1, 15000)
    port = random.randint(1000, 9999)
    is_cosine = lr_scheduler == "cosine"
    cosine_command = "-MLR {} -COSP {}".format(max_lr, cosine_period)

    def getConfigFile():
        blob_ = "/blob/" if blob else ""

        if job_mode != JobMode.TRAIN: # inf
            config_name = "{}_enfi_transformer_philly.sh".format('gen' if job_mode == JobMode.TEST else 'sample')
            if nas:
                prefix = "fetia/Src/FairSeqDistFlex/run/"
            else:
                prefix = "fetia/Src/{}/run/".format("fairseq_latest" if cluster == "wu2" and not blob else "FairSeqDist")
        else : # train
            if nas:
                prefix = "fetia/Src/FairSeqDistFlex/run/"
                config_name = "train_enfi_transformer_philly_dist_nas.sh"
            else:
                prefix = "fetia/Src/{}/run/".format("fairseq_latest" if cluster == "wu2" and not blob else "FairSeqDist")
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
    'ExtraParams':" --dropout {} --dataset {} --warm-update {} -M {} --uf {} -E {} --nodes {} --port {} -s {} --nproc {} "
                  "-A {}  -LR {} -LRS {} -SI 1 --max-update {} -SIU {} --enc {} --dec {} -LI {} {} "
                  "{} {} --src {} --tgt {} {} {}"
                  "{} {} {} "
                  "{} {} {} "
                  "{} {} --batch-size {} --gen-set {} ".
        format(dropout, dataset, warm_updates, max_toks, uf, extra, nnodes, port, seed, nprocs,
               arch, lr, lr_scheduler, max_updates, save_interval_updates, layers, layers, log_interval, "--nccl" if nccl else "",
               "-RD {}".format(reload_dir) if reload_dir != " " else "", cosine_command if is_cosine else "", src, tgt, "--r2l" if r2l else "",
               "--c10d" if c10d else "", "-BLOB" if blob else "", "-UC" if update_code else " ", "--alpha {}".format(gen_alpha) if job_mode is not JobMode.TRAIN else "",
               "-N usr_net_code/{}".format(net_code) if nas else "", "-I {}".format(initial_model) if initial_model is not None else "", "--sdp {}".format(sdp) if sdp is not None else "",
               "--src-file {}".format(source_file) if source_file is not None else "", "--fp16" if fp16 else "", sample_bs, gen_subset,
               ),
    "SubmitCode": "p",
    "IsMemCheck": False,
    "IsCrossRack": False,
    "Registry": "phillyregistry.azurecr.io",
    "Repository": "philly/jobs/custom/pytorch",
    "Tag": "fairseq-0.6.0" if not fp16 else "pytorch1.0.0-py36-nlp",
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

    if cluster == 'wu2':
        job['queue'] = 'msra'

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
         },

    'ef_ls_transf_combo_bt1':
        {'net_code': 'e6_lstm3_d6_lstm.json',
         'reload_dir': 'wmt19.tokenized.en-fi.joined_nas_transformer_wmt_en_de_big_nc_e6_lstm3_d6_lstm_dp0.3_seed884_maxtok3072_lr0.001_SI1_ef_ls_transf_nc_1.0',
        'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.00125, 'warm_updates': 8000, 'max_toks': 2560,
         'dataset': 'wmt19.bt1_model_combo1.tokenized.en-fi.joined',},

    'fe_ls_transf_combo_bt1':
        {'net_code': 'e6_lstm3_d6_lstm.json',
         'reload_dir': 'wmt19_fien_lstm_combo_transf',
          'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.00125, 'warm_updates': 8000, 'max_toks': 2560, 'src': 'fi', 'tgt': 'en',
         'dataset': 'wmt19.bt1_model_combo1.tokenized.fi-en.joined',
         },

    'ef_ls_transf_combo2_base':
        {'net_code': 'e6_lstm3_d6_lstm6.json',
         'reload_dir': 'wmt19_enfi_lstm_more_combo_transf',
        'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.00125, 'warm_updates': 8000, 'max_toks': 3072,
         },

    'fe_ls_transf_combo2_base':
        {'net_code': 'e6_lstm3_d6_lstm6.json',
         'reload_dir': 'wmt19_fien_lstm_more_combo_transf',
        'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.00125, 'warm_updates': 8000, 'max_toks': 2560, 'src': 'fi', 'tgt': 'en',},

    'ef_rl_bt1':
        {
            'reload_dir': 'EnFiR2LBT1', 'dataset': 'wmt19.r2l.bt1.tokenized.en-fi.joined', 'log_interval': 200, 'r2l': True,
        },
    'fe_rl_bt1':
        {
            'reload_dir': 'FiEnR2LBT1', 'dataset': 'wmt19.r2l.bt1.tokenized.en-fi.joined', 'log_interval': 200, 'r2l': True, 'src': 'fi', 'tgt': 'en',
        },

    'ef_rl_btkd2':
        {
            'reload_dir': 'EnFiR2LBTKD2', 'dataset': 'wmt19.2nd_r2l.tokenized.en-fi.joined ', 'log_interval': 250, 'r2l': True, 'save_interval_updates': 1500,
        },

    'p_rl_btkd2_cont_wu2':
        {
            'cluster': 'wu2', 'max_toks': 3584, 'nnodes': 1, 'nprocs': 4, 'lr': 0.00005,
            'reload_dir': 'EnFiR2LBTKD2', 'dataset': 'wmt19.2nd_r2l.tokenized.en-fi.joined ', 'log_interval': 250, 'r2l': True, 'save_interval_updates': 1500,
        },

    'p_rl_btkd2_cont_wu2_small':
        {
            'cluster': 'wu2', 'max_toks': 3072, 'nnodes': 1, 'nprocs': 4, 'lr': 0.00005,
            'reload_dir': 'EnFiR2LBTKD2', 'dataset': 'wmt19.2nd_r2l.tokenized.en-fi.joined ', 'log_interval': 250, 'r2l': True, 'save_interval_updates': 1500,
        },

    'fe_rl_btkd2':
        {
            'reload_dir': 'FiEnR2LBTKD2', 'dataset': 'wmt19.2nd_r2l.tokenized.en-fi.joined ', 'log_interval': 250, 'r2l': True, 'src': 'fi', 'tgt': 'en', 'save_interval_updates': 1500,
        },
    'n_rl_btkd2_cont_wu2':
        {
            'cluster': 'wu2', 'max_toks': 4096, 'nnodes': 1, 'nprocs': 4, 'lr': 0.00005,
            'reload_dir': 'FiEnR2LBTKD2', 'dataset': 'wmt19.2nd_r2l.tokenized.en-fi.joined ', 'log_interval': 250, 'r2l': True, 'src': 'fi', 'tgt': 'en', 'save_interval_updates': 1500,
        },

    'p_wu2_base_dist':
        {
            'cluster': 'wu2','reload_dir': 'EnFiBaseWu2', 'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 50, 'r2l': False,
            'blob': False, 'nnodes': 4, 'nprocs': 4, 'dropout': 0.25, 'max_toks': 3072,
        },

    'p_wu2_base_dist_blob':
        {
            'cluster': 'wu2','reload_dir': 'EnFiBaseWu2', 'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 1,
            'blob': True, 'nnodes': 2, 'nprocs': 2, 'dropout': 0.25, 'max_toks': 3072, 'save_interval_updates': 8, 'max_updates': 25,
        },

    'n_wu2_base_dist':
        {
            'cluster': 'wu2','reload_dir': 'FiEnBaseWu2', 'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 50, 'r2l': False, 'src': 'fi', 'tgt': 'en',
            'blob': False, 'nnodes': 4, 'nprocs': 2, 'dropout': 0.25, 'max_toks': 3584,
        },
    'n_wu2_base_single':
        {
            'cluster': 'wu2','reload_dir': 'FiEnBaseWu2', 'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 50, 'r2l': False, 'src': 'fi', 'tgt': 'en',
            'blob': True, 'nnodes': 1, 'nprocs': 4, 'dropout': 0.25, 'max_toks': 3584,
        },

    'ef_base_CNM_Cosine':
        {
            'net_code': 'CNM.json',
            'reload_dir': 'EnFiCNMBaseCosineReal',
            'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 100,
            'blob': True, 'arch': 'nas_transformer_wmt_en_de_big', 'lr': 1e-7, 'warm_updates': 8000,
            'nnodes': 1, 'nprocs': 8, 'max_toks': 3072,
            'dropout': 0.2,
            'lr_scheduler': 'cosine', 'max_lr':0.0015, 'cosine_period':12000,
        },

    'fe_base_CNM_Cosine':
        {
            'net_code': 'CNM.json',
            'reload_dir': 'EnFiCNMBaseCosine',
            'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 100,
            'blob': True, 'arch': 'nas_transformer_wmt_en_de_big', 'lr': 1e-7, 'warm_updates': 8000,
            'nnodes': 1, 'nprocs': 8, 'max_toks': 3072,
            'dropout': 0.3, 'src': 'fi', 'tgt': 'en',
            'lr_scheduler': 'cosine', 'max_lr':0.0015, 'cosine_period':12000,
        },

    'fe_BT1_CNM':
        {
            'net_code': 'CNM.json',
            'reload_dir': 'FiEnCNMBT1',
            'dataset': 'wmt19.CNMbt1.tokenized.fi-en.joined', 'log_interval': 100,
            'blob': True, 'arch': 'nas_transformer_wmt_en_de_big', 'lr': 1e-7, 'warm_updates': 8000,
            'nnodes': 1, 'nprocs': 8, 'max_toks': 3072,
            'dropout': 0.25, 'src': 'fi', 'tgt': 'en',
            'lr_scheduler': 'cosine', 'max_lr':0.0015, 'cosine_period':12000,
        },

    'fe_BT1_CNM_NormalLR':
            {
                'net_code': 'CNM.json',
                'reload_dir': 'FiEnCNMBT1NormalLR',
                'dataset': 'wmt19.CNMbt1.tokenized.fi-en.joined', 'log_interval': 100,
                'blob': True, 'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.0005, 'warm_updates': 8000,
                'nnodes': 1, 'nprocs': 8, 'max_toks': 3072,
                'dropout': 0.25, 'src': 'fi', 'tgt': 'en',
            },


    'ef_base_CNM':
        {
            'net_code': 'CNM.json',
            'reload_dir': 'EnFiCNMBase',
            'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 50,
            'blob': True, 'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.0015, 'warm_updates': 8000,
            'nnodes': 1, 'nprocs': 8, 'max_toks': 3072,
            'dropout': 0.2,
        },

    'ef_BT1_CNM':
        {
            'net_code': 'CNM.json',
            'reload_dir': 'EnFiCNMBT1',
            'dataset': 'wmt19.CNMbt1.tokenized.en-fi.joined', 'log_interval': 50,
            'blob': True, 'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.0015, 'warm_updates': 8000,
            'nnodes': 1, 'nprocs': 8, 'max_toks': 3072,
            'dropout': 0.25,
        },

    'fe_base_CNM':
        {
            'net_code': 'CNM.json',
            'reload_dir': 'FiEnCNMBase',
            'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 25,
            'blob': True, 'arch': 'nas_transformer_wmt_en_de_big', 'lr': 0.0015, 'warm_updates': 8000,
            'nnodes': 1, 'nprocs': 8, 'max_toks': 3072,
            'dropout': 0.25, 'src': 'fi', 'tgt': 'en',
        },

    'ef_bt1_rerun':
        {
            'reload_dir': 'EnFiBT1Rerun',
            'dataset': 'wmt19.db.bt1.tokenized.en-fi.joined', 'log_interval': 100,
            'max_toks': 2560,
        },

    'fe_bt1_rerun':
        {
            'reload_dir': 'FiEnBT1Rerun',
            'dataset': 'wmt19.db.bt1.tokenized.fi-en.joined', 'log_interval': 100,
            'max_toks': 2560, 'src': 'fi', 'tgt': 'en',
        },

    'ef_sc3_base_cont':
        {
            'reload_dir': 'EnFiBaseRunSc3ContWu2', 'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 50, 'r2l': False,
             'nnodes': 1, 'nprocs': 8, 'dropout': 0.25, 'max_toks': 2304,
        },

    'fe_sc3_base_cont':
        {
            'reload_dir': 'FiEnBaseSc3', 'dataset': 'wmt19.tokenized.en-fi.joined', 'log_interval': 50, 'r2l': False, 'src': 'fi', 'tgt': 'en',
            'nnodes': 1, 'nprocs': 8, 'dropout': 0.25, 'max_toks': 2560,
        },

    'ef_lr_btkd2':
        {
            'reload_dir': 'EnFiBaseSc3Old', 'dataset': 'wmt19.2nd_l2r.tokenized.en-fi.joined ', 'log_interval': 250, 'save_interval_updates': 1500,
        },

    'fe_lr_btkd2':
        {
            'reload_dir': 'FiEnBaseSc3', 'dataset': 'wmt19.2nd_l2r.tokenized.en-fi.joined ', 'log_interval': 250, 'save_interval_updates': 1500, 'src': 'fi', 'tgt': 'en',
        },
}

def getJobConfigs(name = "", job_mode = JobMode.TRAIN, **kwargs):
    if name not in job_pools:
        raise Exception('name {} not in job pools'.format(name))

    job_config = {**base_job_args, **job_pools[name], **kwargs}
    job_config['job_mode'] = job_mode
    job_config['name'] = '{}.{}'.format('tr' if job_mode == JobMode.TRAIN else 'ge' if job_mode == JobMode.TEST else 'sam', name)
    if (job_mode == JobMode.TEST):
        job_config['name'] += job_config['gen_subset'][-2:]

    if(job_mode == JobMode.SAMPLE):
        job_config['name'] += '_' + job_config['sample_source_file']
    if job_mode is not JobMode.TRAIN:
        assert 'reload_dir' in job_config and job_config['reload_dir'] != ''
    else:
        job_config['uf'] = 4 * 4096 * 32 // (job_config['max_toks'] * job_config['nnodes'] * job_config['nprocs'])
    job_config['extra'] = name
    return job_config


def submit():

    specific_args = {
        'cluster': 'sc3',
        'sample_batch_size': 5120,
        'update_code': False,
        'initial_model': "checkpoint100.pt",
        'gen_alpha': 1.2,
        'gen_subset': 'wmt17',
        'sample_source_file': None,
        'sdp': "train.mono.filtered.bpe.1st_round.small"
    }

    job_name = "fe_bt1_rerun"
    job_mode = JobMode.TEST

    if job_mode is not JobMode.SAMPLE:
        job_args = getJobConfigs(job_name, job_mode, **specific_args)
        post(**job_args)
    else:
        NPieces = 20
        for i in range(0, NPieces, 1):
            first_char = chr(ord('a') + i // 26)
            second_char = chr(ord('a') + i % 26)
            datapart = first_char + second_char
            specific_args['sample_source_file'] = 'part.{}'.format(datapart)
            job_args = getJobConfigs(job_name, job_mode, **specific_args)
            post(**job_args)
            #print(i, datapart)


if __name__=='__main__':
    submit()
    print('Submitted.')