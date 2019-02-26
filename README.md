# Introduction

This is the project specially for distributed training fairseq on Microsoft Philly.

To submit distributed jobs with the required #machines and #GPUs, see 

```
run/SubmitPhilly.py
run/train_enfi_transformer_philly_dist_tcp.sh
```

The script `train_enfi_transformer_philly_dist_tcp.sh` should be put on your hdfs (or blob but I have not tested yet), acting as the `ConfigFile` for your Philly job. 

Please note it is your duty and free to set up some params related to **actural batch size**, where `actural batch size = max_tokens * number of GPUs per machine * number of machines * update_freq`, such as the `lr` and `update_freq`. You can change all these variables in `scripts/SubmitPhilly.py`.

Some referece jobs on Philly to gain an overview of the performance comparision between single node job and distribued jobs:

```
Single node job: https://philly/#/job/wu2/msrmt/1549581441265_5735, actural batch size = 1 * 4 * 4096 * 32
2 nodes job: https://philly/#/job/wu2/msrmt/1549581441265_7946, actural batch size = 2 * 4 * 3277 * 20
4 nodes job: https://philly/#/job/wu2/msrmt/1549581441265_7512, actural batch size = 4 * 4 * 3277 * 10
```

If you would like to customize your code (based on FairSeq **v0.6.1**) with distributed training, but not directly use this repo, besides the running script above, maybe (**WARNING**: not tested) it is enough to only copy two functions in this project: the `infer_init_method` in `./fairseq/fairseq/distributed_utils.py` and `save_checkpoint` in `./train.py`, and replace the original ones in your code. If your code is based on FairSeq <= v0.6.0, please come to me for help.

# Warning

I have not tested whether this code/script could support single node jobs.

I have not tested the support of Blob.

Philly is in stable. If you encounter exeptions in running this code, please tell me the job link and I will have a check.

#What I am doing

Currently if the `actural batch size` is enlarged (e.g., by maintaining the same `max_tokens` and `update_freq` with the single node job), accompanied with a larger learning rate, the training is instable. See the output of this job: https://philly/#/logs/wu2/msrmt/fetia/1549581441265_7399/stdout/latest/ns. I'm trying to solve this.