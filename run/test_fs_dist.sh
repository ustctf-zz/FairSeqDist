#!/usr/bin/env bash

ProjectDir=/home/fetia/Src/

DataDir=/home/fetia/Data
SaveDir=/home/fetia/Ckpts
LogDir=/home/fetia/Logs
DistDir=/home/fetia/Dist

Dataset=wmt19.50k.tokenized.en-fi
Arch=transformer_wmt_en_de_big
seed=1
dropout=0.3
install_fairseq=false
max_update=0
no_epoch_checkpoints=""
UpdateFreq=64
Extra="Dist"
MaxTokens=2048
LR=0.0005
DdpBackend="no_c10d"
ShareEmb=""
FP16=false
DistWorldSize=8
DistBackEnd="gloo"
Nnodes=2
NProcPerNode=2


# Share all embeddings for joined dict automatically.
if [[ "${Dataset}" =~ .*\.joined$ ]]; then
	ShareEmb="--share-all-embeddings"
fi

FullSaveDir=${SaveDir}/${Dataset}_${Arch}_dp${dropout}_seed${seed}_maxtok${MaxTokens}_${Extra}_1.0
LogFilename=${Dataset}-${Arch}-dp${dropout}-seed${seed}-maxtok${MaxTokens}-${Extra}_1.0
DistFilename=${Dataset}-${Arch}-dp${dropout}-seed${seed}-maxtok${MaxTokens}-${Extra}_1.0

>DistFilename

set -x
python -m torch.distributed.launch --nproc_per_node=${NProcPerNode} \
	--nnodes=${Nnodes} --node_rank=0 --master_addr="10.150.144.93" \
    --master_port=1234 
	${ProjectDir}/train.py \
	--ddp-backend ${DdpBackend} \
	${FP16Args} \
	${DataDir}/${Dataset} \
    --arch ${Arch} \
    ${ShareEmb} \
    --max-update ${max_update} \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --update-freq ${UpdateFreq} \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr ${LR} --min-lr 1e-09 \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens ${MaxTokens} \
    --no-progress-bar \
    ${no_epoch_checkpoints} \
    --save-dir ${FullSaveDir} \
    --log-interval 10 \
    --save-interval 1 --save-interval-updates 10000 --keep-interval-updates 0 --dropout ${dropout} --seed ${seed}
	#\
	# --distributed-backend ${DistBackEnd} --distributed-init-method ${DistFilename} --distributed-world-size ${Nnodes}
    #2>&1 | tee ${LogDir}/${LogFilename}-train.log.txt
set +x

