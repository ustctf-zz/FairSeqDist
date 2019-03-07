#!/usr/bin/env bash

# FIXME: Hard code here, assume that the training script in `.../ProjectRoot/ignored_scripts/` (depth = 1)
ProjectDir=$(readlink -f $(dirname ${PHILLY_CONFIG_FILE})/../ )

DataDir=/hdfs/${PHILLY_VC}/fetia/Data
SaveDir=/hdfs/${PHILLY_VC}/fetia/Ckpts
LogDir=/hdfs/${PHILLY_VC}/fetia/Log
DistDir=/hdfs/${PHILLY_VC}/fetia/Dist

Dataset=wmt19.tokenized.en-fi
Arch=transformer_wmt_en_de_big
seed=1
dropout=0.3
install_fairseq=false
max_update=0
no_epoch_checkpoints=""
UpdateFreq=64
Extra="Dist"
MaxTokens=2048
LRScheduler="inverse_sqrt"
LR=0.0005
MaxLR=0.0005
DdpBackend="no_c10d"
ShareEmb=""
FP16=false
DistWorldSize=8
DistBackEnd="gloo"
Nnodes=2
NProcPerNode=2
LogInterval=50
SaveInterval=1
SaveIntervalUpdates=10000
enc=6
dec=6
PORT=1234
WarmUpdates=4000
ReloadDirName=""
CosinePeriod=40000

Generate="false"

while [ "$1" != "" ]; do
	case $1 in
		-E | --extra )
			shift
			Extra=$1
			;;
		-LRS | --lr-scheduler )
			shift
			LRScheduler=$1
			;;
		-RD | --reload-dir )
			shift
			ReloadDirName=$1
			;;
		-LR | --learning_rate )
			shift
			LR=$1
			;;
		-MLR | --max_lr )
			shift
			MaxLR=$1
			;;
		-A | --arch )
			shift
			Arch=$1
			;;
		-D | --dataset )
			shift
			Dataset=$1
			;;
		-s | --seed )
			shift
			seed=$1
			;;
		-d | --dropout )
			shift
			dropout=$1
			;;
		--enc )
			shift
			enc=$1
			;;
		--dec )
			shift
			dec=$1
			;;
		-I | --install-fairseq )
			install_fairseq=true
			;;
		--max-update )
			shift
			max_update=$1
			;;
		--warm-update )
			shift
			WarmUpdates=$1
			;;
		--ec | --epoch-checkpoints )
			no_epoch_checkpoints=""
			;;
		--no-ec | --no-epoch-checkpoints )
			no_epoch_checkpoints="--no-epoch-checkpoints"
			;;
		-G | --generate )
			Generate="true"
			;;
		--no-c10d )
			DdpBackend="no_c10d"
			;;
		--c10d )
			DdpBackend="c10d"
			;;
		--nccl )
			DistBackEnd="nccl"
			;;
		--fp16 )
			FP16=true
			FP16Args="--fp16"; FP16SaveName="fp16_"; FP16LogName="fp16-"
			;;
		--uf | --update-freq )
			shift
			UpdateFreq=$1
			;;
		--nodes )
			shift
			Nnodes=$1
			;;
		--nproc )
			shift
			NProcPerNode=$1
			;;
		--port )
			shift
			PORT=$1
			;;
		-SI | --save-intervals )
			shift
			SaveInterval=$1
			;;
		-SIU | --save-interval-updates )
			shift
			SaveIntervalUpdates=$1
			;;
		-CP | --cosine-period )
			shift
			CosinePeriod=$1
			;;
		-LI | --log-intervals )
			shift
			LogInterval=$1
			;;
		-M | --max-tokens )
			shift
			MaxTokens=$1
			;;
		* )
			;;
	esac
	shift
done

function status_check {
	echo "======================= GPU & CUDA Version Checks ========================"
	nvidia-smi
	cat /usr/local/cuda/version.txt
	nvcc -V

	echo "=================== Python & PyTorch Version Checks ==================="
	python -V
	python -c 'import torch; print(torch.__version__)'
	echo "PHILLY_GPU_COUNT" ${PHILLY_GPU_COUNT}

	echo "======================= Philly File System Checks ========================"
	echo "I am " $(whoami)
	echo -n "CURRENT_DIRECTORY "
	pwd
	echo "PHILLY_HOME" ${PHILLY_HOME}
	ls -alh ${PHILLY_HOME}
	echo "PHILLY_USER_DIRECTORY" ${PHILLY_USER_DIRECTORY}
	ls -alh ${PHILLY_USER_DIRECTORY}

	mkdir -p ${LogDir}
	echo "ProjectDir" ${ProjectDir}
	ls -alh ${ProjectDir}
}

function install_fairseq_fn {
	OldPwd=$(pwd)
	set -x
	cd ${ProjectDir}
	python setup.py build
	# TODO: Python3.6 is hard coding here, change "--prefix" to "--install-dir"?
	export PYTHONPATH="$(pwd)/.local/lib/python3.6/site-packages:${PYTHONPATH}"
	python setup.py install --prefix .local
	python setup.py develop --prefix .local
	cd ${OldPwd}
	set +x
}

status_check

if [ "$install_fairseq" = "true" ]; then
	install_fairseq_fn
else
	# TODO: Python3.6 is hard coding here, change "--prefix" to "--install-dir"?
	export PYTHONPATH="${ProjectDir}/.local/lib/python3.6/site-packages:${PYTHONPATH}"
fi

# Share all embeddings for joined dict automatically.
if [[ "${Dataset}" =~ .*\.joined$ ]]; then
	ShareEmb="--share-all-embeddings"
fi

if ["$ReloadDirName" = ""]; then 
	FullSaveDir=${SaveDir}/${Dataset}_${Arch}_dp${dropout}_seed${seed}_maxtok${MaxTokens}_uf${UpdateFreq}_lr${LR}_SI${SaveInterval}_enc${enc}_dec${dec}_${Extra}_1.0
else
	FullSaveDir=${SaveDir}/${ReloadDirName}
fi

LogFilename=${Dataset}-${Arch}-dp${dropout}-seed${seed}-maxtok${MaxTokens}-uf${UpdateFreq}-lr${LR}-SI${SaveInterval}-enc${enc}-dec${dec}-${Extra}_1.0
DistFilename=${FullSaveDir}/${Dataset}-${Arch}-dp${dropout}-seed${seed}-maxtok${MaxTokens}-uf${UpdateFreq}-lr${LR}-SI${SaveInterval}-enc${enc}-dec${dec}-${Extra}_1.0

mkdir -p ${FullSaveDir}
echo "FullSaveDir" ${FullSaveDir}

MASTER_IP="192.162.1.1"
THIS_IP="`ifconfig eth0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'`"

if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
		rm -rf ${DistFilename}
		ifconfig eth0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'>${DistFilename}
		MASTER_IP=${THIS_IP}
		echo "I am master node with IP" ${THIS_IP}
		while [ ! -f ${DistFilename} ]
		do
			echo "I am master node, sleep another 5 seconds"
			sleep 5
		done
		sleep 5
		echo "Files in Full Save Dir"
		ls -alh ${FullSaveDir}
		echo "Sizes in Save Dir"
		du --max-depth=1 -h ${SaveDir}
else
		sleep 5
		while [ ! -f ${DistFilename} ]
		do
			echo "I am slave node, sleep another 5 seconds" ${THIS_IP}
			sleep 5
		done
		MASTER_IP="`cat ${DistFilename}`"
		echo "I am slave node with IP" ${THIS_IP} 
		echo "my master IP" ${MASTER_IP}
fi	

set -x
python -m torch.distributed.launch --nproc_per_node=${NProcPerNode} \
	--nnodes=${Nnodes} --node_rank=${OMPI_COMM_WORLD_RANK} --master_addr=${MASTER_IP} \
    --master_port=${PORT} \
	${ProjectDir}/train.py \
	--ddp-backend ${DdpBackend} \
	${FP16Args} \
	${DataDir}/${Dataset} \
    --arch ${Arch} \
	--encoder-layers ${enc} --decoder-layers ${dec} \
    ${ShareEmb} \
    --max-update ${max_update} \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --update-freq ${UpdateFreq} \
    --lr-scheduler ${LRScheduler} --warmup-init-lr 1e-07 --warmup-updates ${WarmUpdates} --lr ${LR} \
	--min-lr 1e-09 --max-lr ${MaxLR} --lr-period-updates ${CosinePeriod} --lr-shrink 1.0 \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens ${MaxTokens} \
    --no-progress-bar \
    --save-dir ${FullSaveDir} \
    --log-interval ${LogInterval} \
    --save-interval ${SaveInterval} --save-interval-updates ${SaveIntervalUpdates} --keep-interval-updates 0 \
	--dropout ${dropout} --seed ${seed} --distributed-backend ${DistBackEnd} --master-address-file ${DistFilename} \
    2>&1 | tee ${LogDir}/${LogFilename}-train.log.txt
set +x

if [ "$Generate" == "true" ]
then
	set -x
	# Average 10 latest checkpoints:
	python ${ProjectDir}/scripts/average_checkpoints.py --inputs ${FullSaveDir} \
	    --num-epoch-checkpoints 10 --output ${FullSaveDir}/model.pt \
		2>&1 | tee ${LogDir}/${LogFilename}-score.log.txt

	# Generate:
	python ${ProjectDir}/generate.py ${DataDir}/${Dataset} \
		${FP16Args} \
	    --path ${FullSaveDir}/model.pt \
	    --batch-size 128 --beam 5 --remove-bpe \
	    --quiet \
	    2>&1 | tee ${LogDir}/${LogFilename}-score.log.txt -a
	set +x
fi
