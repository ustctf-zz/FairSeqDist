#!/usr/bin/env bash

# FIXME: Hard code here, assume that the training script in `.../ProjectRoot/ignored_scripts/` (depth = 1)
ProjectDir=$(readlink -f $(dirname ${PHILLY_CONFIG_FILE})/../ )

DataDir=/hdfs/msrmt/fetia/Data
SaveDir=/hdfs/msrmt/fetia/Ckpts
LogDir=/hdfs/msrmt/fetia/Log

Dataset=wmt19.tokenized.en-fi
Arch=transformer_wmt_en_de_big
seed=1
dropout=0.3
install_fairseq=false
max_update=2000
# no_epoch_checkpoints="--no-epoch-checkpoints"
no_epoch_checkpoints=""
UpdateFreq=64
Extra=""
MaxTokens=2048
decoder_heads_dropout=0.1
encoder_heads_dropout=0.1
LR=0.0005
DdpBackend="no_c10d"
ShareEmb=""
FP16=false

Generate="false"

while [ "$1" != "" ]; do
	case $1 in
		-E | --extra )
			shift
			Extra=$1
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
		-enc_hd | --encoder_heads_dropout )
			shift
			encoder_heads_dropout=$1
			;;
		-dec_hd | --decoder_heads_dropout )
			shift
			decoder_heads_dropout=$1
			;;
		-I | --install-fairseq )
			install_fairseq=true
			;;
		--max-update )
			shift
			max_update=$1
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
		--fp16 )
			FP16=true
			FP16Args="--fp16"; FP16SaveName="fp16_"; FP16LogName="fp16-"
			;;
		--uf | --update-freq )
			shift
			UpdateFreq=$1
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

FullSaveDir=${SaveDir}/${Dataset}_${Arch}_dp${dropout}_seed${seed}_ehd${encoder_heads_dropout}_dhd${decoder_heads_dropout}_${FP16SaveName}${Extra}
LogFilename=${Dataset}-${Arch}-dp${dropout}-seed${seed}-ehd${encoder_heads_dropout}-dhd${decoder_heads_dropout}-${FP16LogName}${Extra}

set -x
python ${ProjectDir}/train.py \
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
    --log-interval 50 \
    --save-interval 1 --save-interval-updates 100 --keep-interval-updates 0 \
	--dropout ${dropout} --seed ${seed} \
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
