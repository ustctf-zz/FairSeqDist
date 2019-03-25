#!/usr/bin/env bash

# FIXME: Hard code here, assume that the training script in `.../ProjectRoot/ignored_scripts/` (depth = 1)
ProjectDir=$(readlink -f $(dirname ${PHILLY_CONFIG_FILE})/../ )
OutputDir=${PHILLY_INPUT_DIRECTORY}
DataDir=/hdfs/${PHILLY_VC}/fetia/Data
SaveDir=/hdfs/${PHILLY_VC}/fetia/Ckpts
LogDir=/hdfs/${PHILLY_VC}/fetia/Log

SrcLan="en"
TgtLan="fi"

echo "Sizes in Save Dir"
du --max-depth=1 -h ${SaveDir}
		
BeamSize=5
LenPen=1.2
FP16=false
Sacre=true
CapOut=""
SacreArgs=""
Detokenizer=${ProjectDir}/../mosesdecoder/scripts/tokenizer/detokenizer.perl

Dataset=wmt19.tokenized.en-fi
CkptDir=wmt19.tokenized.en-fi.joined_transformer_vaswani_wmt_en_de_big_dp0.1_seed1_
GenSubset=test
SacreBLEUTestSet=wmt18
R2LArgs=""
UpdateCode=false

while [ "$1" != "" ]; do
	case $1 in
		-C | --ckpt-dir )
			shift
			CkptDir=$1
			;;
		-D | --dataset )
			shift
			Dataset=$1
			;;
		-b | --beam )
			shift
			BeamSize=$1
			;;
		--alpha | --lenpen )
			shift
			LenPen=$1
			;;
		-s | --sacre )
			Sacre=true
			;;
		--no-sacre )
			Sacre=false
			SacreArgs=""
			;;
		--r2l )
			R2LArgs="--r2l --recover-l2r"
			;;
		--cap-out )
			CapOut="--cap-output"
			;;
		--src )
			shift
			SrcLan=$1
			;;
		--tgt )
			shift
			TgtLan=$1
			;;
		-G | --gen-set )
			shift
			GenSubset=$1
			SacreBLEUTestSet=$1
			;;
		-UC | --update-code )
			UpdateCode=true
			;;
		* )
			;;
	esac
	shift
done

TmpFile=${CkptDir}-tmp.txt
LogFilename=${LogDir}/${CkptDir}-score.log.txt

rm -v ${LogFilename}

FullSaveDir=${SaveDir}/${CkptDir}

# Detect FP16 automatically.
if [[ "${CkptDir}" =~ .*fp16.* ]]; then
	FP16=true
	FP16Args="--fp16"
fi


# Set path.
OldPwd=$(pwd)
set -x
cd ${ProjectDir}
export PYTHONPATH="$(pwd)/.local/lib/python3.6/site-packages:${PYTHONPATH}"
set +x

if [ "${Sacre}" == "true" ]; then
	SacreArgs="--output-file ${TmpFile}"
	echo "Installing SacreBLEU..."
	export PATH="$(pwd)/.local/bin:${PATH}"
	python -m pip install sacrebleu --prefix .local
else
	SacreArgs=""
fi

if [ "$UpdateCode" = "true" ]; then
	OldPwd=$(pwd)
	set -x
	cd ${ProjectDir}
	git pull
	cd ${OldPwd}
	set +x
fi
		
echo "Scoring task ${CkptDir} on subset '${GenSubset}' and SacreBLEU subset '${SacreBLEUTestSet}'..." | tee -a ${LogFilename}
for ckpt in $(ls ${FullSaveDir}); do
	echo "Scoring checkpoint ${ckpt}..." | tee -a ${LogFilename}
	set -x
	# Generate:
	python ${ProjectDir}/generate.py ${DataDir}/${Dataset} \
		${FP16Args} \
		--gen-subset ${GenSubset} \
	    --path ${FullSaveDir}/${ckpt} \
	    --batch-size 128 \
	    --beam ${BeamSize} \
	    --lenpen ${LenPen} \
	    --remove-bpe \
	    --quiet --source-lang ${SrcLan}  --target-lang ${TgtLan} ${R2LArgs} \
	    ${SacreArgs} \
	    ${CapOut} \
	    2>&1 | tee -a ${LogFilename}
	set +x
	if [[ "${Sacre}" = "true" ]]; then
		echo "Running SacreBLEU..." | tee -a ${LogFilename}
		echo "The translated file contains $(wc -l ${TmpFile}) lines."

		set -x
		cat ${TmpFile} | \
			perl ${Detokenizer} -l ${TgtLan} | \
			sacrebleu -t ${SacreBLEUTestSet} -l ${SrcLan}-${TgtLan} -w 2 | \
			tee -a ${LogFilename}
		set +x
	fi
	echo "Scoring checkpoint ${ckpt} done." | tee -a ${LogFilename}
	rm -v ${TmpFile}
done
