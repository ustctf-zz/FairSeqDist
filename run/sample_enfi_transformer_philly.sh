#!/usr/bin/env bash

# FIXME: Hard code here, assume that the training script in `.../ProjectRoot/ignored_scripts/` (depth = 1)
ProjectDir=$(readlink -f $(dirname ${PHILLY_CONFIG_FILE})/../ )
OutputDir=${PHILLY_INPUT_DIRECTORY}
DataDir=/hdfs/${PHILLY_VC}/fetia/Data
SaveDir=/hdfs/${PHILLY_VC}/fetia/Ckpts
LogDir=/hdfs/${PHILLY_VC}/fetia/Log

Score=false

BeamSize=5
LenPen=1.0
FP16=false
BatchSize=1024

Dataset=wmt19.bt1.tokenized.en-fi.joined
CkptDir=wmt19.bt1.tokenized.en-fi.joined_transformer_vaswani_wmt_en_de_big_dp0.3_seed2305_maxtok4096_lr0.0005_SI1_enc6_dec6_Dist2x4_uf4_1.0
Ckpt=checkpoint20.pt
SourceDirPrefix=wmt19.train.mono.final_30000000
SourceFilename=part.aa
RemoveBPEArgs=""
src_l=en
tgt_l=fi

while [ "$1" != "" ]; do
	case $1 in
		--cd | --ckpt-dir )
			shift
			CkptDir=$1
			;;
		-C | --ckpt )
			shift
			Ckpt=checkpoint${1}.pt
			;;
		-D | --dataset )
			shift
			Dataset=$1
			;;
		-b | --beam )
			shift
			BeamSize=$1
			;;
		--bs | --batch-size | --max-tokens )
			shift
			BatchSize=$1
			;;
		--alpha | --lenpen )
			shift
			LenPen=$1
			;;
		--sdp | --src-dir-prefix )
			shift
			SourceDirPrefix=$1
			;;
		-S | --score )
			Score=true
			;;
		--remove_bpe )
			RemoveBPEArgs=" --remove-bpe "
			;;
		--src_lan )
			shift
			src_l=$1
			;;
		--tgt_lan )
			shift
			tgt_l=$1
			;;
		--src | --source )
			shift
			SourceFilename=$1
			;;
		* )
			;;
	esac
	shift
done


SourceDir=${SourceDirPrefix}.${src_l}
FullSourceFile=${DataDir}/${SourceDir}/${SourceFilename}
LogFilename=${LogDir}/${CkptDir}-${Ckpt}-${SourceFilename}.log.txt
OutputDir=${SaveDir}/${CkptDir}-${Ckpt}-sample/${SourceDir}
FullOutputFile=${OutputDir}/translated.${SourceFilename}.${tgt_l}

mkdir -pv ${OutputDir}

FullSavePath=${SaveDir}/${CkptDir}/${Ckpt}

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

echo "Sampling checkpoint ${CkptDir}/${Ckpt} on monolingual data ${FullSourceFile}, output save to ${FullOutputFile}..." | tee -a ${LogFilename}
set -x
# Generate:
# [NOTE]: Does NOT remove bpe.
python ${ProjectDir}/generate_v2.py ${DataDir}/${Dataset} \
	${FP16Args} \
    --path ${FullSavePath} \
    --max-tokens ${BatchSize} \
    --beam ${BeamSize} \
    --nbest ${BeamSize} \
    --lenpen ${LenPen} \
    --source-lang ${src_l} --target-lang ${tgt_l} \
    --quiet \
    --decode-source-file ${FullSourceFile} \
	--decode-output-file ${FullOutputFile} \
	--skip-invalid-size-inputs-valid-test  ${RemoveBPEArgs}\
	--decode-to-file
    2>&1 | tee -a ${LogFilename}
set +x
echo "Sampling checkpoint ${CkptDir}/${Ckpt} done." | tee -a ${LogFilename}


# Detect language automatically.
# l=$(python -c "s='${Dataset}';i=s.index('-');print(s[i-2:i+3])")
# src_l=$(python -c "print('${l}'.split('-')[0])")
# tgt_l=$(python -c "print('${l}'.split('-')[1])")

if [ "$Score" = "true" ]; then
	echo "Installing SacreBLEU..."
	export PATH="$(pwd)/.local/bin:${PATH}"
	python -m pip install sacrebleu --prefix .local
	
	year=$(python -c "print('${SourceFilename}'.split('.')[0][-2:])")
	echo "Running SacreBLEU..." | tee -a ${LogFilename}
	echo "The translated file contains $(wc -l ${FullOutputFile}) lines."
	Detokenizer=${ProjectDir}/../mosesdecoder/scripts/tokenizer/detokenizer.perl
	set -x
	cat ${FullOutputFile} | \
		perl ${Detokenizer} -l ${tgt_l} | \
		sacrebleu -t wmt${year} -l ${src_l}-${tgt_l} -w 2 
	set +x
fi

