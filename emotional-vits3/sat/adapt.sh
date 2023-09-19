#!/bin/bash

. ./path.sh || exit 1;

# recording data directory
data_dir="$(pwd)/data"
work_dir="$(pwd)/work"

# training related setting
ckptG="./pretrain/G_0.pth" # generator checkpoint path to load pretrained parameters
ckptD="./pretrain/D_0.pth" # discriminator checkpoint path to load pretrained parameters
outdir="../checkpoint/"    # export generator checkpoint and emo to this

# parse argv
. ./parse_options.sh || exit 1;
set -euo pipefail

# check argv
if [ "x${outdir}" == "x" ]; then
    echo "ERROR: You must set outdir"
    exit 1;
fi
[ ! -e "${outdir}" ] && mkdir -p ${outdir}

# check dataset
echo ""
num_spk=$(ls "${data_dir}/" | wc -l)
echo "Check dataset, number of speaker = ${num_spk}"
if [ $num_spk -eq 0 ]; then
    echo "ERROR: there is no any speaker record in [$data_dir]"
    exit 1;
fi

# make dataset for each speaker
echo ""
echo "Make dataset for all speaker"
echo "Clean all work directory"
[ -e "${work_dir}" ] && rm -rf ${work_dir}/* >/dev/null 2>&1
mapid=1024
declare -A spk_array
for spkid in `ls "${data_dir}/"`; do
    if [ ! -d "${data_dir}/${spkid}" ]; then
        echo "IGNORE: ${spkid} is not spkid"
        continue
    fi
    num=$(find "${data_dir}/${spkid}/" -name "*.wav" -type f | wc -l)
    if [ $num -le 0 ]; then
        echo "IGNORE: the number of recording wav file of ${spkid} is ${num}, but less than 1"
        continue
    fi
    ((mapid=mapid-1));
    spk_array[$mapid]=$spkid;
    echo "sh ./pre_data.sh --data_dir "${data_dir}" --work_dir "${work_dir}" --spkid $spkid --mapid $mapid"
    sh ./pre_data.sh --data_dir "${data_dir}" --work_dir "${work_dir}" --spkid $spkid --mapid $mapid
done

# generate filelist
train_scp="${work_dir}/train.scp"
valid_scp="${work_dir}/valid.scp"
[ -e "${train_scp}" ] && rm -f ${train_scp}
[ -e "${valid_scp}" ] && rm -f ${valid_scp}
find "${work_dir}" -name "*.scp" -exec cat {} \; > "${work_dir}/~~tmp~~"
cp -f "${work_dir}/~~tmp~~" "${train_scp}";
while true; do
    if [ $(cat "${train_scp}" | wc -l) -le 50 ]; then
        cat "${work_dir}/~~tmp~~" >> "${train_scp}"
    else
        break
    fi
done
rm -f "${work_dir}/~~tmp~~" >/dev/null 2>&1
head -n 50 "${train_scp}" > "${valid_scp}";

# sat training
echo ""
model="adapt"
expdir="./logs/$model"
[ ! -e "${expdir}" ] && mkdir -p "${expdir}"
echo "Clear $expdir"
rm -rf ${expdir}/* >/dev/null 2>&1
logfn=./adapt.log
echo "Adaptive training start. See the progress via $logfn";
python3 ../train.py -m $model -c configs/adapt.json -a --ckptG $ckptG --ckptD $ckptD >$logfn 2>&1
echo "Successfully finished adaptive training.";

# export checkpoint
echo "Expport whole model parameters from [${expdir}] to [$outdir]"
i=0
for fn in `find "${expdir}" -name "G_*.pth" | perl -ne '/G\_(\d+).pth/; print "$1\n";' | sort -n -r`; do
    ((i=i+1))
    if [ $i -le 5 ]; then
        continue
    fi
    rm -f "${expdir}/G_${fn}.pth"
done
rm -f "${outdir}/checkpoint.pth" "${outdir}/config.json" >/dev/null 2>&1
python3 ../export.py --ckpt "${expdir}" --outdir "${outdir}" --greedy

# export spkid.map
echo "Export speaker ID mapping to [$outdir]"
mapfn="${outdir}/spkid.map"
for mapid in "${!spk_array[@]}"; do
    spkid=${spk_array[$mapid]};
    echo "$spkid $mapid";
done > $mapfn

# export emo
for mapid in "${!spk_array[@]}"; do
    spkid=${spk_array[$mapid]};
    echo "Export emotion of $mapid"
    rm -f "${outdir}/${mapid}.emo" "${outdir}/${spkid}.emo" >/dev/null 2>&1
    cp -f "${work_dir}/${spkid}/emo.cluster" "${outdir}/${mapid}.emo"
    ln -sf -T "${outdir}/${mapid}.emo" "${outdir}/${spkid}.emo"
done

# clean
echo ""
#echo "Clean up"
#rm -rf ${work_dir}/* >/dev/null 2>&1
#rm -rf ${expdir}/* >/dev/null 2>&1

echo "Okay, all is well."
