#!/bin/bash

. ./path.sh || return 1;

# data related setting
data_dir=""
work_dir=""
spkid=""
mapid=""

# parse argv
. ./parse_options.sh || return 1;
set -euo pipefail

# check argv
if [ "x${spkid}" == "x" ]; then
    echo "pre_data.sh::ERROR: You must tell me spkid=?"
    return 1;
fi
if [ "x${mapid}" == "x" ]; then
    echo "pre_data.sh::ERROR: You must tell me mapid=?"
    return 1;
fi

# check dataset
echo ""
echo "pre_data.sh::Check dataset, spkid=${spkid}"
datdir="${data_dir}/${spkid}"
if [ ! -d "$datdir" ]; then
    echo "pre_data.sh::ERROR: There is no data directory, spkid=${spkid}"
    return 1;
elif [ $(find "$datdir" -name "*.wav" | wc -l) -le 0 ]; then
    echo "pre_data.sh::ERROR: There is no enough wav file to training, spkid=${spkid}"
    return 1;
fi

# make dataset
echo ""
echo "pre_data.sh::Make dataset, spkid=${spkid}"
wrkdir="${work_dir}/${spkid}"
[ ! -e "${wrkdir}" ] && mkdir -p "${wrkdir}"
rm -rf ${wrkdir}/* >/dev/null 2>&1

# denoise
echo denoise8k-infer --dumpdir $datdir --outdir $wrkdir --trim-silence
denoise8k-infer --dumpdir $datdir --outdir $wrkdir --trim-silence

# extract and cluster emotion
scpfn=${mapid}.filist
find $wrkdir -name "*.wav" > $scpfn
rm -f $wrkdir/*.emo >/dev/null 2>&1
echo python3 toolkits/extract_emotion.py --scp $scpfn --outdir $wrkdir
python3 toolkits/extract_emotion.py --scp $scpfn --outdir $wrkdir
find $wrkdir -name "*.emo" > $scpfn
echo python3 toolkits/cluster_emotion.py 3 $scpfn $wrkdir/emo.cluster
python3 toolkits/cluster_emotion.py 3 $scpfn $wrkdir/emo.cluster
rm -f $scpfn

# parse text
train_txt="${wrkdir}/${spkid}.txt"
for wavfn in $datdir/*.wav; do
    txtfn=$datdir/`basename $wavfn .wav`.txt;
    if [ ! -f $txtfn ]; then
        echo "pre_data.sh::ERROR: There is no text file=[$txtfn], wave file counterpart!"
        return 1;
    fi
    perl -e 'use File::Basename; $fn=$ARGV[0]; print basename($fn, ".txt") ," "; open F,$fn; $_=<F>; s/^\s+|\s+$//g; print "$_\n"; close F;' $txtfn; 
done > ${train_txt}
text-parser 0 $wrkdir < ${train_txt}

# generate filelist
train_scp="${wrkdir}/${spkid}.scp"
for wavfn in $wrkdir/*.wav; do
    base=`basename $wavfn .wav`;
    vecfn="$wrkdir/$base.vec192";
    emofn="$wrkdir/$base.emo";
    if [ -f "$vecfn" -a -f "$emofn" ]; then
        echo "$vecfn|$wavfn|$emofn|$mapid"
    fi
done > "${train_scp}"

echo "pre_data.sh::Okay, data of ${spkid} is ready."
echo ""
