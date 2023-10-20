# ecoding: utf-8

import os, sys
import numpy as np
import librosa
import torch

import audeer
import audonnx

# ref to: https://github.com/audeering/w2v2-how-to


url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('~/.cache/audeer/cache')
model_root = audeer.mkdir('~/.cache/audeer/model')

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)


def extract_emotion(wavfn, outdir, embeddings=True):
    x, sr = librosa.load(wavfn, sr=16000)
    x /= max(abs(x))
    assert x.ndim == 1 # Only support mono channel
    y = model(x, sr)    
    emb = y['hidden_states'] if embeddings else y['logits']
    outfn = os.path.join(outdir, os.path.basename(wavfn).replace('.wav', '.emo'))
    emb.flatten().tofile(outfn)
    return emb


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Emotion Extraction Preprocess')
    parser.add_argument('--scp', nargs="+", type=str, required=True, help='path of the filelists or wave files')
    parser.add_argument('--outdir', type=str, required=False, help='output directory')
    args = parser.parse_args()
    
    outdir = args.outdir
    if outdir is not None and not os.path.exists(outdir):
        os.makedirs(outdir)

    wavlist = []
    for file in args.scp:
        if file[-4:].lower() == ".wav":
            wavlist.append(file)
        else:
            with open(file, 'rt') as f:
                lines = f.readlines()
            for wavfn in lines:
                wavfn = wavfn.strip()
                if wavfn != '' and wavfn[0] != '#':
                    wavlist.append(wavfn)
    
    print("-----start emotion extract-----")
    for idx, wavfn in enumerate(wavlist):
        sys.stdout.write(f"extract from {wavfn} ... ")
        outdir = os.path.dirname(wavfn) if args.outdir is None else args.outdir
        extract_emotion(wavfn, outdir)
        sys.stdout.write("done!\n")
    print("-----end emotion extract-----")
