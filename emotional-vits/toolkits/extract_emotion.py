# ecoding: utf-8

import os, sys
import numpy as np
import librosa
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


# load model from hub
device = 'cuda' if torch.cuda.is_available() else "cpu"
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)


def process_func(
        x: np.ndarray,
        sampling_rate: int,
        embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device).view(1, -1)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy().astype(np.float32)

    return y


def extract_emotion(wavfn, outdir):
    wav, sr = librosa.load(wavfn, sr=16000)
    wav /= max(abs(wav))
    emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
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
