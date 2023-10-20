# coding: utf-8

import os, sys
import torch
import torch.nn.functional as F
import numpy as np

import utils
from export import load_model
from commons import infer_path

class EmoVITS(object):
    def __init__(self, checkpoint_path=None, device=None, *, loglv=0):
        self.loglv = loglv

        if checkpoint_path is None:
            checkpoint_path = os.path.dirname(os.path.abspath(__file__)) + "/checkpoint/checkpoint.pth"
        self.res_root_path = os.path.dirname(checkpoint_path)

        if self.loglv > 0:
            func_name = f"{self.__class__.__name__}::{sys._getframe().f_code.co_name}"
            sys.stdout.write(f"{func_name}: checkpoint path={checkpoint_path}\n")

        # setup config
        config_path = os.path.join(self.res_root_path, "config.json")
        hps = utils.get_hparams_from_file(config_path)

        self.sampling_rate = hps.data['sampling_rate']
        self.hop_size = hps.data['hop_length']
        self.text_channels = hps.data['text_channels']
        self.inter_channels = hps.model['inter_channels']
        self.num_speaker = hps.data['n_speakers']
        self.noise_scale = hps.data['noise_scale']

        # setup speaker id mapping
        self.spkid_mapping = dict()
        self.spkid_mapping_mtime = dict()
        for map_path in utils.find_files(self.res_root_path, "*.map"):
            if self.loglv > 0:
                sys.stdout.write(f"{func_name}: load speaker id mapping from {map_path}\n")
            self._load_spkid_mapping(map_path)

        # setup emotion embedding
        self.spk_emo_embed = dict()
        self.spk_emo_embed_mtime = dict()
        for emo_path in utils.find_files(self.res_root_path, "*.emo"):
            spkid = int(os.path.splitext(os.path.basename(emo_path))[0])
            if self.loglv > 0:
                sys.stdout.write(f"{func_name}: load emotion of {spkid} from {emo_path}\n")
            self._load_spk_emo_embed(spkid)

        # setup device
        if device is not None:
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # setup model
        model = load_model(checkpoint_path, hps)
        model.remove_weight_norm()
        self.model = model.half().eval().to(self.device)

        # noise buffer
        self.noise = (torch.randn(1 * self.inter_channels * 4096) * self.noise_scale).half()

        # alias inference
        self.inference = self.infer

        if self.loglv > 0:
            sys.stderr.write(f"{func_name}: Successful !\n")
    
    def _load_spkid_mapping(self, mapfn):
        if not os.path.exists(mapfn):
            return
        with open(mapfn, 'rt') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 or line[0] == '#':
                    continue
                arr = line.split()
                if len(arr) != 2 and not (arr[0].isdigit() and arr[1].isdigit()):
                    continue
                self.spkid_mapping[int(arr[0])] = int(arr[1])
        self.spkid_mapping_mtime[mapfn] = int(os.stat(mapfn).st_mtime)

    def _get_spk_emo_embed(self, emo: tuple):
        # emo: (spkid|ndarray, eid)
        if isinstance(emo[0], int):
            emo_embed = self.spk_emo_embed.get(emo[0])
            if emo_embed is None:
                emo_embed = self._load_spk_emo_embed(emo[0])
            assert emo_embed is not None
        elif isinstance(emo[0], np.ndarray):
            emo_embed = torch.from_numpy(emo[0].reshape(-1, 1024).astype(np.float32)).half()
        else:
            raise ValueError("emo[0] must be int or ndarray")

        eid = -1 if len(emo) == 1 else int(emo[1])
        if eid < 0 or eid > emo_embed.size(0):
            eid = np.random.randint(0, emo_embed.size(0))
        return emo_embed[eid]

    def _load_spk_emo_embed(self, spkid: int):
        emo_path = os.path.join(self.res_root_path, f"{spkid}.emo")
        if os.path.exists(emo_path):
            emo_embed = np.fromfile(emo_path, dtype=np.float32).reshape(-1, 1024)
            emo_embed = torch.from_numpy(emo_embed).half()
            self.spk_emo_embed[spkid] = emo_embed
            self.spk_emo_embed_mtime[emo_path] = int(os.stat(emo_path).st_mtime)
            return emo_embed
        return None

    def update(self):
        for map_path in self.spkid_mapping_mtime.keys():
            if not os.path.exists(map_path):
                self.spkid_mapping_mtime.pop(map_path)
                continue
            if int(os.stat(map_path).st_mtime) == self.spkid_mapping_mtime[map_path]:
                continue
            self._load_spkid_mapping(map_path)
        for emo_path in self.spk_emo_embed_mtime.keys():
            if not os.path.exists(emo_path):
                self.spk_emo_embed_mtime.pop(emo_path)
                continue
            if int(os.stat(emo_path).st_mtime) == self.spk_emo_embed_mtime[emo_path]:
                continue
            spkid = int(os.path.splitext(os.path.basename(emo_path))[0])
            self._load_spk_emo_embed(spkid)

    @torch.no_grad()
    def infer(self, spkid, text, emo, *, duration_rate=1.0):
        # spkid: int
        # text: (N,c)
        # emo: tuple(int|ndarray, int) | None
        # duration_rate: float
        x_length = text.shape[0]
        batch_size = 1

        spkid = self.spkid_mapping.get(spkid, spkid)
        assert spkid < self.num_speaker, f"spkid={spkid} must be less than {self.num_speaker}"
        sid = torch.tensor([spkid], dtype=torch.long)

        # get emotion embedding
        if isinstance(emo, torch.Tensor):
            if emo.dtype is not torch.float16:
                emo = emo.half()
        else:
            if emo is None:
                emo = (spkid, -1)
            if isinstance(emo[0], int):
                emo = tuple([self.spkid_mapping.get(emo[0], emo[0]) if emo[0] != 0 else spkid, -1 if len(emo) == 1 else emo[1]])
            emo = self._get_spk_emo_embed(emo)
            emo = emo.unsqueeze(0)

        # prepare input part1
        text = torch.from_numpy(text).half().to(self.device).unsqueeze(0)
        emo = emo.to(self.device)
        sid = sid.to(self.device)

        # inference part1
        m_p, s_p, logw, g = self.model.infer_p1(text, emo, sid)

        # prepare input part2
        w = torch.exp(logw) * duration_rate
        w_ceil = torch.ceil(w)
        y_length = int(torch.clamp_min(torch.sum(w_ceil), 1).item())
        nl = batch_size * self.inter_channels * y_length
        start = np.random.randint(self.noise.size(0) - nl)
        noise = self.noise[start:start+nl].view(
            batch_size, self.inter_channels, y_length).to(self.device)
        attn = infer_path(w_ceil, x_length, y_length).half().to(self.device)

        # inference part2
        wav = self.model.infer_p2(attn, m_p, s_p, g, noise)

        # post result
        wav = wav.float().view(-1).cpu().numpy()

        return wav, emo


def main():

    import argparse
    import logging
    import time
    import soundfile as sf

    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Neural TTS Generator "
                    "(See detail in VITS/emotional-vits/infer.py).")
    parser.add_argument("--scpfn", "--scp", type=str, required=True,
                        help="feats.scp file. ")
    parser.add_argument("--spkid", "--sid", default=None, type=int,
                        help="speaker Id. (default=1)")
    parser.add_argument("--emotion", "--emo", default=None, type=str,
                        help="speaker Id or emotion file path. format: (spkid|path, eid), "
                             "which `path` is emotion embedding, `eid` is index. (default=1:0)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", "--ckpt", default=None, type=str,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", "--conf", default=None, type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # setup model
    model = EmoVITS(args.checkpoint, args.config, loglv=1)

    # get feature files
    features = dict()
    with open(args.scpfn) as fid:
        for line in fid:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            lines = line.split('|')
            utt_id = os.path.splitext(os.path.basename(lines[0]))[0]
            if args.spkid is not None:
                spkid = int(args.spkid)
            elif len(lines) > 1:
                spkid = int(lines[-1])
            else:
                spkid = 1
            if args.emotion is not None:
                emo = args.emotion.split(':')
            elif len(lines) > 2:
                emo = lines[1].split(':')
            else:
                emo = None
            if emo is not None:
                if isinstance(emo[0], str) and os.path.exists(emo[0]):
                    emo[0] = np.fromfile(emo[0], dtype=np.float32).reshape(-1, 1024)
                else:
                    emo[0] = int(emo[0])
                if len(emo) == 1:
                    emo.append(-1)
                else:
                    emo[1] = int(emo[1])
                emo = tuple(emo)
            features[utt_id] = (spkid, emo, lines[0])
    logging.info(f"The number of features to be decoded = {len(features)}.")

    # start generation
    total_rtf = 0.0
    for idx, (utt_id, (spkid, emo, vecfn)) in enumerate(features.items(), 1):
        start = time.time()

        # load feature
        text = np.fromfile(vecfn, dtype=np.float32).reshape(-1, model.text_channels)

        # inference
        wav, _ = model.infer(spkid, text, emo)  # (1,t)

        # save audio
        sf.write(os.path.join(args.outdir, f"{utt_id}.wav"),
            wav, model.sampling_rate, "PCM_16")

        rtf = (time.time() - start) / (len(wav) / model.sampling_rate)
        total_rtf += rtf

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
