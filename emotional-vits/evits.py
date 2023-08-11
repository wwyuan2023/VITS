# coding: utf-8

import os
import sys
import struct
import time
import numpy as np
import torch
from librosa import resample

from textparser import TextParser
from infer import EVITS


def _genWavHeader(sampleNum, sampleRate=8000, bitNum=16):
    wavHeadInfo = b'\x52\x49\x46\x46'  # RIFF
    wavHeadInfo += struct.pack('i', sampleNum * 2 + 44 - 8)
    wavHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00'
    wavHeadInfo += struct.pack('i', sampleRate)
    wavHeadInfo += struct.pack('i', sampleRate * bitNum // 8)
    wavHeadInfo += struct.pack('H', bitNum // 8)
    wavHeadInfo += struct.pack('H', bitNum)
    wavHeadInfo += b'\x64\x61\x74\x61'
    wavHeadInfo += struct.pack('i', sampleNum * 2)
    return wavHeadInfo


class DialTTS(object):

    # global configuration
    default_spkid = 678
    default_volume = 1.0
    default_speed = 1.0
    default_pitch = 1.0

    def __init__(
        self,
        ckpt_path: str = None,
        device: torch.device = None,
        loglv : int = 0,
    ) -> None:
        self.loglv = loglv

        self.textparser = TextParser(loglv=loglv)
        self.speecher = EVITS(ckpt_path, device=device)
        
        self.sampling_rate = self.speecher.sampling_rate
        self.max_utt_length = self.textparser.max_utt_length

        if self.loglv > 0:
            func_name = f"{self.__class__.__name__}::{sys._getframe().f_code.co_name}"
            sys.stderr.write(f"{func_name}: Successful !\n")
    
    def update(self):
        self.textparser.update()
        self.speecher.update()

        # empty cuda buffer
        torch.cuda.empty_cache()
    
    def _parse_input(self, inputs):

        volume = float(inputs.get('volume', self.default_volume))
        speed = float(inputs.get('speed', self.default_speed))
        pitch = float(inputs.get('pitch', self.default_pitch))

        volume = max(0., min(1., volume))
        speed = max(0.5, min(2., speed))
        pitch = max(0.5, min(2., pitch))

        speed /= pitch

        utt_id = inputs.get('id', str(time.time()).replace('.', '_'))
        utt_text = inputs.get('text', '。')
        spkid = int(inputs.get('spkid', self.default_spkid))
        emotion = inputs.get('emotion')

        return inputs, utt_id, utt_text, spkid, volume, speed, pitch, emotion

    def _handle_outputs(self, inputs, wav_bytes, segtext_str, time_used_frontend, time_used_backend, rtf):
        outputs = inputs
        outputs['wav'] = _genWavHeader(len(wav_bytes)//2, self.sampling_rate, 16) + wav_bytes
        outputs['segtext'] = segtext_str
        outputs['time_used_frontend'] = time_used_frontend * 1000  # ms
        outputs['time_used_backend'] = time_used_backend * 1000  # ms
        outputs['rtf'] = rtf
        return outputs
    
    def _split_utt_text(self, utt_id, utt_text):
        if utt_text is None or utt_text == '':
            utt_text = '。'
        utt_text = utt_text.strip()

        if len(utt_text) <= self.max_utt_length:
            return [utt_id], [utt_text]
        
        center_pos = int(self.max_utt_length * 0.618)
        max_length = self.max_utt_length
        def find_nearest_center(texts):
            if len(texts) < center_pos + 2: return len(texts)
            if len(texts) >= center_pos + 2 and texts[center_pos:center_pos+2] in ['——', '……']:
                return center_pos + 2
            for _chr in ['。', '！', '!', '？', '?', '；', ';', '，']:
                find_pos = texts[:center_pos][::-1].find(_chr)
                if 0 <= find_pos < center_pos:
                    return center_pos - find_pos
                find_pos = texts.find(_chr, center_pos)
                if 0 <= find_pos < max_length:
                    return find_pos + len(_chr)
            for _chr in ['.', ',', ':', '：']:
                texts_reverse = texts[:center_pos][::-1]
                cl = len(_chr)
                find_pos = texts_reverse.find(_chr)
                if (0 <= find_pos < center_pos
                    and (find_pos - cl >= 0 and not (texts_reverse[find_pos-cl].isdigit()))
                    and (find_pos + cl < center_pos and not (texts_reverse[find_pos+cl].isdigit()))
                ):
                    return center_pos - find_pos
                find_pos = texts.find(_chr, center_pos)
                if (0 <= find_pos < max_length
                    and (find_pos - cl >= 0 and not (texts[find_pos-cl].isdigit()))
                    and (find_pos + cl < len(texts) and not (texts[find_pos+cl].isdigit()))
                ):
                    return find_pos + cl
            for _chr in ['——', '……', '、', '（', '）', '(', ')', '[', ']', '【', '】']:
                find_pos = texts[:center_pos][::-1].find(_chr)
                if 0 <= find_pos < center_pos:
                    return center_pos - find_pos
                find_pos = texts.find(_chr, center_pos)
                if 0 <= find_pos < max_length:
                    return find_pos + len(_chr)
            for _chr in ['~', ' ', '\t']:
                find_pos = texts[:center_pos][::-1].find(_chr)
                if 0 <= find_pos < center_pos:
                    return center_pos - find_pos
                find_pos = texts.find(_chr, center_pos)
                if 0 <= find_pos < max_length:
                    return find_pos + len(_chr)
            return min(len(texts), max_length)
        
        batch_utt_id, batch_utt_text = [], []
        i = 0
        while len(utt_text) > 0:
            pos = find_nearest_center(utt_text)
            if pos > self.max_utt_length:
                pos = self.max_utt_length - 1
                batch_utt_text.append(utt_text[:pos] + "，")
            else:
                batch_utt_text.append(utt_text[:pos])
            batch_utt_id.append(f"{utt_id}-{i}")
            utt_text = utt_text[pos:]
            i += 1
        
        return batch_utt_id, batch_utt_text

    @torch.no_grad()
    def speaking(self, inputs : dict) -> dict:
        inputs, utt_id, utt_text, spkid, volume, speed, pitch, emotion = \
            self._parse_input(inputs)
        
        batch_utt_id, batch_utt_text = self._split_utt_text(utt_id, utt_text)
        batch_segtext, batch_wavlen = "", 0
        batch_wav = []
        time_used_frontend, time_used_backend = 0, 0
        for idx, (utt_id, utt_text) in enumerate(zip(batch_utt_id, batch_utt_text), 1):
            start = time.time()
            utt_id, utt_segtext, utt_vector = self.textparser(utt_id, utt_text)
            batch_segtext += utt_segtext.printer()
            end = time.time()
            time_used_frontend += end - start

            start = end
            wav, emotion = self.speecher.infer(spkid, utt_vector, emotion, duration_rate=speed)
            batch_wavlen += len(wav)
            if pitch != 1.0:
                wav = resample(wav, orig_sr=int(self.sampling_rate/pitch), target_sr=self.sampling_rate)
            wav = np.clip(wav * volume * 32767, -32768, 32767).astype(np.int16)
            batch_wav.append(wav)
            end = time.time()
            time_used_backend += end - start
        
        rtf = (time_used_frontend + time_used_backend) / (batch_wavlen / self.sampling_rate)
        batch_wav_bytes = bytes()
        for idx, wav in enumerate(batch_wav, 1):
            batch_wav_bytes += wav.tobytes()

        outputs = self._handle_outputs(
            inputs, batch_wav_bytes, batch_segtext, time_used_frontend, time_used_backend, rtf)
        return outputs



if __name__ == "__main__":

    mytts = DialTTS(device='cuda', loglv=1)
    
    # spkid, volume, speed, pitch, emotion, suffix, text
    batch_inputs = [
        [678, 1.0, 1.0, 1.0, (678, 0), "", "这是一个测试用例1"],
        [678, 1.0, 1.0, 1.0, (678, 0), "", "这是一个测试用例2"],
    ]

    for idx, batch in enumerate(batch_inputs, 1):
        inputs = {
            "spkid": batch[0],
            "volume": batch[1],
            "speed": batch[2],
            "pitch": batch[3],
            "emotion": batch[4],
            "text": batch[-1],
        }
        outputs = mytts.speaking(inputs)
        wav = outputs.pop('wav')
        suffix = "" if len(batch[5]) == 0 else f"-{batch[5]}"
        outfn = os.path.join('output', f"{idx:04d}{suffix}.wav")
        with open(outfn, 'wb') as f:
            f.write(wav)
        segtext = outputs.pop('segtext')
        print(f"input={batch}, wav={outfn}, segtext={segtext}")

        # spkid == 1
        # Neutral 0
        # Happy 7
        # Angry 14
        # Sad 21
        # Fear 28
        # Surprise 35
        # Disgusting 42
        
