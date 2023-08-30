# coding: utf-8

import os
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_sid, load_binfn


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) text vector, loads audio, speaker_id
        2) computes spectrograms from audio files.
    """
    def __init__(self, filepaths_sid, hparams):
        self.filepaths_sid = load_filepaths_and_sid(filepaths_sid)
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length

        self.text_channels = hparams.data.text_channels
        self.segment_size = hparams.train.segment_size
        
        self.min_text_len = getattr(hparams.data, "min_text_len", 2)
        self.max_text_len = getattr(hparams.data, "max_text_len", 384)
        self.min_wav_len = max(self.segment_size, getattr(hparams.data, "min_wav_len", 0))
        self.max_wav_len = getattr(hparams.data, "max_wav_len", 10*self.sampling_rate)

        self._filter()
        random.seed(1234)
        random.shuffle(self.filepaths_sid)

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        filepaths_sid_new = []
        lengths = []
        for vecfn, wavfn, emofn, sid in self.filepaths_sid:
            vec = load_binfn(vecfn, self.text_channels)
            wav, sr = load_wav_to_torch(wavfn)
            if self.min_text_len < len(vec) < self.max_text_len and self.min_wav_len < len(wav) < self.max_wav_len:
                filepaths_sid_new.append([vecfn, wavfn, emofn, sid])
                lengths.append(len(wav) // self.hop_length)
        self.filepaths_sid = filepaths_sid_new
        self.lengths = lengths

    def get_item(self, filepaths_sid):
        # separate filename, speaker_id
        vecfn, wavfn, emofn, sid = filepaths_sid
        vec = self.get_text(vecfn)
        spec, wav = self.get_audio(wavfn)
        emo = self.get_emo(emofn)
        sid = self.get_sid(sid)
        return (vec, spec, wav, emo, sid)

    def get_audio(self, filename):
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))
        assert len(audio_norm) >= self.segment_size
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, vecfn):
        vec = load_binfn(vecfn, self.text_channels)
        vec = torch.from_numpy(vec)
        return vec
    
    def get_emo(self, emofn):
        emo = load_binfn(emofn, 1024).flatten()  # (1024,)
        emo = torch.from_numpy(emo)
        return emo

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_item(self.filepaths_sid[index])

    def __len__(self):
        return len(self.filepaths_sid)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, emo, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.FloatTensor(len(batch), max_text_len, batch[0][0].size(1))
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        emo = torch.FloatTensor(len(batch), 1024)
        sid = torch.LongTensor(len(batch))

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        emo.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0), :] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            emo[i, :] = row[3]

            sid[i] = row[4]
          

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, emo, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, emo, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
