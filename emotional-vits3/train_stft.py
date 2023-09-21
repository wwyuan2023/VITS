# coding: utf-8

import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler
from models import SynthesizerTrn, DurationDiscriminator
from mrd import MultiWaveSTFTDiscriminator
from stft_loss import MultiResolutionSTFTLoss
from losses import generator_loss, discriminator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from radam import RAdam


torch.backends.cudnn.benchmark = True
global_step = 0
use_dur_dis = False


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29795'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
    global global_step, use_dur_dis
    use_dur_dis = hps.use_dur_dis
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,300,400,500,600,700,800,900,1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, 
        pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
        eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
            batch_size=hps.train.batch_size, pin_memory=True,
            drop_last=False, collate_fn=collate_fn)
        logger.info(f"Load train files = {len(train_dataset)}")
        logger.info(f"Load valid files = {len(eval_dataset)}")

    net_g = SynthesizerTrn(
        hps.data.text_channels,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        align_noise=hps.train.align_noise,
        align_noise_decay=hps.train.align_noise_decay,
        **hps.model
    ).cuda(rank)
    net_d = MultiWaveSTFTDiscriminator().cuda(rank)
    net_p = DurationDiscriminator(hps.model.hidden_channels, 64, 5).cuda(rank) if use_dur_dis else None
    mstft_loss = MultiResolutionSTFTLoss().cuda(rank)
    
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        weight_decay=hps.train.weight_decay,
        eps=hps.train.eps
    )
    optim_d = RAdam(
        net_d.parameters(),
        1e-4,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-6
    )
    optim_p = RAdam(
        net_p.parameters(),
        1e-4,
        betas=(0.9, 0.999),
        weight_decay=0,
        eps=1e-6
    ) if use_dur_dis else None
    
    if rank == 0:
        logger.info(net_g)
        logger.info(net_d)
        total_params_g = int(sum([np.prod(p.size()) if 'enc_q.' not in n and '.weight_g' not in n else 0 for n,p in net_g.named_parameters()]))
        total_params_d = int(sum([np.prod(p.size()) for p in net_d.parameters()]))
        logger.info(f"Total parameters of Generator: {total_params_g}")
        logger.info(f"Total parameters of Discriminator: {total_params_d}")

    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])
    if use_dur_dis: net_d = DDP(net_d, device_ids=[rank])

    try:
        ckptG = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth") if hps.ckptG is None else hps.ckptG
        ckptD = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth") if hps.ckptD is None else hps.ckptD
        ckptP = utils.latest_checkpoint_path(hps.model_dir, "P_*.pth")
        _, _, epoch_str = utils.load_checkpoint(ckptG, net_g, optim_g, adapt=hps.adapt)
        _, _, epoch_str = utils.load_checkpoint(ckptD, net_d, optim_d, adapt=hps.adapt)
        if use_dur_dis and ckptP is not None:
            utils.load_checkpoint(ckptP, net_p, optim_p, adapt=hps.adapt)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=-1)
    if use_dur_dis: scheduler_p = torch.optim.lr_scheduler.ExponentialLR(optim_p, gamma=hps.train.lr_decay, last_epoch=-1)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_p, mstft_loss], [optim_g, optim_d, optim_p], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_p, mstft_loss], [optim_g, optim_d, optim_p], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()
        if use_dur_dis: scheduler_p.step()
        if (hps.adapt and global_step > hps.train.steps) or optim_g.param_groups[0]['lr'] <= 5e-6:
            break

    utils.save_checkpoint(net_g, optim_g, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
    utils.save_checkpoint(net_d, optim_d, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    if use_dur_dis: utils.save_checkpoint(net_p, optim_p, epoch, os.path.join(hps.model_dir, "P_{}.pth".format(global_step)))


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, logger, writers):
    net_g, net_d, net_p, mstft_loss = nets
    optim_g, optim_d, optim_p = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if use_dur_dis: net_p.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, emo, speakers) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        emo = emo.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q), z_q, (x_hidden, logw, logw_) = net_g(x, x_lengths, spec, spec_lengths, emo, speakers)

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice
            
            y_mel = mel_spectrogram_torch(
                y[:1].squeeze(1), 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate, 
                hps.data.hop_length, 
                hps.data.win_length, 
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat[:1].squeeze(1), 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate, 
                hps.data.hop_length, 
                hps.data.win_length, 
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )
            
            sc_loss, mag_loss, y_mag, y_hat_mag = mstft_loss(y.squeeze(1), y_hat.squeeze(1))

            # Discriminator
            y_d_hat_r = net_d(y, y_mag)
            y_d_hat_g = net_d(y_hat.detach(), [x.detach() for x in y_hat_mag])
            if use_dur_dis: d_d_hat_r, d_d_hat_g = net_p(x_hidden, x_mask, logw, logw_.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                if use_dur_dis: loss_disc_p, losses_p_r, losses_p_g = discriminator_loss(d_d_hat_r, d_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        if use_dur_dis:
            optim_p.zero_grad()
            scaler.scale(loss_disc_p).backward()
            scaler.unscale_(optim_p)
            grad_norm_p = commons.clip_grad_value_(net_p.parameters(), None)
            scaler.step(optim_p)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_g = net_d(y_hat, y_hat_mag)
            if use_dur_dis: d_d_hat_r, d_d_hat_g = net_p(x_hidden, x_mask, logw, logw_)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float()) * hps.train.c_dur
                loss_stft = (sc_loss.float() + mag_loss.float()) * hps.train.c_stft
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_kl_q = kl_loss(z_q, logs_p, m_q, logs_q, z_mask) * hps.train.c_kl_q
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_stft + loss_dur + loss_kl + loss_kl_q
                if use_dur_dis:
                    loss_gen_p, losses_gen_p = generator_loss(d_d_hat_g)
                    loss_gen_all = loss_gen_all + loss_gen_p
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank==0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_stft, loss_dur, loss_kl, loss_kl_q]
                if use_dur_dis: losses.append(loss_gen_p)
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([f"{x.item():.05f}" for x in losses] + [f"{global_step}", f"{lr:.06f}"])
                
                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/stft": loss_stft, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl, "loss/g/kl_q": loss_kl_q})
                if use_dur_dis:
                    scalar_dict.update({"loss/p/total": loss_disc_p, "grad_norm_p": grad_norm_p})
                    scalar_dict.update({"loss/p/{}".format(i): v for i, v in enumerate(losses_gen_p)})
                    scalar_dict.update({"loss/p_r/{}".format(i): v for i, v in enumerate(losses_p_r)})
                    scalar_dict.update({"loss/p_g/{}".format(i): v for i, v in enumerate(losses_p_g)})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = { 
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
                        "all/attn": utils.plot_alignment_to_numpy(attn[0].data.cpu().numpy())
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step, 
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                if use_dur_dis: utils.save_checkpoint(net_p, optim_p, epoch, os.path.join(hps.model_dir, "P_{}.pth".format(global_step)))
        global_step += 1
    
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, emo, speakers) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            emo = emo.cuda(0)
            speakers = speakers.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            emo = emo[:1]
            speakers = speakers[:1]
            break
        y_hat, attn, mask, *_ = generator.module.inference(x, x_lengths, emo, speakers, max_len=1000)
        y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

        mel = spec_to_mel_torch(
            spec[:1],
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat[:1].squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
        "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
