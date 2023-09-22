# coding: utf-8

import os, sys
import argparse
import logging
import glob
import numpy as np
import torch

import utils
import commons
from models import SynthesizerTrn, MultiPeriodDiscriminator


def find_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return f_list


def load_model(checkpoint, hps=None, *, greedy=False, is_dis=False):
    # load config if not provided
    if hps is None:
        dirname = checkpoint if os.path.isdir(checkpoint) else \
            os.path.dirname(checkpoint)
        config_path = os.path.join(dirname, "config.json")
        hps = utils.get_hparams_from_file(config_path)

    # get model
    if not is_dis:
        model = SynthesizerTrn(
            hps.data.text_channels,
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
    else:
        model = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    
    # load parameters
    ckpt_paths = [checkpoint] if not os.path.isdir(checkpoint) else \
        find_checkpoint_path(checkpoint, "G_*.pth" if not is_dis else "D_*.pth")
    avg = torch.load(ckpt_paths[-1], map_location="cpu")['model']
    if greedy and len(ckpt_paths) > 1:
        for ckpt in ckpt_paths[:-1]:
            logging.info(f"Load [{ckpt}] for averaging.")
            states = torch.load(ckpt, map_location="cpu")['model']
            for k in avg.keys():
                avg[k] += states[k]
        for k in avg.keys():
            avg[k] = torch.true_divide(avg[k], len(ckpt_paths))
    model.load_state_dict(avg)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export Neural TTS model (See detail in export.py).")
    parser.add_argument("--outdir", "-o", type=str, required=True,
                        help="directory to save checkpoints, filename is `checkpoint.pth`.")
    parser.add_argument("--checkpoint", "--ckpt", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", "--conf", default=None, type=str,
                        help="yaml format configuration file.")
    parser.add_argument("--discriminator", "--dis", "-d", action='store_true',
                        help="export discriminator if setting true, default is generator.")
    parser.add_argument("--init-spk-embed", action='store_true',
                        help="initialize speaker embedding, default not.")
    parser.add_argument("--greedy-soup", "--greedy", action='store_true',
                        help="use average of checkpoints.")
    parser.add_argument("--convert", "-c", default=0, type=int,
                        help="convert to TorchScript or ONNX for generator if setting 1/2.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # load config if not provided
    if args.config is None:
        dirname = args.checkpoint if os.path.isdir(args.checkpoint) else \
            os.path.dirname(args.checkpoint)
        config_path = os.path.join(dirname, "config.json")
    else:
        config_path = args.config
    hps = utils.get_hparams_from_file(config_path)

    # load model
    model = load_model(args.checkpoint, hps, greedy=args.greedy_soup, is_dis=args.discriminator)
    
    # initialize speaker embedding
    if args.init_spk_embed and not args.discriminator:
        model.emb_g.weight.data.normal_()
    
    # print parameters
    logging.info(model)
    total_params, trainable_params, nontrainable_params = 0, 0, 0
    for name, param in model.named_parameters():
        if 'enc_q.' in name or '.weight_g' in name:
            continue
        num_params = np.prod(param.size())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            nontrainable_params += num_params
        if 'alpha' in name:
            logging.info(f"{name} = {param.data}")
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Non-trainable parameters: {nontrainable_params}\n")

    # save config
    config_save_path = os.path.join(args.outdir, "config.json")
    with open(config_path, 'r') as f:
        data = f.read()
    with open(config_save_path, 'w') as f:
        f.write(data)

    # save model to outdir
    checkpoint_path = os.path.join(args.outdir, "checkpoint.pth")
    state_dict = {
        "model": model.state_dict()
    }
    torch.save(state_dict, checkpoint_path)
    logging.info(f"Successfully export generator parameters from [{args.checkpoint}] to [{checkpoint_path}].")

    if args.convert == 0 or args.discriminator:
        return
    
    # export TorchScript
    generator = model.eval()
    if hasattr(generator, "remove_weight_norm"):
        generator.remove_weight_norm()
    
    # convert to torch script
    batch_size = 1
    dummy_input1 = [
        torch.randn(batch_size, 48, hps.data.text_channels, dtype=torch.float32),  # text
        torch.randn(batch_size, 1024, dtype=torch.float32),  # emo
        torch.ones(batch_size, dtype=torch.long),  # sid
    ]
    dur = torch.tensor([3, 4, 5], dtype=torch.long).unsqueeze(0).unsqueeze(0)
    x_length = dur.size(-1)
    y_length = int(torch.sum(dur))
    dummy_input2 = [
        commons.infer_path(dur, x_length, y_length),  # attn
        torch.randn(batch_size, hps.model.inter_channels, x_length, dtype=torch.float32),  # m_p
        torch.randn(batch_size, hps.model.inter_channels, x_length, dtype=torch.float32),  # s_p
        torch.ones(batch_size, hps.model.gin_channels, dtype=torch.float32),  # g
        torch.ones(batch_size, hps.model.inter_channels, y_length, dtype=torch.float32),  # noise
    ]
    
    generator.forward = generator.infer_p1
    traced1 = torch.jit.trace(generator, dummy_input1, check_trace=True)
    script_path1 =  os.path.join(args.outdir, "model_p1.pth")
    torch.jit.save(traced1, script_path1)
    logging.info(f"Successfully convert part1 to torch script: [{script_path1}]\n\n")

    generator.forward = generator.infer_p2
    traced2 = torch.jit.trace(generator, dummy_input2, check_trace=True)
    script_path2 =  os.path.join(args.outdir, "model_p2.pth")
    torch.jit.save(traced2, script_path2)
    logging.info(f"Successfully convert part2 to torch script: [{script_path2}]\n\n")

    if args.convert == 1:
        return

    # export ONNX
    input_names1 = ["input_text", "input_emo", "input_g"]
    input_names2 = ["input_attn", "input_m_p", "input_s_p", "input_g", "input_noise"]
    output_names1 = ["output_m_p", "output_s_p", "output_logw", "output_g"]
    output_names2 = ["output_wav"]
    
    onnx_path1 = os.path.join(args.outdir, "model_p1.onnx")
    torch.onnx.export(
        traced1, dummy_input1, onnx_path1,
        input_names=input_names1,
        output_names=output_names1,
        dynamic_axes={"input_text": [1]},
        verbose=args.verbose >= 1,
        opset_version=17,
    )
    logging.info(f"Successfully convert part1 to onnx: [{onnx_path1}].\n\n")

    onnx_path2 = os.path.join(args.outdir, "model_p2.onnx")
    torch.onnx.export(
        traced2, dummy_input2, onnx_path2,
        input_names=input_names2,
        output_names=output_names2,
        dynamic_axes={"input_attn": [1,2],
                      "input_m_p": [1],
                      "input_s_p": [1],
                      "input_noise": [2]},
        verbose=args.verbose >= 1,
        opset_version=17,
    )
    logging.info(f"Successfully convert part2 to onnx: [{onnx_path2}].\n\n")

if __name__ == "__main__":

    main()
