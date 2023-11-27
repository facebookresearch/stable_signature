# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import shutil
import tqdm
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, compute_statistics_of_path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import utils
import utils_img
import utils_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_imgs(img_dir, img_dir_nw, save_dir, num_imgs=None, mult=10):
    filenames = os.listdir(img_dir)
    filenames.sort()
    if num_imgs is not None:
        filenames = filenames[:num_imgs]
    for ii, filename in enumerate(tqdm.tqdm(filenames)):
        img_1 = Image.open(os.path.join(img_dir_nw, filename))
        img_2 = Image.open(os.path.join(img_dir, filename))
        diff = np.abs(np.asarray(img_1).astype(int) - np.asarray(img_2).astype(int)) *10
        diff = Image.fromarray(diff.astype(np.uint8))
        shutil.copy(os.path.join(img_dir_nw, filename), os.path.join(save_dir, f"{ii:02d}_nw.png"))
        shutil.copy(os.path.join(img_dir, filename), os.path.join(save_dir, f"{ii:02d}_w.png"))
        diff.save(os.path.join(save_dir, f"{ii:02d}_diff.png"))

def get_img_metric(img_dir, img_dir_nw, num_imgs=None):
    filenames = os.listdir(img_dir)
    filenames.sort()
    if num_imgs is not None:
        filenames = filenames[:num_imgs]
    log_stats = []
    for ii, filename in enumerate(tqdm.tqdm(filenames)):
        pil_img_ori = Image.open(os.path.join(img_dir_nw, filename))
        pil_img = Image.open(os.path.join(img_dir, filename))
        img_ori = np.asarray(pil_img_ori)
        img = np.asarray(pil_img)
        log_stat = {
            'filename': filename,
            'ssim': structural_similarity(img_ori, img, channel_axis=2),
            'psnr': peak_signal_noise_ratio(img_ori, img),
            'linf': np.amax(np.abs(img_ori.astype(int)-img.astype(int)))
        }
        log_stats.append(log_stat)
    return log_stats

def cached_fid(path1, path2, batch_size=32, device='cuda:0', dims=2048, num_workers=10):
    for p in [path1, path2]:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    # cache path2
    storage_path = Path.home() / f'.cache/torch/fid/{path2.replace("/", "_")}'
    if (storage_path / 'm.pt').exists():
        m2 = torch.load(storage_path / 'm.pt')
        s2 = torch.load(storage_path / 's.pt')
    else:
        storage_path.mkdir(parents=True)
        m2, s2 = compute_statistics_of_path(str(path2), model, batch_size, dims, device, num_workers)
        torch.save(m2, storage_path / 'm.pt')
        torch.save(s2, storage_path / 's.pt')
    m1, s1 = compute_statistics_of_path(str(path1), model, batch_size, dims, device, num_workers)    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

@torch.no_grad()
def get_bit_accs(img_dir: str, msg_decoder: nn.Module, key: torch.Tensor, batch_size: int = 16, attacks: dict = {}):
    # resize crop
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_loader = utils.get_dataloader(img_dir, transform, batch_size=batch_size, collate_fn=None)

    log_stats = {ii:{} for ii in range(len(data_loader.dataset))}
    for ii, imgs in enumerate(tqdm.tqdm(data_loader)):

        imgs = imgs.to(device)
        keys = key.repeat(imgs.shape[0], 1)

        for name, attack in attacks.items():
            imgs_aug = attack(imgs)
            decoded = msg_decoder(imgs_aug) # b c h w -> b k
            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_accs = (bit_accs == 1) # b
            for jj in range(bit_accs.shape[0]):
                img_num = ii*batch_size+jj
                log_stat = log_stats[img_num]
                log_stat[f'bit_acc_{name}'] = bit_accs[jj].item()
                log_stat[f'word_acc_{name}'] = 1.0 if word_accs[jj].item() else 0.0

    log_stats = [{'img': img_num, **log_stats[img_num]} for img_num in range(len(data_loader.dataset))]
    return log_stats

@torch.no_grad()
def get_msgs(img_dir: str, msg_decoder: nn.Module, batch_size: int = 16, attacks: dict = {}):
    # resize crop
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_loader = utils.get_dataloader(img_dir, transform, batch_size=batch_size, collate_fn=None)

    log_stats = {ii:{} for ii in range(len(data_loader.dataset))}
    for ii, imgs in enumerate(tqdm.tqdm(data_loader)):

        imgs = imgs.to(device)

        for name, attack in attacks.items():
            imgs_aug = attack(imgs)
            decoded = msg_decoder(imgs_aug)>0 # b c h w -> b k
            for jj in range(decoded.shape[0]):
                img_num = ii*batch_size+jj
                log_stat = log_stats[img_num]
                log_stat[f'decoded_{name}'] = "".join([('1' if el else '0') for el in decoded[jj].detach()])

    log_stats = [{'img': img_num, **log_stats[img_num]} for img_num in range(len(data_loader.dataset))]
    return log_stats

def main(params):

    # Set seeds for reproductibility 
    np.random.seed(params.seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    save_img_dir = os.path.join(params.output_dir, 'imgs')
    params.save_img_dir = save_img_dir
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir, exist_ok=True)

    if params.eval_imgs:

        print(f">>> Saving {params.save_n_imgs} diff images...")
        if params.save_n_imgs > 0:
            save_imgs(params.img_dir, params.img_dir_nw, save_img_dir, num_imgs=params.save_n_imgs)

        print(f'>>> Computing img-2-img stats...')
        img_metrics = get_img_metric(params.img_dir, params.img_dir_nw, num_imgs=params.num_imgs)
        img_df = pd.DataFrame(img_metrics)
        img_df.to_csv(os.path.join(params.output_dir, 'img_metrics.csv'), index=False)
        ssims = img_df['ssim'].tolist()
        psnrs = img_df['psnr'].tolist()
        linfs = img_df['linf'].tolist()
        ssim_mean, ssim_std, ssim_max, ssim_min = np.mean(ssims), np.std(ssims), np.max(ssims), np.min(ssims) 
        psnr_mean, psnr_std, psnr_max, psnr_min = np.mean(psnrs), np.std(psnrs), np.max(psnrs), np.min(psnrs)
        linf_mean, linf_std, linf_max, linf_min = np.mean(linfs), np.std(linfs), np.max(linfs), np.min(linfs)
        print(f"SSIM: {ssim_mean:.4f}±{ssim_std:.4f} [{ssim_min:.4f}, {ssim_max:.4f}]")
        print(f"PSNR: {psnr_mean:.4f}±{psnr_std:.4f} [{psnr_min:.4f}, {psnr_max:.4f}]")
        print(f"Linf: {linf_mean:.4f}±{linf_std:.4f} [{linf_min:.4f}, {linf_max:.4f}]")

        if params.img_dir_fid is not None:
            print(f'>>> Computing image distribution stats...')
            fid = cached_fid(params.img_dir, params.img_dir_fid)
            print(f"FID watermark : {fid:.4f}")
            fid_nw = cached_fid(params.img_dir_nw, params.img_dir_fid)
            print(f"FID vanilla   : {fid_nw:.4f}")

    if params.eval_bits:

        # Loads hidden decoder
        print(f'>>> Building hidden decoder with weights from {params.msg_decoder_path}...')
        if 'torchscript' in params.msg_decoder_path:
            msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        else:
            msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
            ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
            print(msg_decoder.load_state_dict(ckpt, strict=False))
            msg_decoder.eval()

            # whitening
            print(f'>>> Whitening...')
            with torch.no_grad():
                data_dir = "/checkpoint/pfz/watermarking/data/coco_10k_orig/0"
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                loader = utils.get_dataloader(data_dir, transform, batch_size=16, collate_fn=None)
                ys = []
                for i, x in enumerate(loader):
                    x = x.to(device)
                    y = msg_decoder(x)
                    ys.append(y.to('cpu'))
                ys = torch.cat(ys, dim=0)
                nbit = ys.shape[1]
                mean = ys.mean(dim=0, keepdim=True) # NxD -> 1xD
                ys_centered = ys - mean # NxD
                cov = ys_centered.T @ ys_centered
                e, v = torch.linalg.eigh(cov)
                L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
                weight = torch.mm(L, v.T)
                bias = -torch.mm(mean, weight.T).squeeze(0)
                linear = nn.Linear(nbit, nbit, bias=True)
                linear.weight.data = np.sqrt(nbit) * weight
                linear.bias.data = np.sqrt(nbit) * bias
                msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
                torchscript_m = torch.jit.script(msg_decoder)
                torch.jit.save(torchscript_m, params.msg_decoder_path.replace(".pth", "_whit.torchscript.pt"))
        
        msg_decoder.eval()
        nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]

        if params.attack_mode == 'all':
            attacks = {
                'none': lambda x: x,
                'crop_05': lambda x: utils_img.center_crop(x, 0.5),
                'crop_01': lambda x: utils_img.center_crop(x, 0.1),
                'rot_25': lambda x: utils_img.rotate(x, 25),
                'rot_90': lambda x: utils_img.rotate(x, 90),
                'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
                'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
                'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
                'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
                'contrast_1p5': lambda x: utils_img.adjust_contrast(x, 1.5),
                'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
                'saturation_1p5': lambda x: utils_img.adjust_saturation(x, 1.5),
                'saturation_2': lambda x: utils_img.adjust_saturation(x, 2),
                'sharpness_1p5': lambda x: utils_img.adjust_sharpness(x, 1.5),
                'sharpness_2': lambda x: utils_img.adjust_sharpness(x, 2),
                'resize_05': lambda x: utils_img.resize(x, 0.5),
                'resize_01': lambda x: utils_img.resize(x, 0.1),
                'overlay_text': lambda x: utils_img.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
                'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
            }
        elif params.attack_mode == 'few':
            attacks = {
                'none': lambda x: x,
                'crop_01': lambda x: utils_img.center_crop(x, 0.1),
                'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
                'contrast_2': lambda x: utils_img.adjust_contrast(x, 2),
                'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
                'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
            }
        else:
            attacks = {'none': lambda x: x}

        if params.decode_only:
            log_stats = get_msgs(params.img_dir, msg_decoder, batch_size=params.batch_size, attacks=attacks)
        else:    
            # Creating key
            key = torch.tensor([k=='1' for k in params.key_str]).to(device)
            log_stats = get_bit_accs(params.img_dir, msg_decoder, key, batch_size=params.batch_size, attacks=attacks)

        print(f'>>> Saving log stats to {params.output_dir}...')
        df = pd.DataFrame(log_stats)
        df.to_csv(os.path.join(params.output_dir, 'log_stats.csv'), index=False)
        print(df)


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--img_dir", type=str, default="", help="")
    aa("--num_imgs", type=int, default=None)

    group = parser.add_argument_group('Eval imgs')
    aa("--eval_imgs", type=utils.bool_inst, default=True, help="")
    aa("--img_dir_nw", type=str, default="/checkpoint/pfz/2023_logs/0104_aisign_sd_txt2img/_ldm_decoder_ckpt=0_config=0_ckpt=0/samples", help="")
    aa("--img_dir_fid", type=str, default=None, help="")
    aa("--save_n_imgs", type=int, default=10)

    group = parser.add_argument_group('Eval bits')
    aa("--eval_bits", type=utils.bool_inst, default=True, help="")
    aa("--decode_only", type=utils.bool_inst, default=False, help="")
    aa("--key_str", type=str, default="111010110101000001010111010011010100010000100111")
    aa("--msg_decoder_path", type=str, default= "models/dec_48b_whit.torchscript.pt")
    aa("--attack_mode", type=str, default= "all")
    aa("--num_bits", type=int, default=48)
    aa("--redundancy", type=int, default=1)
    aa("--decoder_depth", type=int, default=8)
    aa("--decoder_channels", type=int, default=64)
    aa("--img_size", type=int, default=512)
    aa("--batch_size", type=int, default=32)

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
