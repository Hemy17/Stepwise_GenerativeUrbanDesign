# =============================================================================
# Inference script for Stepwise Generative Urban Design
# Paper: "Human-guided urban form generation using multimodal diffusion models"
#        Building and Environment, 2026
#
# ---- Modes ------------------------------------------------------------------
#   all         Full test set, single seed, guidance scale from config.
#               Output: ./output_image/{model}/all_epoch_{epoch}/{x}_{y}_{r}_{d}.png
#
#   div         Diversity evaluation: specific tiles × multiple seeds.
#               Tiles and seeds are defined per model in configs.py.
#               Output: .../div_epoch_{epoch}/{x}_{y}_{r}_{d}_s{seed}.png
#
#   trans       Cross-city transfer: this model's test data with the other
#               city's checkpoint. Only valid for step2_nyc and step3_nyc.
#               Output: .../trans_epoch_{epoch}/{x}_{y}_{r}_{d}.png
#
#   scale3      Full test set with guidance scale=3.0 (ablation study).
#               Output: .../scale3_epoch_{epoch}/{x}_{y}_{r}_{d}.png
#
#   consecutive Human-in-the-loop case study: a single tile with a manually
#               edited hint image, run over multiple seeds.
#               Only valid for step2_nyc and step3_nyc.
#               Place your hint image at:
#                 {Vector_Image_Partition}/{consecutive_hint path from configs.py}
#               Output: .../consecutive_epoch_{epoch}/{x}_{y}_{r}_{d}_s{seed}.png
#
# ---- Usage ------------------------------------------------------------------
#   python inference.py --model step1_nyc --mode all   --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step1_chi --mode all   --gpu 1 --data_dir /path/to/Urban_Data/
#   python inference.py --model step2_nyc --mode all   --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step2_nyc --mode trans --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step2_nyc --mode consecutive --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step3_nyc --mode all   --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step3_nyc --mode div   --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step3_nyc --mode scale3 --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step3_nyc --mode trans --gpu 0 --data_dir /path/to/Urban_Data/
#   python inference.py --model step3_nyc --mode consecutive --gpu 0 --data_dir /path/to/Urban_Data/
# =============================================================================

from share import *
import config   # sets config.save_memory

import argparse
import cv2
import einops
import numpy as np
import os
import pandas as pd
import random
import torch
from PIL import Image

from pytorch_lightning import seed_everything

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from configs import MODELS, build_data_dirs


# ---- Fixed inference hyper-parameters (shared across all modes) ---------------
NUM_SAMPLES = 1
STRENGTH    = 1.0
GUESS_MODE  = False
DDIM_STEPS  = 20
ETA         = 0.0
A_PROMPT = 'best quality, extremely detailed'
N_PROMPT = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')

SINGLE_SEED = 5354   # default seed for all / trans / scale3 modes

# Module-level model references (assigned after load_model())
model        = None
ddim_sampler = None


# ---- Model loading -----------------------------------------------------------

def load_model(ckpt_path, device):
    """Load ControlNet model and DDIM sampler from a checkpoint."""
    global model, ddim_sampler

    torch.cuda.empty_cache()
    if model is not None:
        del model
    if ddim_sampler is not None:
        del ddim_sampler
    torch.cuda.empty_cache()

    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
    model.to(device)
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def find_checkpoint(ckpt_dir, epoch):
    """Return the checkpoint path for the given epoch in ckpt_dir."""
    matches = [f for f in os.listdir(ckpt_dir)
               if f.startswith(f'epoch={epoch}-step=')]
    if not matches:
        raise FileNotFoundError(
            f'No checkpoint found for epoch={epoch} in {ckpt_dir}')
    return os.path.join(ckpt_dir, matches[0])


# ---- Core inference ----------------------------------------------------------

def process(image_dir, city, cfg, xtile, ytile, r, d,
            prompt, scale, seed, hint_override=None):
    """
    Run one forward pass of ControlNet diffusion for a single tile.

    Args:
        image_dir     (str):        path to Vector_Image_Partition/
        city          (str):        city name used in directory path construction
        cfg           (dict):       model config from configs.MODELS
        xtile, ytile  (int/float):  tile row and column indices
        r, d          (str):        grid-offset codes, e.g. 'r0', 'd0'
        prompt        (str):        text description
        scale         (float):      classifier-free guidance scale
        seed          (int):        random seed (-1 for random)
        hint_override (str|None):   if set, use this path as the hint image
                                    instead of the default (used in 'consecutive' mode)

    Returns:
        tuple: (hint_rgb, ground_truth_bgr, generated_rgb)  — all numpy uint8
    """
    with torch.no_grad():
        tile_stem     = f'{xtile}_{ytile}_{r}_{d}'
        hint_subdir   = cfg['hint_pattern'].format(city=city, r=r, d=d)
        target_subdir = cfg['target_pattern'].format(city=city, r=r, d=d)

        hint_path   = hint_override or f'{image_dir}{hint_subdir}/{tile_stem}.tif'
        target_path = f'{image_dir}{target_subdir}/{tile_stem}{cfg["target_ext"]}'

        # Ground-truth target is read only to obtain spatial dimensions (H, W)
        input_image = cv2.imread(target_path)
        if input_image is None:
            raise FileNotFoundError(f'Target image not found: {target_path}')
        H, W, _ = input_image.shape

        detected_map = cv2.imread(hint_path, cv2.IMREAD_UNCHANGED)
        if detected_map is None:
            raise FileNotFoundError(f'Hint image not found: {hint_path}')

        # Handle 4-channel RGBA: replace transparent pixels with white
        if detected_map.shape[2] == 4:
            trans_mask = detected_map[:, :, 3] == 0
            detected_map[trans_mask] = [255, 255, 255, 255]
            detected_map = cv2.cvtColor(detected_map, cv2.COLOR_BGRA2BGR)

        control = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        control = torch.from_numpy(control.copy()).float().cuda() / 255.0
        control = torch.stack([control] * NUM_SAMPLES, dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            'c_concat':    [control],
            'c_crossattn': [model.get_learned_conditioning(
                [prompt + ', ' + A_PROMPT] * NUM_SAMPLES)],
        }
        un_cond = {
            'c_concat':    None if GUESS_MODE else [control],
            'c_crossattn': [model.get_learned_conditioning(
                [N_PROMPT] * NUM_SAMPLES)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [STRENGTH * (0.825 ** float(12 - i)) for i in range(13)]
            if GUESS_MODE else [STRENGTH] * 13
        )
        samples, _ = ddim_sampler.sample(
            DDIM_STEPS, NUM_SAMPLES, shape, cond,
            verbose=False, eta=ETA,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
        ).cpu().numpy().clip(0, 255).astype(np.uint8)

        hint_rgb = cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)
        return hint_rgb, input_image, x_samples[0]


# ---- Helpers -----------------------------------------------------------------

def build_test_df(data_dir_root, cfg):
    """Load the test-set DataFrame (Train ≠ 1 AND img == 1)."""
    df = pd.DataFrame()
    for f in build_data_dirs(data_dir_root, cfg):
        dfi = pd.read_csv(f)
        df = pd.concat([df, dfi[(dfi['Train'] != 1) & (dfi['img'] == 1)]])
    print(f'Test samples: {df.shape[0]}')
    return df.reset_index(drop=True)


def filter_tiles(df, tile_list):
    """Keep only rows matching the (row, col, r, d) tuples in tile_list."""
    tile_set = set(tile_list)
    mask = df.apply(
        lambda x: (x['row'], x['col'], x['r'], x['d']) in tile_set, axis=1)
    return df[mask].reset_index(drop=True)


def save_image(image_array, output_dir, filename):
    """Save a numpy RGB array as a JPEG-encoded .png file (quality 85)."""
    Image.fromarray(image_array).save(
        os.path.join(output_dir, filename),
        format='JPEG', quality=85, optimize=True,
    )


# ---- Main --------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run inference for one model and mode.')
    parser.add_argument('--model',    required=True, choices=list(MODELS.keys()),
                        help='e.g. step3_nyc')
    parser.add_argument('--mode',     required=True,
                        choices=['all', 'div', 'trans', 'scale3', 'consecutive'])
    parser.add_argument('--gpu',      required=True, type=int,
                        help='CUDA device index')
    parser.add_argument('--data_dir', required=True,
                        help='Root of Urban_Data/, e.g. /path/to/Urban_Data/')
    return parser.parse_args()


def main():
    args  = parse_args()
    cfg   = MODELS[args.model]
    mode  = args.mode
    city  = cfg['city']
    epoch = cfg['final_epoch']

    # ---- Device setup -------------------------------------------------------
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    print(f'Device : {torch.cuda.get_device_name(device)}')

    data_dir_root = args.data_dir.rstrip('/') + '/'
    image_dir     = data_dir_root + 'Vector_Image_Partition/'

    # ---- Validate mode -------------------------------------------------------
    if mode not in cfg['modes']:
        raise ValueError(
            f"Mode '{mode}' is not supported for model '{args.model}'. "
            f"Supported modes: {list(cfg['modes'].keys())}")

    mode_cfg = cfg['modes'][mode]

    # ---- Load checkpoint -----------------------------------------------------
    if mode == 'trans':
        other_cfg = MODELS[mode_cfg['model']]
        ckpt_path = find_checkpoint(other_cfg['ckpt_dir'], other_cfg['final_epoch'])
        print(f'Transfer: using {mode_cfg["model"]} checkpoint → {ckpt_path}')
    else:
        ckpt_path = find_checkpoint(cfg['ckpt_dir'], epoch)
    print(f'Checkpoint: {ckpt_path}')

    load_model(ckpt_path, device)

    # ---- Build test DataFrame ------------------------------------------------
    df = build_test_df(data_dir_root, cfg)

    # ---- Mode-specific setup -------------------------------------------------
    output_subdir = f'./output_image/{args.model}/{mode}_epoch_{epoch}'

    scale   = mode_cfg['scale']
    # Modes with per-seed outputs store a 'seeds' list; others use a single fixed seed.
    seeds   = mode_cfg.get('seeds', [SINGLE_SEED])
    hint_fn = None

    if mode == 'div':
        df = filter_tiles(df, mode_cfg['tiles'])
        print(f'Diversity: {len(df)} tile(s) × {len(seeds)} seeds')

    elif mode == 'consecutive':
        hint_fn = image_dir + mode_cfg['hint']
        df      = filter_tiles(df, [mode_cfg['tile']])
        print(f'Consecutive: hint={hint_fn}, {len(seeds)} seeds, scale={scale}')
        if not os.path.exists(hint_fn):
            raise FileNotFoundError(
                f'Hint image not found: {hint_fn}\n'
                f'Place your manually edited hint at that path and retry.')

    os.makedirs(output_subdir, exist_ok=True)

    # ---- Inference loop ------------------------------------------------------
    for i in range(len(df)):
        xtile  = df.iloc[i]['row']
        ytile  = df.iloc[i]['col']
        r      = df.iloc[i]['r']
        d      = df.iloc[i]['d']
        prompt = df.iloc[i][cfg['desc_col']]

        for seed in seeds:
            # Include seed in filename only for multi-seed modes (div, consecutive)
            if 'seeds' in mode_cfg:
                filename = f'{xtile}_{ytile}_{r}_{d}_s{seed}.png'
            else:
                filename = f'{xtile}_{ytile}_{r}_{d}.png'

            save_path = os.path.join(output_subdir, filename)
            if os.path.exists(save_path):
                print(f'  skip: {save_path}')
                continue

            try:
                _, _, generated = process(
                    image_dir, city, cfg, xtile, ytile, r, d,
                    prompt, scale, seed, hint_override=hint_fn,
                )
                save_image(generated, output_subdir, filename)
                print(f'  saved: {save_path}')
            except FileNotFoundError as e:
                print(f'  ERROR: {e}')

    print(f'\nDone. Output in: {output_subdir}')


if __name__ == '__main__':
    main()
