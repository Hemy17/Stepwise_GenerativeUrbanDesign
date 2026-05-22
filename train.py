# =============================================================================
# Training script for Stepwise Generative Urban Design
# Paper: "Human-guided urban form generation using multimodal diffusion models"
#        Building and Environment, 2026
#
# Usage:
#   python train.py --model step1_nyc --gpu 0 --data_dir /path/to/Urban_Data/
#   python train.py --model step1_chi --gpu 0 --data_dir /path/to/Urban_Data/
#   python train.py --model step2_nyc --gpu 0 --data_dir /path/to/Urban_Data/
#   python train.py --model step2_chi --gpu 0 --data_dir /path/to/Urban_Data/
#   python train.py --model step3_nyc --gpu 0 --data_dir /path/to/Urban_Data/
#   python train.py --model step3_chi --gpu 0 --data_dir /path/to/Urban_Data/
#
# To resume from a checkpoint:
#   python train.py --model step1_nyc --gpu 0 --data_dir /path/to/Urban_Data/ \
#       --resume ./ckpts_s/checkpoints_step1_nyc/epoch=N-step=xxx.ckpt
#
# Without --resume, training starts from ./models/control_sd15_ini.ckpt
# (the SD1.5 ControlNet initialisation checkpoint).
#
# Optional:
#   --batch_size N   mini-batch size (default: 2)
#   --lr LR          learning rate (default: 1e-5)
#
# Checkpoints are saved every 5 epochs to ./ckpts_s/checkpoints_{model}/.
# Training runs until manually stopped (no fixed max_epochs).
# =============================================================================

from share import *

import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import MyDataset
from configs import MODELS, build_data_dirs
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a ControlNet model for one pipeline step and city.')
    parser.add_argument('--model',      required=True,
                        choices=list(MODELS.keys()),
                        help='Model to train, e.g. step1_nyc')
    parser.add_argument('--gpu',        required=True, type=int,
                        help='CUDA device index, e.g. 0')
    parser.add_argument('--data_dir',   required=True,
                        help='Root of Urban_Data directory (with trailing slash), '
                             'e.g. /home/user/Urban_Data/')
    parser.add_argument('--resume',     default=None,
                        help='Path to checkpoint to resume from. '
                             'Defaults to ./models/control_sd15_ini.ckpt')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr',         default=1e-5, type=float,
                        help='Learning rate (default: 1e-5)')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = MODELS[args.model]
    device_str = f'cuda:{args.gpu}'
    device = torch.device(device_str)
    torch.cuda.set_device(device)
    print(f'Using device: {torch.cuda.get_device_name(device)}')

    # Ensure data_dir ends with a slash
    data_dir_root = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
    image_dir = data_dir_root + 'Vector_Image_Partition/'
    data_dir  = build_data_dirs(data_dir_root, cfg)

    ck_output_file = cfg['ckpt_dir']
    resume_path    = args.resume if args.resume else './models/control_sd15_ini.ckpt'

    print(f'Model config   : {args.model} ({cfg["old_id"]})')
    print(f'City           : {cfg["city"]}')
    print(f'Pipeline stage : {cfg["infer_stage"]}')
    print(f'Checkpoint dir : {ck_output_file}')
    print(f'Resuming from  : {resume_path}')

    # ---- Model ----------------------------------------------------------
    model = create_model('./models/cldm_v15.yaml').to(device)
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate  = args.lr
    model.sd_locked      = False   # fine-tune entire SD backbone
    model.only_mid_control = False
    model.to(device)

    # ---- Checkpoint callback --------------------------------------------
    os.makedirs(ck_output_file, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ck_output_file,
        filename='{epoch}-{step}',
        save_top_k=-1,         # keep all checkpoints
        every_n_epochs=5,
        save_last=False,
    )

    # ---- Dataset & DataLoader -------------------------------------------
    dataset = MyDataset(
        image_dir      = image_dir,
        data_dir       = data_dir,
        hint_dir       = image_dir,   # kept for API compatibility
        hint_pattern   = cfg['hint_pattern'],
        target_pattern = cfg['target_pattern'],
        target_ext     = cfg['target_ext'],
        desc_col       = cfg['desc_col'],
    )
    dataloader = DataLoader(dataset, num_workers=0,
                            batch_size=args.batch_size, shuffle=True)

    # ---- Trainer --------------------------------------------------------
    logger  = ImageLogger(batch_frequency=300)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[args.gpu],
        precision=32,
        callbacks=[logger, checkpoint_callback],
    )

    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
