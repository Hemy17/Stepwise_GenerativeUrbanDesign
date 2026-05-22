# =============================================================================
# Dataset for Stepwise Generative Urban Design
#
# Unified dataset class for all 6 models (2 cities × 3 pipeline steps).
# Model-specific differences (image paths, text column) are passed as
# arguments drawn from the model config in configs.py.
#
# This file is not meant to be run directly; it is imported by train.py.
# =============================================================================

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)


class MyDataset(Dataset):
    """
    Dataset returning (hint, target, prompt) triples for ControlNet fine-tuning.

    Each sample consists of:
        hint   – source condition image (control signal), normalised to [0, 1]
        target – ground-truth output image, normalised to [−1, 1]
        prompt – text description read from the CSV

    Args:
        image_dir      (str):       path to Vector_Image_Partition/, trailing slash required
        data_dir       (list[str]): list of 9 CSV file paths (use configs.build_data_dirs())
        hint_dir       (str):       same as image_dir; kept for API compatibility with
                                    the original satellite_tiles_tXX.py interface
        hint_pattern   (str):       subdirectory template under Vector_Image_Partition/,
                                    e.g. '{city}_bg_Stage_1_{r}_{d}'
        target_pattern (str):       subdirectory template for the target image,
                                    e.g. '{city}_bg_Stage_2_{r}_{d}'
        target_ext     (str):       file extension for target images:
                                    '.tif'  for Steps 1 and 2
                                    '.jpg'  for Step 3
        desc_col       (str):       CSV column name for text descriptions:
                                    'descriptions_num'    →  Step 1
                                    'descriptions_num_gc' →  Step 2
                                    'descriptions_sa'     →  Step 3

    Note:
        Hint images always use the .tif extension.
        4-channel RGBA hints are handled by replacing transparent pixels with white
        and converting to 3-channel BGR before further processing.
    """

    def __init__(self, image_dir, data_dir, hint_dir,
                 hint_pattern, target_pattern, target_ext, desc_col):
        self.image_dir      = image_dir
        self.hint_pattern   = hint_pattern
        self.target_pattern = target_pattern
        self.target_ext     = target_ext

        df = pd.DataFrame()
        for f in data_dir:
            dfi = pd.read_csv(f)
            # Training rows only (Train == 1) with an image present (img == 1)
            df = pd.concat([df, dfi[(dfi['Train'] == 1) & (dfi['img'] == 1)]])

        self.image_city       = df['city'].to_list()
        self.image_list_xtile = df['row'].to_list()
        self.image_list_ytile = df['col'].to_list()
        self.image_list_right = df['r'].to_list()
        self.image_list_down  = df['d'].to_list()
        self.description      = df[desc_col].to_list()

    def __len__(self):
        return len(self.image_list_xtile)

    def __getitem__(self, item):
        xtile = self.image_list_xtile[item]
        ytile = self.image_list_ytile[item]
        r     = self.image_list_right[item]
        d     = self.image_list_down[item]
        city  = self.image_city[item]

        tile_stem     = f'{xtile}_{ytile}_{r}_{d}'
        hint_subdir   = self.hint_pattern.format(city=city, r=r, d=d)
        target_subdir = self.target_pattern.format(city=city, r=r, d=d)

        # Hint images always use .tif; target extension is model-specific
        hint_path   = f'{self.image_dir}{hint_subdir}/{tile_stem}.tif'
        target_path = f'{self.image_dir}{target_subdir}/{tile_stem}{self.target_ext}'

        target = cv2.imread(target_path)
        source = cv2.imread(hint_path, cv2.IMREAD_UNCHANGED)
        if source is None:
            print(f'WARNING: hint image not found: {hint_path}')

        # Handle 4-channel RGBA hints: replace transparent pixels with white
        if source.shape[2] == 4:
            trans_mask = source[:, :, 3] == 0
            source[trans_mask] = [255, 255, 255, 255]
            source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)

        # OpenCV reads images in BGR order; convert both to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalise: hint → [0, 1],  target → [−1, 1]
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(
            jpg=torch.tensor(target).to(device),
            txt=self.description[item],
            hint=torch.tensor(source).to(device),
        )
