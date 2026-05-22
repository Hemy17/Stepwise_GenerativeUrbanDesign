
# Human-guided urban form generation using multimodal diffusion models

Generative AI for Urban Design: A Stepwise Approach Integrating Human Expertise with Multimodal Diffusion Models

[Full paper](https://doi.org/10.1016/j.buildenv.2025.113892)
[Arxiv](https://arxiv.org/abs/2505.24260)
[Model weights](https://huggingface.co/Hemy24/stepwise-generative-urban-design)

**Abstract**: Urban morphological design plays a critical role in shaping environmental quality and urban livability. Recent advances in generative AI offer new opportunities for urban form generation, yet existing methods often lack mechanisms for human intervention and struggle to produce high-fidelity and functionally viable designs. This study proposes a human-in-the-loop generative framework for urban design based on multimodal diffusion models. We enhance Stable Diffusion with ControlNet to support high-fidelity urban form generation under environmental constraints and expert guidance. The generation process is structured as a hierarchical, stepwise pipeline, where land use configurations, building layouts, and satellite imagery are sequentially produced based on human input. Using spatial data from Chicago and New York City, we demonstrate that our framework outperforms GAN-based baselines in visual realism, compliance with design intent, and spatial diversity. Compared to end-to-end approaches, the stepwise framework produces more functionally viable and context-sensitive urban forms. A case study involving professional urban designers further illustrates the framework's effectiveness in supporting collaborative human–AI design. These findings highlight the advantages of diffusion-based models and human-guided generation in supporting scalable and environmentally responsive urban design.

This repository is forked from the original [ControlNet](https://github.com/lllyasviel/ControlNet) and [GenerativeUrbanDesign](https://github.com/sunnyqywang/GenerativeUrbanDesign).

---

## Repository Structure

```
.
├── configs.py          # Model configurations for all 6 trained models
├── dataset.py          # Unified dataset class (replaces satellite_tiles_tXX.py)
├── train.py            # Training script (replaces train_tXX.py)
├── inference.py        # Inference script with 5 evaluation modes
├── cldm/               # ControlNet model implementation
├── ldm/                # Latent Diffusion Model backbone
├── annotator/          # Image annotator utilities
├── models/             # Model config YAML and initial checkpoint (download separately)
└── ckpts_s/            # Fine-tuned checkpoints (download separately, see below)
```

---

## Setup

**Environment**

```bash
conda env create -f environment.yaml
conda activate control
```

**Data**

Download the urban spatial data and place it under a local `Urban_Data/` directory:

```
Urban_Data/
├── Vector_Image_Partition/    # Input/output image tiles
└── Descriptions_s{2,3,4}/    # CSV files with tile metadata and text descriptions
```

**Model checkpoints**

Download the initial SD1.5 ControlNet checkpoint and the six fine-tuned checkpoints, then place them as follows:

```
models/
└── control_sd15_ini.ckpt

ckpts_s/
├── checkpoints_step1_nyc/
├── checkpoints_step1_chi/
├── checkpoints_step2_nyc/
├── checkpoints_step2_chi/
├── checkpoints_step3_nyc/
└── checkpoints_step3_chi/
```

---

## Models

| Name | City | Pipeline Step |
|---|---|---|
| `step1_nyc` | New York | Step 1: site constraints → land use + roads |
| `step1_chi` | Chicago  | Step 1 |
| `step2_nyc` | New York | Step 2: land use + roads → building layout |
| `step2_chi` | Chicago  | Step 2 |
| `step3_nyc` | New York | Step 3: building layout → satellite image |
| `step3_chi` | Chicago  | Step 3 |

---

## Training

```bash
# Train a model from scratch (starting from control_sd15_ini.ckpt)
python train.py --model step1_nyc --gpu 0 --data_dir /path/to/Urban_Data/

# Resume from an intermediate checkpoint
python train.py --model step1_chi --gpu 1 --data_dir /path/to/Urban_Data/ \
    --resume ./ckpts_s/checkpoints_t34/epoch=24-step=24_244549.ckpt

# Optional arguments
#   --batch_size N   mini-batch size (default: 2)
#   --lr LR          learning rate (default: 1e-5)
```

Checkpoints are saved every 5 epochs to `./ckpts_s/checkpoints_{id}/`.

---

## Inference

### Modes

| Mode | Description |
|---|---|
| `all` | Full test set, single seed (5354), guidance scale 9.0 |
| `div` | Diversity evaluation: specific tiles × multiple seeds |
| `trans` | Cross-city transfer: this city's test data with the other city's checkpoint |
| `scale3` | Full test set with guidance scale 3.0 (ablation) |
| `consecutive` | Human-in-the-loop case study: manually edited hint image × 80 seeds |

### Usage

```bash
# Standard evaluation on full test set
python inference.py --model step1_nyc  --mode all  --gpu 0 --data_dir /path/to/Urban_Data/
python inference.py --model step1_chi  --mode all  --gpu 1 --data_dir /path/to/Urban_Data/
python inference.py --model step2_nyc  --mode all  --gpu 0 --data_dir /path/to/Urban_Data/
python inference.py --model step2_chi  --mode all  --gpu 1 --data_dir /path/to/Urban_Data/
python inference.py --model step3_nyc  --mode all  --gpu 0 --data_dir /path/to/Urban_Data/
python inference.py --model step3_chi  --mode all  --gpu 1 --data_dir /path/to/Urban_Data/

# Diversity evaluation
python inference.py --model step3_nyc  --mode div  --gpu 0 --data_dir /path/to/Urban_Data/

# Cross-city transfer (only step2_nyc and step3_nyc)
python inference.py --model step2_nyc  --mode trans --gpu 0 --data_dir /path/to/Urban_Data/
python inference.py --model step3_nyc  --mode trans --gpu 0 --data_dir /path/to/Urban_Data/

# Guidance scale ablation (only step3_nyc and step3_chi in the paper)
python inference.py --model step3_nyc  --mode scale3 --gpu 0 --data_dir /path/to/Urban_Data/
python inference.py --model step3_chi  --mode scale3 --gpu 1 --data_dir /path/to/Urban_Data/

# Human-in-the-loop case study (only step2_nyc and step3_nyc)
# Requires a manually edited hint image placed at:
#   Vector_Image_Partition/consecutive_test/stage1-revise-88_70_r0_d0_s7.998.jpg  (step2_nyc)
#   Vector_Image_Partition/consecutive_test/stage2-revise-88_70_r0_d0_s192.jpg    (step3_nyc)
python inference.py --model step2_nyc  --mode consecutive --gpu 0 --data_dir /path/to/Urban_Data/
python inference.py --model step3_nyc  --mode consecutive --gpu 0 --data_dir /path/to/Urban_Data/
```

### Output

Results are saved to `./output_image/{model}/{mode}_epoch_{epoch}/`.

---

## Citation

```bibtex
@article{he2025human,
  title   = {Human-guided urban form generation using multimodal diffusion models},
  author  = {He, Mingyi and Liang, Yuebing and Wang, Shenhao and Zheng, Yunhan
             and Wang, Qingyi and Zhuang, Dingyi and Tian, Li and Zhao, Jinhua},
  journal = {Building and Environment},
  pages   = {113892},
  year    = {2025},
  doi     = {10.1016/j.buildenv.2025.113892}
}

@article{he2025generative,
  title   = {Generative {AI} for urban design: a stepwise approach integrating
             human expertise with multimodal diffusion models},
  author  = {He, Mingyi and Liang, Yuebing and Wang, Shenhao and Zheng, Yunhan
             and Wang, Qingyi and Zhuang, Dingyi and Tian, Li and Zhao, Jinhua},
  journal = {arXiv preprint arXiv:2505.24260},
  year    = {2025}
}
```
