# =============================================================================
# Model configurations for Stepwise Generative Urban Design
# Paper: "Human-guided urban form generation using multimodal diffusion models"
#        Building and Environment, 2025
#
# Six models covering two cities × three pipeline steps:
#   step1_nyc   NYC  Step 1: site constraints → land use + road network
#   step1_chi   Chi  Step 1
#   step2_nyc   NYC  Step 2: land use + roads → building footprint layout
#   step2_chi   Chi  Step 2
#   step3_nyc   NYC  Step 3: building layout → satellite image
#   step3_chi   Chi  Step 3
#
# Pipeline stage numbers match the Vector_Image_Partition directory naming:
#   Stage_1 → Stage_2        : Step 1 hint/target
#   Stage_2 → Stage_3        : Step 2 hint/target
#   Stage_3 → Stage_4_grid   : Step 3 hint/target
#
# Each model has a 'modes' dict listing only the inference modes it supports:
#   all         Full test set, single seed, default guidance scale
#   div         Diversity: specific tiles × multiple seeds
#   scale3      Full test set, guidance scale 3.0 (ablation; step3 only)
#   trans       Cross-city transfer (step2_nyc and step3_nyc only)
#   consecutive Human-in-the-loop case study (step2_nyc and step3_nyc only)
# =============================================================================

MODELS = {

    # -------------------------------------------------------------------------
    # Step 1: site constraints (background) → land use + road network
    # -------------------------------------------------------------------------
    'step1_nyc': {
        'city':           'NewYork',
        # infer_stage: integer suffix for Descriptions_s{N}/ directory
        'infer_stage':    2,
        'final_epoch':    '39',
        # csv_prefix: stem shared by all 9 CSVs
        # Full path pattern: Descriptions_s2/NewYork_bg_grid_r_r{0,1,2}_d{0,1,2}.csv
        'csv_prefix':     'NewYork_bg_grid_r',
        # desc_col: text-prompt column in the CSV
        'desc_col':       'descriptions_num',
        # hint/target: subdirectory templates under Vector_Image_Partition/
        'hint_pattern':   '{city}_bg_Stage_1_{r}_{d}',
        'target_pattern': '{city}_bg_Stage_2_{r}_{d}',
        'target_ext':     '.tif',
        'ckpt_dir':       './ckpts_s/checkpoints_step1_nyc',
        'modes': {
            'all': {'scale': 9.0},
            'div': {'scale': 9.0,
                    'tiles': [(54, 60, 'r0', 'd0')],
                    'seeds': list(range(101))},
        },
    },

    'step1_chi': {
        'city':           'Chicago',
        'infer_stage':    2,
        'final_epoch':    '69',
        'csv_prefix':     'Chicago_of_grid_r',
        'desc_col':       'descriptions_num',
        'hint_pattern':   '{city}_of_Stage_1_{r}_{d}',
        'target_pattern': '{city}_of_Stage_2_{r}_{d}',
        'target_ext':     '.tif',
        'ckpt_dir':       './ckpts_s/checkpoints_step1_chi',
        'modes': {
            'all': {'scale': 9.0},
            'div': {'scale': 9.0,
                    'tiles': [(13, 70, 'r0', 'd0'), (68, 37, 'r0', 'd0'), (87, 23, 'r0', 'd0')],
                    'seeds': list(range(10))},
        },
    },

    # -------------------------------------------------------------------------
    # Step 2: land use + roads → building footprint layout
    # -------------------------------------------------------------------------
    'step2_nyc': {
        'city':           'NewYork',
        'infer_stage':    3,
        'final_epoch':    '49',
        'csv_prefix':     'NewYork_gc_grid_r',
        'desc_col':       'descriptions_num_gc',
        'hint_pattern':   '{city}_bgr_Stage_2_{r}_{d}',
        'target_pattern': '{city}_bgrNluc_Stage_3_{r}_{d}',
        'target_ext':     '.tif',
        'ckpt_dir':       './ckpts_s/checkpoints_step2_nyc',
        'modes': {
            'all':  {'scale': 9.0},
            'div':  {'scale': 9.0,
                     'tiles': [(61, 52, 'r0', 'd0'), (88, 70, 'r0', 'd0')],
                     'seeds': list(range(10))},
            # Cross-city transfer: run Chicago's Step 2 model on NYC test data
            'trans': {'model': 'step2_chi', 'scale': 9.0},
            # Human-in-the-loop case study:
            #   Place your manually edited hint image at the path below
            'consecutive': {'hint':  'consecutive_test/hint_step2.jpg',
                            'tile':  (88, 70, 'r0', 'd0'),
                            'scale': 9.0,
                            'seeds': list(range(80))},
        },
    },

    'step2_chi': {
        'city':           'Chicago',
        'infer_stage':    3,
        'final_epoch':    '44',
        'csv_prefix':     'Chicago_of_grid_r',
        'desc_col':       'descriptions_num_gc',
        'hint_pattern':   '{city}_ofr_Stage_2_{r}_{d}',
        'target_pattern': '{city}_ofrNluc_Stage_3_{r}_{d}',
        'target_ext':     '.tif',
        'ckpt_dir':       './ckpts_s/checkpoints_step2_chi',
        'modes': {
            'all': {'scale': 9.0},
            'div': {'scale': 9.0,
                    'tiles': [(13, 70, 'r0', 'd0'), (68, 37, 'r0', 'd0'), (87, 23, 'r0', 'd0')],
                    'seeds': list(range(10))},
        },
    },

    # -------------------------------------------------------------------------
    # Step 3: building footprint layout → satellite image
    # -------------------------------------------------------------------------
    'step3_nyc': {
        'city':           'NewYork',
        'infer_stage':    4,
        'final_epoch':    '29',
        'csv_prefix':     'NewYork_gce_grid_r',
        'desc_col':       'descriptions_sa',
        'hint_pattern':   '{city}_gc_Stage_3_{r}_{d}',
        'target_pattern': '{city}_gce_Stage_4_grid_{r}_{d}',
        'target_ext':     '.jpg',
        'ckpt_dir':       './ckpts_s/checkpoints_step3_nyc',
        'modes': {
            'all':    {'scale': 9.0},
            'scale3': {'scale': 3.0},
            'div':    {'scale': 9.0,
                       'tiles': [(61, 52, 'r0', 'd0'), (88, 70, 'r0', 'd0')],
                       'seeds': list(range(10))},
            # Cross-city transfer: run Chicago's Step 3 model on NYC test data
            'trans': {'model': 'step3_chi', 'scale': 3.0},
            # Human-in-the-loop case study:
            #   Place your manually edited hint image at the path below
            'consecutive': {'hint':  'consecutive_test/hint_step3.jpg',
                            'tile':  (88, 70, 'r0', 'd0'),
                            'scale': 3.0,
                            'seeds': list(range(80))},
        },
    },

    'step3_chi': {
        'city':           'Chicago',
        'infer_stage':    4,
        'final_epoch':    '29',
        'csv_prefix':     'Chicago_of_grid_r',
        'desc_col':       'descriptions_sa',
        'hint_pattern':   '{city}_of_Stage_3_{r}_{d}',
        'target_pattern': '{city}_of_Stage_4_grid_{r}_{d}',
        'target_ext':     '.jpg',
        'ckpt_dir':       './ckpts_s/checkpoints_step3_chi',
        'modes': {
            'all':    {'scale': 9.0},
            'scale3': {'scale': 3.0},
            'div':    {'scale': 9.0,
                       'tiles': [(13, 70, 'r0', 'd0'), (68, 37, 'r0', 'd0'), (87, 23, 'r0', 'd0')],
                       'seeds': list(range(10))},
        },
    },
}


def build_data_dirs(dir_my, cfg):
    """
    Return the list of 9 CSV paths for a given model config and data root.

    Covers all (r, d) combinations where r ∈ {r0, r1, r2} and d ∈ {d0, d1, d2},
    matching the grid-offset augmentation scheme used during training.

    Args:
        dir_my (str): root of Urban_Data directory, trailing slash required,
                      e.g. '/path/to/Urban_Data/'
        cfg    (dict): a model config from MODELS

    Returns:
        list[str]: 9 CSV file paths
    """
    stage  = cfg['infer_stage']
    prefix = cfg['csv_prefix']
    return [
        f'{dir_my}Descriptions_s{stage}/{prefix}_r{r}_d{d}.csv'
        for d in range(3)
        for r in range(3)
    ]
