"""
Dataset Specifics
Extended from ADNet code by Hansen et al.
"""

import torch
import random


def get_label_names(dataset):
    label_names = {}
    if dataset == 'CMR':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'

    elif dataset == 'CHAOST2':
        label_names[0] = 'BG'
        label_names[1] = 'LIVER'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'SPLEEN'
    elif dataset == 'SABS':
        label_names[0] = 'BG'
        label_names[1] = 'SPLEEN'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'GALLBLADDER'
        label_names[5] = 'ESOPHAGUS'
        label_names[6] = 'LIVER'
        label_names[7] = 'STOMACH'
        label_names[8] = 'AORTA'
        label_names[9] = 'IVC'  # Inferior vena cava
        label_names[10] = 'PS_VEIN'  # portal vein and splenic vein
        label_names[11] = 'PANCREAS'
        label_names[12] = 'AG_R'  # right adrenal gland
        label_names[13] = 'AG_L'  # left adrenal gland
    elif dataset == 'Prostate_MRI':
        label_names[0] = 'BG'
        label_names[1] = 'BLADDER'
        label_names[2] = 'BONE'
        label_names[3] = 'OBTURATOR INTERNUS'
        label_names[4] = 'TRANSITION ZONE'
        label_names[5] = 'CENTRAL GLAND'
        label_names[6] = 'RECTUM'
        label_names[7] = 'SEMINAL VESICLE'
        label_names[8] = 'NEUROVASCULAR BUNDLE'


    return label_names


def get_folds(dataset):
    FOLD = {}
    if dataset == 'CMR':
        FOLD[0] = set(range(0, 2))
        return FOLD
    else:
        raise ValueError(f'Dataset: {dataset} not found')


def sample_xy(spr, k=0, b=215):
    _, h, v = torch.where(spr)

    if len(h) == 0 or len(v) == 0:
        horizontal = 0
        vertical = 0
    else:

        h_min = min(h)
        h_max = max(h)
        if b > (h_max - h_min):
            kk = min(k, int((h_max - h_min) / 2))
            horizontal = random.randint(max(h_max - b - kk, 0), min(h_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            horizontal = random.randint(max(h_min - kk, 0), min(h_max - b + kk, 256 - b - 1))

        v_min = min(v)
        v_max = max(v)
        if b > (v_max - v_min):
            kk = min(k, int((v_max - v_min) / 2))
            vertical = random.randint(max(v_max - b - kk, 0), min(v_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            vertical = random.randint(max(v_min - kk, 0), min(v_max - b + kk, 256 - b - 1))

    return horizontal, vertical
