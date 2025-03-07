"""
Dataset for Training and Test
Extended from ADNet code by Hansen et al.
"""
import torch
from torch.utils.data import Dataset
import glob
import os
import SimpleITK as sitk
import numpy as np
from .dataset_specifics_demo import *

class TestDataset(Dataset):

    def __init__(self, args):

        # reading the paths
        self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'image*'))

        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # remove test fold!
        self.FOLD = get_folds(args['dataset'])
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args['eval_fold']]]

        # split into support/query
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args['supp_idx']]]
        self.image_dirs.pop(idx[args['supp_idx']])  # remove support
        self.label = None

        # self.sprvxl_dirs = glob.glob(os.path.join(args['data_dir'], 'supervoxels_' + str(args['n_sv']), 'super*'))
        # self.sprvxl_dict = {x.split('_')[-1].split('.nii.gz')[0]: x for x in self.sprvxl_dirs}


    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):

        img_path = self.image_dirs[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        # img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))

        # sprvxl = sitk.GetArrayFromImage(
        #     sitk.ReadImage(self.sprvxl_dict[img_path.split('image_')[-1][:-7]]))

        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)

        sample = {'id': img_path}

        # Evaluation protocol.
        idx = lbl.sum(axis=(1, 2)) > 0
        sample['image'] = torch.from_numpy(img[idx])
        sample['label'] = torch.from_numpy(lbl[idx])
        # sample['sprvxl'] = torch.from_numpy(sprvxl[idx])

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to Ouyang et al.
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        img_path = self.support_dir
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        # img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=1)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == label)

        sample = {}
        if all_slices:
            sample['image'] = torch.from_numpy(img)
            sample['label'] = torch.from_numpy(lbl)
        else:
            # select N labeled slices
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            idx = lbl.sum(axis=(1, 2)) > 0
            idx_ = self.get_support_index(N, idx.sum())

            sample['image'] = torch.from_numpy(img[idx][idx_])
            sample['label'] = torch.from_numpy(lbl[idx][idx_])

        return sample
