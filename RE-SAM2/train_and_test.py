#!/usr/bin/env python
import argparse
import copy
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from utils import *
from torchvision import transforms
from model.RE_SAM2 import RE_SAM2
from datetime import datetime
import os
import tqdm
import logging

cudnn.enabled = True
cudnn.benchmark = True
torch.set_num_threads(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def config():
    """Default configurations"""
    num_workers = 16
    seed = 2025
    n_part = 3  # defines the number of chunks for evaluation
    all_supp = [2]  # CHAOST2: 0-4, CMR: 0-7
    batch_size = 1

    optim = {
        'reward_model_lr': 1e-5,
        'policy_model_lr': 1e-6,
    }

    path = {
        'CHAOST2': {'data_dir': '../Dataset/CHAOST2'},
        'SABS': {'data_dir': '../Dataset/SABS'},
        'CMR': {'data_dir': '../Dataset/CMR'},
    }

    return locals().copy()

def main(n_fold):
    _config = config()

    random.seed(_config["seed"])
    torch.manual_seed(_config["seed"])
    torch.cuda.manual_seed_all(_config["seed"])
    cudnn.deterministic = True

    data_config = {
        'data_dir': _config['path'][DATASET]['data_dir'],
        'dataset': DATASET,
        'eval_fold': n_fold,
        'supp_idx': _config['all_supp'][0],
        'n_sv': 1000 if DATASET == 'CMR' else 5000,
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=False,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=True)


    labels = get_label_names(DATASET)
    class_dice = {}
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name == 'BG':
            continue
        elif (not np.intersect1d([label_val], TEST_LABEL)):
            continue

        model = RE_SAM2()
        model = model.cuda()

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])

        test_dataset.label = label_val


        # Unpack support data.
        support_image = [support_sample['image'][[i]].float().cuda() for i in
                         range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W, support_image is a list {3X(1, 3, 256, 256)}
        support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
                           range(support_sample['image'].shape[0])]  # n_shot x H x W

        supp = torch.stack(support_image, dim=1)
        supp = (supp - supp.mean()) / supp.std()

        supp_mask = torch.stack(support_fg_mask, dim=1)

        criterion = nn.BCEWithLogitsLoss()

        dataset_reward = []
        with torch.no_grad():
            e_d = tqdm.tqdm(range(50))
            for i in e_d:
                noise = torch.randn([1, 1, 256, 256]).cuda() * 0.3
                supp_aug = transforms.ColorJitter([0.3,0.5],[0.3,0.5],[0.3,0.5])(supp)[0]
                supp_aug = (supp_aug - supp_aug.mean()) / supp_aug.std()
                tgt_idx = torch.randperm(_config["n_part"])
                supp_aug = torch.cat([supp_aug[tgt_idx]+noise,supp_aug[tgt_idx]])
                roll_x, roll_y = (torch.randn(2) * 3).int()
                supp_aug = supp_aug.roll(shifts=(roll_x, roll_y), dims=(-2, -1))
                target_mask = torch.cat([supp_mask.float()[0][tgt_idx], supp_mask.float()[0][tgt_idx]])
                target_mask = target_mask.roll(shifts=(roll_x, roll_y), dims=(-2, -1))
                out= model(supp_aug, supp, supp_mask)

                loss = criterion(out, target_mask)

                out_final = (out>0).float()
                if loss < 1:
                    dataset_reward.append([supp_aug,out_final,target_mask,out])
                e_d.set_description("Genarating Reward Dataset" % loss)

        reward_model = copy.deepcopy(model).cuda()
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=_config["optim"]["reward_model_lr"])
        lenth = len(dataset_reward)
        e_r = tqdm.tqdm(range(100))
        for j in e_r:
            data = dataset_reward[(j % lenth)]
            target,mask,target_mask,out_logit = data
            out_logit = torch.clamp((out_logit) / out_logit.std(),-10,10)
            qry = torch.cat([target[:,:2],out_logit[:,None]],dim=1)
            out = reward_model(qry, supp, supp_mask)

            loss = criterion(out+out_logit,target_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            e_r.set_description("Reward Model Loss:%.3f" %loss)

        policy_model = copy.deepcopy(model).cuda()
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=_config["optim"]["policy_model_lr"])
        e_p = tqdm.tqdm(range(50))
        for k in e_p:
            noise = torch.randn([1, 1, 256, 256]).cuda() * 0.3
            supp_aug = transforms.ColorJitter([0.3,0.5], [0.3,0.5], [0.3,0.5])(supp)[0]
            supp_aug = (supp_aug - supp_aug.mean()) / supp_aug.std()
            tgt_idx = torch.randperm(_config["n_part"])
            supp_aug = torch.cat([supp_aug[tgt_idx] + noise, supp_aug[tgt_idx]])
            roll_x, roll_y = (torch.randn(2) * 3).int()
            supp_aug = supp_aug.roll(shifts=(roll_x, roll_y), dims=(-2, -1))

            Action = policy_model(supp_aug, supp, supp_mask)

            # action_old
            with torch.no_grad():
                Action_old= model(supp_aug, supp, supp_mask)

            # reward
            with torch.no_grad():
                mask = torch.clamp(Action/ Action.std(), -10, 10)
                qry = torch.cat([supp_aug[:, :2], mask[:, None]], dim=1)
                Reward = reward_model(qry, supp, supp_mask)
                Reward = Reward / Reward.std()

            P1 = (Action / Action_old)
            A1 = -Reward

            loss = torch.mean((-(torch.min(P1 * A1, torch.clamp(P1, 0.8, 1.2) * A1))))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            e_p.set_description("Policy Model Loss:%.3f" %loss)

        # Loop through query volumes.
        scores = Scores()
        tbar = tqdm.tqdm(test_loader)
        for i, sample in enumerate(tbar):  # this "for" loops 4 times
            # Unpack query data.
            query_image = [sample['image'][i].float().cuda() for i in
                           range(sample['image'].shape[0])]  # [C x 3 x H x W] query_image is list {(C x 3 x H x W)}
            query_label = sample['label'].long()  # C x H x W

            query_image_ori = query_image[0]
            q_mean = query_image_ori.mean()
            q_std = query_image_ori.std()# C' x 3 x H x W
            query_image_s = (query_image_ori - q_mean)/q_std

            # RE_SAM2.
            with torch.no_grad():
                Action = policy_model(query_image_s, supp, supp_mask)

                mask = torch.clamp((Action)/Action.std(),-10,10)
                qry = torch.cat([query_image_s[:,:2],mask[:,None]],dim=1)
                Reward = reward_model(qry, supp, supp_mask)

                output = ((Action + Reward) > 0).float()

            scores.record(output.detach().cpu(), query_label)

        # Log class-wise results
        class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()

    Mean_Dice = sum(class_dice.values())/len(class_dice)

    print(f'Mean Dice: {class_dice}')
    print(f'Whole mean Dice: {Mean_Dice}')

    return Mean_Dice,np.array(list(class_dice.values())) * 100,class_dice.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Set Configurations', add_help=False)
    parser.add_argument('--dataset', default=['CHAOST2','SABS','CMR'], type=str)
    args = parser.parse_args()
    all_dataset = args.dataset if isinstance(args.dataset,list) else [args.dataset]

    dataset_record = {}
    organ_record = {}
    for dataset in all_dataset:
        ###### Data configs ######
        DATASET = dataset  # CHAOST2 SABS CMR
        TRAIN_NAME = "RE_SAM2"  # mri1 mri2 ct1 ct2 cmr
        ALL_EV = [0,1,2,3,4]  # [0,1,2,3,4]s
        if dataset == 'CMR':
            TEST_LABEL = [1, 2, 3]
        elif dataset == 'CHAOST2':
            TEST_LABEL = [1, 2, 3, 4]
        elif dataset == 'SABS':
            TEST_LABEL = [1, 2, 3, 6]

        os.makedirs(f'test_result/{TRAIN_NAME}', exist_ok=True)
        logging.basicConfig(format='%(message)s', filename=os.path.join(f'test_result/{TRAIN_NAME}', 'logger.log'),
                            level=logging.INFO)

        Dice = 0
        Organ_Dice = 0
        for EVAL_FOLD in ALL_EV:
            Dice_new, Dice_class, Class_name = main(EVAL_FOLD)
            Dice += Dice_new
            Organ_Dice += Dice_class
        print("Organ_dice: ", Organ_Dice / len(ALL_EV))
        print("result: ", Dice / len(ALL_EV))
        organ_record[str(list(Class_name))] = list((Organ_Dice / len(ALL_EV)))
        dataset_record[dataset] = Dice / len(ALL_EV)

    logging.info(f'{str(datetime.now())[:19]}   Organ Result: {organ_record}')
    logging.info(f'{str(datetime.now())[:19]}   Dataset Result: {dataset_record}')

