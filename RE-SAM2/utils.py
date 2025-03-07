"""
Utils for Dataset
Extended from ADNet and RPT code by Hansen et al and Yazhou et al.
"""
import random
import torch
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CLASS_LABELS = {
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]),  # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]),  # lower_abdomen
    },
}


def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 0
    return fg_bbox, bg_bbox


def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()


def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)


class Scores():

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.patient_dice = []
        self.patient_iou = []

    def record(self, preds, label):
        assert len(torch.unique(preds)) < 3

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        self.patient_dice.append(2 * tp / (2 * tp + fp + fn))
        self.patient_iou.append(tp / (tp + fp + fn))

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self):
        return self.TP / (self.TP + self.FP + self.FN)


def set_logger(path):
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('[%(levelname)] - %(name)s - %(message)s')
    logger.setLevel("INFO")

    # log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

# boundary loss code from https://github.com/yiskw713/boundary_loss_for_remote_sensing

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(
        n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 1.
        y_pred = torch.sigmoid(y_pred)[:, 1, :, :]
        y_true = (y_true > 0.5).float()

        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1 - dice
        return loss

def compute_dice(preds, label):
    assert len(torch.unique(preds)) < 3
    tp = torch.sum((label == 1) * (preds == 1))
    tn = torch.sum((label == 0) * (preds == 0))
    fp = torch.sum((label == 0) * (preds == 1))
    fn = torch.sum((label == 1) * (preds == 0))
    return 2 * tp/ (2 * tp + fp + fn)


def translate_image(image_tensor, tx, ty):
    batch_size, channels, height, width = image_tensor.size()

    theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float).unsqueeze(0)  # 为批次增加维度
    grid = F.affine_grid(theta, image_tensor.size(), align_corners=False)

    translated_image = F.grid_sample(image_tensor, grid.cuda(), align_corners=False)

    return translated_image