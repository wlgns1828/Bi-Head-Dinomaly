import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import cv2
import os
import math
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score

import pandas as pd
from sklearn.metrics import auc
from skimage import measure
from statistics import mean
from numpy import ndarray

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='add', norm_factor=None):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        if norm_factor is not None:
            a_map = 0.1 * (a_map - norm_factor[0][i]) / (norm_factor[1][i] - norm_factor[0][i])

        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map

def image_transform(image):
    transform_image = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),        
        #transforms.CenterCrop(392),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    
    image = transform_image(image)
    return image

def image_transform2(image):
    transform_image = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),        
        transforms.CenterCrop(392),
    ])
    
    image = transform_image(image)
    return image


def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x

def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def visualize(original_image_np, add_result, anomaly_score, save_path, image_name):



    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Anomaly Score : {anomaly_score:.4f}", fontsize=16, fontweight="bold")
    axes[0].imshow(original_image_np)
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(add_result,vmin=0, vmax=1)
    axes[1].set_title("Add Result")
    axes[1].axis('off')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)  
    save_path = os.path.join(save_path, image_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig) 
    print(f"Figure saved to {save_path}")


def visualize2(original_images, anomaly_map, anomaly_map2, save_path, image_name):
    # Tensor를 CPU로 옮기고 numpy로 변환
    original_images_np = original_images.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    anomaly_map_np = anomaly_map.squeeze(0).squeeze(0).detach().cpu().numpy()
    anomaly_map_np2 = anomaly_map2.squeeze(0).squeeze(0).detach().cpu().numpy()

    add_result = 0.5*anomaly_map_np+0.5*anomaly_map_np2
    anomaly_score = np.max(add_result)
    # Matplotlib를 이용한 시각화
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(f"Anomaly Score : {anomaly_score:.4f}", fontsize=16, fontweight="bold")
    axes[0].imshow(original_images_np)
    axes[0].set_title("Image")
    axes[0].axis('off')

    # Heatmaps with fixed vmin and vmax
    axes[1].imshow(anomaly_map_np, vmin=0, vmax=1)
    axes[1].set_title("Unsupervised Head")
    axes[1].axis('off')

    axes[2].imshow(anomaly_map_np2,vmin=0, vmax=1.5)
    axes[2].set_title("Supervised Head")
    axes[2].axis('off')

    axes[3].imshow(add_result,vmin=0, vmax=1)
    axes[3].set_title("Add Result")
    axes[3].axis('off')
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    save_path = os.path.join(save_path, image_name)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # 메모리 누수 방지를 위해 plt를 닫아줍니다.
    print(f"Figure saved to {save_path}")



def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    rows = []
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # append 대신 리스트에 딕셔너리 추가
        rows.append({"pro": mean(pros), "fpr": fpr, "threshold": th})

    # 리스트를 DataFrame으로 변환
    df = pd.DataFrame(rows, columns=["pro", "fpr", "threshold"])

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

