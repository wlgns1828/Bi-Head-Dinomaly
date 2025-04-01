import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import cv2 
import numpy as np
import math

from model import Dinomaly
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
from util import compute_pro, f1_score_max
# 테스트 함수 정의


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


def cal_anomaly_map(fs_list, ft_list, out_size=392):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
        
    anomaly_map = torch.zeros((fs_list[0].shape[0], 1, out_size[0], out_size[1]), device=fs_list[0].device)
        
    for fs, ft in zip(fs_list, ft_list):
        cos_sim = F.cosine_similarity(fs, ft, dim=1)
        a_map = 1 - cos_sim  # (B, H, W)
        a_map = a_map.unsqueeze(1)  # (B, 1, H, W)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map
        
    return anomaly_map

class Dataset_eval(Dataset):
    def __init__(self, images_dir, masks_dir, 
                transform_dinomaly_image=None, transform_dinomaly_mask=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_dinomaly_image = transform_dinomaly_image
        self.transform_dinomaly_mask = transform_dinomaly_mask

        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')) 
        ])

        self.mask_files = sorted([
            f for f in os.listdir(masks_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')) 
        ])

        assert len(self.image_files) == len(self.mask_files), "이미지와 마스크의 수가 일치하지 않습니다."
        print("불러온 이미지 개수:", len(self.image_files))
        print("불러온 마스크 개수:", len(self.mask_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 마스크는 흑백 이미지
        
        label = 0 if 'good' in self.image_files[idx].lower() else 1
        if self.transform_dinomaly_image and self.transform_dinomaly_mask:
            image_din = self.transform_dinomaly_image(image)
            mask_din = self.transform_dinomaly_mask(mask)
            mask_din = (mask_din > 0).float()
        else:
            image_din = None
            mask_din = None

        return image_din, mask_din, label


    
    
def eval(images_dir, masks_dir, model_weights_path, output_dir,
                        batch_size=1, device='cuda' if torch.cuda.is_available() else 'cpu'):


    # Dinomaly에 적용할 변환 정의
    transform_dinomaly_image = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),        
        transforms.CenterCrop(392),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    ])

    transform_dinomaly_mask = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.CenterCrop(392),
    ])

    # 데이터셋 및 데이터로더 초기화
    dataset = Dataset_eval(
        images_dir=images_dir, 
        masks_dir=masks_dir, 
        transform_dinomaly_image=transform_dinomaly_image,
        transform_dinomaly_mask=transform_dinomaly_mask
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 모델 로드
    model = Dinomaly()  
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []

    max_ratio = 0.01
    resize_mask = 256
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    with torch.no_grad():
    # 테스트 루프
        for idx, (images_din, mask_din, label) in tqdm(enumerate(dataloader), desc="Testing", total=len(dataloader)):
            images_din = images_din.to(device)

            with torch.no_grad():
                en,de,de2 = model(images_din)
            anomaly_map1 = cal_anomaly_map(en,de)
            anomaly_map2 = cal_anomaly_map(en,de2)
            gt = mask_din        
            max1 = torch.max(anomaly_map1)
            max2 = torch.max(anomaly_map2)
            w1 = max2 / (max1 + max2)
            w2 = max1 / (max1 + max2)
            anomaly_map3 = w1 * anomaly_map1 + w2 * anomaly_map2
            
            if resize_mask is not None:
                anomaly_map3 = F.interpolate(anomaly_map3, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            anomaly_map3 = gaussian_kernel(anomaly_map3)        
            

            gt = gt.bool()
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map3)
            gt_list_sp.append(label)
            if max_ratio == 0:
                sp_score = torch.max(anomaly_map3.flatten(1), dim=1)[0]
            else:
                anomaly_map3 = anomaly_map3.flatten(1)
                sp_score = torch.sort(anomaly_map3, dim=1, descending=True)[0][:, :int(anomaly_map3.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1)
            if torch.isnan(sp_score).any():
                print(f"NaN detected in sp_score at index {idx}")
            pr_list_sp.append(sp_score)

        gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
        pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

        aupro_px = compute_pro(gt_list_px, pr_list_px)

        gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

        auroc_px = roc_auc_score(gt_list_px, pr_list_px)
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_px = average_precision_score(gt_list_px, pr_list_px)
        ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

        f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
        f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

if __name__ == "__main__":
    base_dir = "/home/ohjihoon/바탕화면/app/datasets"
    class_names = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw", 
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]

    results_list = []
    for class_name in class_names:
        images_directory = os.path.join(base_dir, class_name, "test/images")
        masks_directory = os.path.join(base_dir, class_name, "test/masks")
        model_weights_path = f"/home/ohjihoon/바탕화면/app/final_weights_noclip/{class_name}.pth"  # 저장된 모델 가중치 경로
        output_directory = "/home/ohjihoon/바탕화면/app/test_result_heatmap_focal_nomalize"  # 결과 저장 폴더 경로
        
        model_weights_path = model_weights_path.format(class_name)
        

        results = eval(
            images_dir=images_directory,
            masks_dir=masks_directory,
            model_weights_path=model_weights_path,
            output_dir=output_directory,
            batch_size=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
    )
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

        # 결과를 리스트에 추가
        results_list.append([
            class_name, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px
        ])

        print(f"{class_name} - I-Auroc:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}")

    results_df = pd.DataFrame(results_list, columns=[
    "Class", "I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"
    ])
    output_csv_path = os.path.join(output_directory, "evaluation_results_bi_head.csv")
    results_df.to_csv(output_csv_path, index=False, float_format='%.4f')

    print(f"Evaluation results saved to {output_csv_path}")
