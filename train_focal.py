import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import Dinomaly 
from tqdm import tqdm
from util import cal_anomaly_map, global_cosine_hm_percent
from optimizers import StableAdamW
import torch.nn.functional as F
import numpy as np
from loss import FocalLoss, FocalLoss_smoothl1


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
    
def get_class_name(images_dir):

    return os.path.basename(os.path.dirname(os.path.dirname(images_dir)))

class Dataset_unsupervised(Dataset):
    def __init__(self, images_dir, transform_dinomaly_image=None):
        self.images_dir = images_dir
        self.transform_dinomaly_image = transform_dinomaly_image

        self.image_files = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')) and 'good' in f.lower()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])

        image = Image.open(img_path).convert('RGB')

        if self.transform_dinomaly_image:
            image_din = self.transform_dinomaly_image(image)
        else:
            image_din = None


        return image_din

class Dataset_supervised(Dataset):
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

        # and 'good' not in f.lower()

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

        if self.transform_dinomaly_image and self.transform_dinomaly_mask:
            image_din = self.transform_dinomaly_image(image)
            mask_din = self.transform_dinomaly_mask(mask)
            mask_din = (mask_din > 0).float()
        else:
            image_din = None
            mask_din = None

        
        return image_din, mask_din


def train_combined_model(images_dir, masks_dir, total_iters=5000, batch_size=8, learning_rate=2e-3
                        , device='cuda' if torch.cuda.is_available() else 'cpu', dinomaly_weight_path = None, mode = ''):
    
    class_name = get_class_name(images_dir)
    
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
    if mode == 'good':
        dataset = Dataset_unsupervised(
            images_dir=images_dir, 
            transform_dinomaly_image=transform_dinomaly_image)
    
    elif mode == 'anomaly':
        dataset = Dataset_supervised(
            images_dir=images_dir,
            masks_dir=masks_dir, 
            transform_dinomaly_image=transform_dinomaly_image,
            transform_dinomaly_mask=transform_dinomaly_mask)
    
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(len(dataloader))
    model = Dinomaly(weight=dinomaly_weight_path).to(device)
    
    
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    
    if mode == 'good':
        for param in model.trainable.parameters():
            param.requires_grad = True
        for param in model.trainable2.parameters():
            param.requires_grad = False  
    elif mode == 'anomaly':
        for param in model.trainable.parameters():
            param.requires_grad = False
        for param in model.trainable2.parameters():
            param.requires_grad = True
    if mode == 'good':
        optimizer = StableAdamW([{'params': model.trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    elif mode == 'anomaly':
        optimizer = StableAdamW([{'params': model.trainable2.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    it = 0
    model.train()
    criterion = FocalLoss()
    
    
    for epoch in range(int(np.ceil(total_iters / len(dataloader)))):
        loss_list = []
        
        if mode == 'good':
            for image in tqdm(dataloader, desc=f"it {it+1}/{total_iters}"):
                image = image.to(device)
                en,de,de2 = model(image)
                p_final = 0.9
                p = min(p_final * it / 1000, p_final)

                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1) 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(model.trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                it += 1
                if it == total_iters:
                    break
        
        elif mode == 'anomaly':
            for image, mask in tqdm(dataloader, desc=f"it {it+1}/{total_iters}"):
                image = image.to(device)
                mask = mask.to(device)
                en,de,de2 = model(image)
                p_final = 0.9
                p = min(p_final * it / 1000, p_final)

                loss = criterion(cal_anomaly_map(en,de2), mask)
                
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm(model.trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                it += 1
                if it == total_iters:
                    break
            
            
        print('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
    if mode == 'good':
        save_path = f"/home/ohjihoon/바탕화면/app/datasets2_weights/head1_weights/{class_name}.pth"
    elif mode == 'anomaly':
        save_path = f"/home/ohjihoon/바탕화면/app/datasets2_weights/final_weights2/{class_name}.pth"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"모델 가중치가 저장되었습니다: {save_path}")
    print("학습 완료!")

if __name__ == "__main__":
    base_dir = "/home/ohjihoon/바탕화면/app/datasets2"
    class_names = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw", 
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    for class_name in class_names:
        images_directory = os.path.join(base_dir, class_name, "train/images")
        masks_directory = os.path.join(base_dir, class_name, "train/masks")
        
        train_combined_model(images_dir=images_directory, masks_dir=masks_directory, total_iters=5000, batch_size=4, learning_rate=1e-4, dinomaly_weight_path='', mode = 'good')
        dinomaly_weight_path = "/home/ohjihoon/바탕화면/app/datasets2_weights/head1_weights/{}.pth"
        dinomaly_weight_path = dinomaly_weight_path.format(class_name)
        train_combined_model(images_dir=images_directory, masks_dir=masks_directory, total_iters=5000, batch_size=4, learning_rate=1e-4, dinomaly_weight_path=dinomaly_weight_path, mode = 'anomaly')
        
