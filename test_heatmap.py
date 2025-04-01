from model import Dinomaly
from util import cal_anomaly_map, image_transform,image_transform2, visualize, get_gaussian_kernel
import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

class Dataset_supervised_defect(Dataset):
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

        original_image = np.array(image.resize((448, 448)))
        if self.transform_dinomaly_image and self.transform_dinomaly_mask:
            image_din = self.transform_dinomaly_image(image)
            mask_din = self.transform_dinomaly_mask(mask)
            mask_din = (mask_din > 0).float()
        else:
            image_din = None
            mask_din = None
        
        
        return image_din, mask_din, self.image_files[idx], original_image


def test(images_dir, masks_dir, model_weights_path, output_dir,
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
    dataset = Dataset_supervised_defect(
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

    # 테스트 루프
    for idx, (images_din, mask_din, image_name, original_image) in tqdm(enumerate(dataloader), desc="Testing", total=len(dataloader)):
        images_din = images_din.to(device)

        # 모델 출력
        with torch.no_grad():
            en,de,de2 = model(images_din)
        anomaly_map1 = cal_anomaly_map(en,de)
        anomaly_map2 = cal_anomaly_map(en,de2)
        
        max1 = np.max(anomaly_map1)
        max2 = np.max(anomaly_map2)
        w1 = max2 / (max1 + max2)
        w2 = max1 / (max1 + max2)
        anomaly_map3 = w1 * anomaly_map1 + w2 * anomaly_map2
        
        
        #anomaly_map3 = 0.7*anomaly_map1+0.3*anomaly_map2
        anomaly_score1 = np.max(anomaly_map1) 
        anomaly_score2 = np.max(anomaly_map2) 
        anomaly_score3 = np.max(anomaly_map3)
        
        

        image_filename = os.path.splitext(image_name[0])[0] 
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{image_filename}.png")


        anomaly_map3 = gaussian_filter(anomaly_map3, sigma=4) 
        anomaly_map_normalized3 = (anomaly_map3 - np.min(anomaly_map3)) / (np.max(anomaly_map3) - np.min(anomaly_map3))
        anomaly_map_8bit3 = (anomaly_map_normalized3 * 255).astype(np.uint8)
        heat_map3 = cv2.applyColorMap(anomaly_map_8bit3, cv2.COLORMAP_JET)
        heat_map3 = cv2.resize(heat_map3, (448, 448))
        heat_map3 = cv2.cvtColor(heat_map3, cv2.COLOR_BGR2RGB)
        
        
        
        anomaly_map1 = gaussian_filter(anomaly_map1, sigma=4) 
        anomaly_map1_normalized = (anomaly_map1 - np.min(anomaly_map1)) / (np.max(anomaly_map1) - np.min(anomaly_map1))
        anomaly_map1_8bit = (anomaly_map1_normalized * 255).astype(np.uint8)
        heat_map1 = cv2.applyColorMap(anomaly_map1_8bit, cv2.COLORMAP_JET)
        heat_map1 = cv2.resize(heat_map1, (448, 448))
        heat_map1 = cv2.cvtColor(heat_map1, cv2.COLOR_BGR2RGB)
        

        anomaly_map2 = gaussian_filter(anomaly_map2, sigma=4) 
        anomaly_map2_normalized = (anomaly_map2 - np.min(anomaly_map2)) / (np.max(anomaly_map2) - np.min(anomaly_map2))
        anomaly_map2_8bit = (anomaly_map2_normalized * 255).astype(np.uint8)
        heat_map2 = cv2.applyColorMap(anomaly_map2_8bit, cv2.COLORMAP_JET)
        heat_map2 = cv2.resize(heat_map2, (448, 448))
        heat_map2 = cv2.cvtColor(heat_map2, cv2.COLOR_BGR2RGB)


        mask_din = mask_din.squeeze(0).squeeze(0)
        original_image = original_image.squeeze(0)
        fig, axes = plt.subplots(1, 5, figsize=(15, 10))
        
        # Display the original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Display the anomaly map
        axes[1].imshow(heat_map1)
        axes[1].set_title(f"Anomaly Score: {anomaly_score1:.2f}")
        axes[1].axis("off")

        axes[2].imshow(heat_map2)
        axes[2].set_title(f"Anomaly Score: {anomaly_score2:.2f}")
        axes[2].axis("off")
        
        axes[3].imshow(heat_map3)
        axes[3].set_title(f"Anomaly Score: {anomaly_score3:.2f}")
        axes[3].axis("off")
        
        axes[4].imshow(mask_din, cmap='gray')
        axes[4].set_title(f"Ground Truth")
        axes[4].axis("off")
        
        # Save the combined image

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved: {save_path} (Anomaly Score: {anomaly_score3:.2f})")




if __name__ == "__main__":
    
    base_dir = "/home/ohjihoon/바탕화면/app/datasets"

    class_names = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw", 
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    for class_name in class_names:
        
        images_directory = f"/home/ohjihoon/바탕화면/app/datasets/{class_name}/test/images"
        masks_directory = f"/home/ohjihoon/바탕화면/app/datasets/{class_name}/test/masks"
        model_weights_path = f"/home/ohjihoon/바탕화면/app/final_weights_noclip/{class_name}.pth"  # 저장된 모델 가중치 경로
        output_directory = f"/home/ohjihoon/바탕화면/app/test_result_heatmap_focal_nomalize_noclip/{class_name}"  # 결과 저장 폴더 경로

        test(
            images_dir=images_directory,
            masks_dir=masks_directory,
            model_weights_path=model_weights_path,
            output_dir=output_directory,
            batch_size=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

