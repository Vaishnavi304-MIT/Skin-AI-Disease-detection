import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SkinDiseaseDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 224):
        self.img_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            print(f"Directory '{root_dir}' does not exist.")
            return

        # Locate class subfolders inside the target directory
        class_names = sorted([
            f for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")

        for name in class_names:
            class_folder = os.path.join(root_dir, name)
            for ext in valid_exts:
                for path in glob.glob(os.path.join(class_folder, ext)):
                    self.img_paths.append(path)
                    self.labels.append(self.class_to_idx[name])

        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image file at: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(image=image)['image']
        return tensor_img, torch.tensor(self.labels[idx], dtype=torch.long)