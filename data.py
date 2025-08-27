# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import glob
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import PIL
from skimage import io, exposure
from torch.utils.data import DataLoader,random_split,Dataset, ConcatDataset ,SubsetRandomSampler 
import random

label2idx = {
    'R1': 0,
    'R2': 1,
    'R3': 2
}


class OCTDataset(Dataset):
    def __init__(self, root_dir, patient_csv, transform=None):
        self.red_ch_root = root_dir + '/R'
        self.folder_list = glob.glob(self.red_ch_root + '*/*')
        self.transform = transform
        self.patient_data = pd.read_excel(patient_csv)[['newnum', 'BCVAlogmar']]
        self.patient_data['newnum'] = 'R' + self.patient_data['newnum'].astype(str)

        

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        
        path = self.folder_list[idx]
        red_ch_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # red_ch_img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        # red_ch_img = cv2.equalizeHist(red_ch_img)
        
        # print(path)
        # print(path.split('\\'))
        root_folder, img_ch, img_name = path.split('\\')
        
        label = label2idx[img_ch]
        patient_name = img_name.split(".")[0]
        visual_acuity = self.patient_data[self.patient_data['newnum'] == str(patient_name)]['BCVAlogmar'].values[0]

        img_name = img_name.replace('R', 'B')
        img_ch = img_ch.replace('R', 'B')

        path = os.path.join(root_folder, img_ch, img_name)
        blue_ch_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # blue_ch_img = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # blue_ch_img = cv2.equalizeHist(blue_ch_img)

        img_name = img_name.replace('B', 'G')
        img_ch = img_ch.replace('B', 'G')

        path = os.path.join(root_folder, img_ch, img_name)
        green_ch_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # green_ch_img = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # green_ch_img = cv2.equalizeHist(green_ch_img)


        # img = np.stack([red_ch_img, blue_ch_img, green_ch_img], axis=-1)
        img = np.stack([red_ch_img, green_ch_img, blue_ch_img], axis=-1)


        # # crop the unwanted metadata of below the image 
        # img = img[:770, :]
        # # print(img.shape)

        if self.transform:
            img = self.transform(img)
        

        return img, label, visual_acuity, patient_name
        
class RandomApply:
    def __init__(self, transforms_list, p=0.5):
        self.transforms_list = transforms_list
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            for t in self.transforms_list:
                img = t(img)
        return img



def prepare_data(batch_size=4, train_dir='train', test_dir='tests'):
    train_transform = transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.RandomHorizontalFlip(p=0.8),
        # torchvision.transforms.RandomVerticalFlip(p=0.8),       
        # RandomApply([torchvision.transforms.RandomAutocontrast()], p=0.4), 
        # RandomApply([torchvision.transforms.RandomRotation(degrees=15)], p=0.6), 
        # torchvision.transforms.RandomInvert(p=0.1),
        # RandomApply([torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.4),
        # Color jitter for brightness, contrast, saturation, and hue
        # RandomApply([torchvision.transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0)], p=0.5),    
        # Random perspective transformation
        # torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.2, interpolation=3),        
        # Random affine transformation
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Randomly erase a portion of the image
        # transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        # transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])

    trainset = OCTDataset(root_dir=train_dir, transform=train_transform, patient_csv="Cataractfull.xlsx")


    # dataset_length = len(trainset)
    # valid_length = int(0.15 * dataset_length)
    # train_length = dataset_length - valid_length
    # print("Train length", train_length)
    # print("Valid length", valid_length)

    
    # train_dataset, valid_dataset = random_split(trainset, [train_length, valid_length])
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)


    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])


    testset = OCTDataset(root_dir=test_dir, transform=test_transform, patient_csv="Cataractfull.xlsx")
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)


    classes = ('normal_cataract', 'mild_cataract', 'severe_cataract')
    return train_loader, test_loader, classes

