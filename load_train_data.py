import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils import data
from torchvision import transforms
from natsort import natsorted
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import time
import os

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform = None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        if ('.DS_Store' or './train_altered/.DS_Store' or './DS_Store') in all_imgs:
            all_imgs.remove('.DS_Store')
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#
    ])


train_data = pd.read_csv('train.csv', sep = ',', header = 0)


le = preprocessing.LabelEncoder()
le.fit(train_data['source'])
train_data['label'] = le.transform(train_data['source'])
train_data = train_data.drop('source', axis = 1)

onlyfiles = [f for f in listdir('./train_altered')]
# onlyfiles = [f.split('.')[0] for f in onlyfiles]
print(len(onlyfiles))

img_folder_path = './train_altered/'
my_dataset = CustomDataSet(img_folder_path, transform=transform)
print(len(list(my_dataset)))
train_loader = data.DataLoader(my_dataset , batch_size=32, shuffle=False, num_workers=4, drop_last=True)
