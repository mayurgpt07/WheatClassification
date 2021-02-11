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
    def __init__(self, main_dir, dataFrame, transform = None):
        self.main_dir = main_dir
        self.transform = transform
        self.dataframe = dataFrame
        all_imgs = os.listdir(main_dir)
        if ('.DS_Store' or './train/.DS_Store' or './DS_Store') in all_imgs:
            all_imgs.remove('.DS_Store')
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image_id = str(self.total_imgs[idx]).split('.')[0]
        if len(self.dataframe.loc[self.dataframe['image_id'] == image_id, 'label']) == 0:
            label = -1
        else:
            label = self.dataframe.loc[self.dataframe['image_id'] == image_id, 'label']
            label = list(label)[0]

        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        # print(torch.tensor(label))
        # print(type(tensor_image))
        print(label)
        return tensor_image, label

def aggregate_labels(df):
    aggregate_df = df.groupby(by=['image_id'])['label'].apply(lambda x: x.value_counts().index[0]).reset_index()
    return aggregate_df

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

agg_train_data = aggregate_labels(train_data)

onlyfiles = [f for f in listdir('./train')]
print(len(onlyfiles))

img_folder_path = './train/'
my_dataset = CustomDataSet(img_folder_path, agg_train_data,transform=transform)
print(len(list(my_dataset)))
train_loader = data.DataLoader(my_dataset , batch_size=1, shuffle=False, num_workers=4, drop_last = True)

# print(list(train_loader), type(train_loader))
for image, label in train_loader:
    print(image.shape, label)

# image, label = next(iter(train_loader))