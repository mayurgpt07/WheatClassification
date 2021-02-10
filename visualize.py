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

# for idx, img in enumerate(train_loader):

train_data = pd.read_csv('train.csv', sep = ',', header = 0)


le = preprocessing.LabelEncoder()
le.fit(train_data['source'])
train_data['label'] = le.transform(train_data['source'])
train_data = train_data.drop('source', axis = 1)

unique_images = list(train_data['image_id'].unique())
onlyfiles = [f for f in listdir('./train')]
onlyfiles = [f.split('.')[0] for f in onlyfiles]
print(len(onlyfiles))
image_id = train_data.loc[0,'image_id']


def create_image(train_data, image_id):
    required_df = train_data[train_data['image_id'] == image_id]
    required_df = required_df.reset_index()
    # print(required_df)

    if len(required_df) == 0:
        copyfile('./train/'+image_id+'.jpg', './train_altered/'+image_id+'.jpg')
    
    else:

        f, ax = plt.subplots()
        image = Image.open('./train/'+image_id+'.jpg')

        ax.imshow(image)
        ax.axis('off')
        for i in range(0, len(required_df)):
            dimensions = required_df.loc[i, 'bbox']
            dimensions = dimensions.replace('[', '').replace(']', '').split(',')
            x, y, w, h = float(dimensions[0]),float(dimensions[1]),float(dimensions[2]),float(dimensions[3])
            rect = mpl.patches.Rectangle((x,y), w, h, linewidth = 1, edgecolor='r', facecolor = 'none')
            ax.add_patch(rect)
        plt.savefig('./train_altered/'+image_id+'.jpg', bbox_inches='tight', dpi = 1024)
        del(f)
        del(ax)
        del(dimensions)
        plt.close()
        if len(required_df) > 40:
            del(required_df)
            time.sleep(1)


for image_id in unique_images[3200:3422]:
    create_image(train_data, image_id)

# img_folder_path = './train_altered/'
# my_dataset = CustomDataSet(img_folder_path, transform=transform)
# print(len(list(my_dataset)))
# train_loader = data.DataLoader(my_dataset , batch_size=16, shuffle=False, num_workers=4, drop_last=True)
