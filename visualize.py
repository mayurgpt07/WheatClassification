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
PRIMARY_DIRECTORY = './train/'
ALTERED_DIRECTORY = './train_altered/'

train_data = pd.read_csv('train.csv', sep = ',', header = 0)


le = preprocessing.LabelEncoder()
le.fit(train_data['source'])
train_data['label'] = le.transform(train_data['source'])
train_data = train_data.drop('source', axis = 1)

unique_images = list(train_data['image_id'].unique())
onlyfiles = [f for f in listdir(PRIMARY_DIRECTORY)]
onlyfiles = [f.split('.')[0] for f in onlyfiles]
print(len(onlyfiles))
image_id = train_data.loc[0,'image_id']


def create_image(train_data, image_id):
    required_df = train_data[train_data['image_id'] == image_id]
    required_df = required_df.reset_index()
    # print(required_df)

    if len(required_df) == 0:
        copyfile(PRIMARY_DIRECTORY+image_id+'.jpg', ALTERED_DIRECTORY+image_id+'.jpg')
    
    else:

        f, ax = plt.subplots()
        image = Image.open(PRIMARY_DIRECTORY+image_id+'.jpg')

        ax.imshow(image)
        ax.axis('off')
        for i in range(0, len(required_df)):
            dimensions = required_df.loc[i, 'bbox']
            dimensions = dimensions.replace('[', '').replace(']', '').split(',')
            x, y, w, h = float(dimensions[0]),float(dimensions[1]),float(dimensions[2]),float(dimensions[3])
            rect = mpl.patches.Rectangle((x,y), w, h, linewidth = 1, edgecolor='r', facecolor = 'none')
            ax.add_patch(rect)
        plt.savefig(ALTERED_DIRECTORY+image_id+'.jpg', bbox_inches='tight')
        del(f)
        del(ax)
        del(dimensions)
        plt.close()
        if len(required_df) > 40:
            del(required_df)
            time.sleep(1)


for image_id in unique_images:
    create_image(train_data, image_id)

onlyfiles_altered = [f for f in listdir(ALTERED_DIRECTORY)]
onlyfiles_altered = [f.split('.')[0] for f in onlyfiles_altered]

for i in onlyfiles:
    if i not in onlyfiles_altered:
        copyfile(PRIMARY_DIRECTORY+i+'.jpg', ALTERED_DIRECTORY+i+'.jpg')

