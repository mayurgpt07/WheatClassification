import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
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
ALTERED_DIRECTORY = './train_alter/'


class CustomDataSet(Dataset):
    def __init__(self, main_dir, dataFrame, read_mode = 'train', transform = None):
        self.main_dir = main_dir
        self.transform = transform
        self.dataframe = dataFrame
        self.read_mode = read_mode
        all_imgs = os.listdir(main_dir)
        if ('.DS_Store' or './train_alter/.DS_Store' or './DS_Store') in all_imgs:
            all_imgs.remove('.DS_Store')
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        if self.read_mode == 'train':
            image_id = str(self.total_imgs[idx]).split('.')[0]
            if len(self.dataframe.loc[self.dataframe['image_id'] == image_id, 'label']) == 0:
                label = 7
            else:
                label = self.dataframe.loc[self.dataframe['image_id'] == image_id, 'label']
                label = list(label)[0]
            return tensor_image, label
        elif self.read_mode == 'test':
            return tensor_image


class Unit(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        

        self.conv = torch.nn.Conv2d(in_channels=in_channels,kernel_size=5,out_channels=out_channels,padding=2)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace = True)

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(torch.nn.Module):
    def __init__(self,num_classes):
        super(SimpleNet,self).__init__()
        
        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = torch.nn.AvgPool2d(kernel_size=4)
        
        #Add all the units into the Sequential layer in exact order
        self.net = torch.nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = torch.nn.Linear(in_features=128*8*8,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        # print(output.shape)
        output = output.view(-1,128*8*8)
        output = self.fc(output)
        return output

def train_model(epochs, train_loader, model):
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for i, (image, label) in enumerate(train_loader):
            # print(label)
            
            optimizer.zero_grad()
            outputs = model(image)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * image.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == label.data)
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))

def test_model(test_loader, model):
    model.eval()
    test_acc = 0
    for i, (image, label) in test_loader:
        outputs = model(image)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == label.data)

    print(test_acc)



def aggregate_labels(df):
    aggregate_df = df.groupby(by=['image_id'])['label'].apply(lambda x: x.value_counts().index[0]).reset_index()
    return aggregate_df

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


train_data = pd.read_csv('train.csv', sep = ',', header = 0)


le = preprocessing.LabelEncoder()
le.fit(train_data['source'])
train_data['label'] = le.transform(train_data['source'])
train_data = train_data.drop('source', axis = 1)
# train_data['label'] = train_data['label'].apply(lambda x: x+1)
# print(train_data['label'].unique())
agg_train_data = aggregate_labels(train_data)

onlyfiles = [f for f in listdir(ALTERED_DIRECTORY)]
print(len(onlyfiles))

img_folder_path = ALTERED_DIRECTORY
my_dataset = CustomDataSet(img_folder_path, agg_train_data, 'train',transform=transform)

train_dataset, test_dataset = random_split(my_dataset, [160, 32])
print(len(list(my_dataset)))
train_loader = data.DataLoader(train_dataset , batch_size=32, shuffle=False, num_workers=4, drop_last = True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last = True)
model = SimpleNet(num_classes=8)
train_model(10, train_loader, model)
test_model(test_loader, model)