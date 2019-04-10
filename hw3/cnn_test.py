import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test_file')
parser.add_argument('output_file')

a = parser.parse_args()

width = height = 48
'''
f = open('train.csv')
data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
data = np.array(data)
image = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1,width, height, 1)).astype('uint8')
label = data[::width*height+1].astype('int')

mean = image.mean()/255
std = image.std()/255
'''
mean = 0.5077425080522144
std = 0.25500891562522027


fk = open(a.test_file)
data = fk.read().strip('\r\n').replace(',', ' ').split()[2:]
data = np.array(data)
test_image = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1,width, height, 1)).astype('uint8')

print('read data finish')

test_label = np.zeros(shape = (7178,)) # all = 0 just for class experimental_dataset's label

class experimental_dataset(Dataset):
    
    def __init__(self, data, label, transform):
        self.data = data
        self.transform = transform
        self.label = torch.from_numpy(label).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item)
        #label = torch.from_numpy(label).long()
        label = self.label[idx]
        return item, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(0, translate=(0.2,0.2), scale=(0.8,1.2), shear=None, fillcolor=0),
    transforms.RandomRotation(30, resample=False, expand=False, center=None),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([mean], [std], inplace=False)
])
valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomAffine(30, translate=(0.2,0.2), scale=(0.8,1.2), shear=None, fillcolor=0),
    #transforms.RandomRotation(30, resample=False, expand=False, center=None),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([mean], [std], inplace=False)
])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 3*3*512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

test_dataset = experimental_dataset(test_image,test_label,valid_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

ans = open(a.output_file,'w')
ans.write('id,label\n')

print('predict test data')
device = torch.device('cuda')
model = Net()
optimizer = Adam(model.parameters(), lr=0.001)
state = torch.load('model_fit_5_277.pth')
model.load_state_dict(state['state_dict'])
model.to(device)
optimizer.load_state_dict(state['optimizer'])

model.eval()
with torch.no_grad():
    for idx, (data, target)in enumerate(test_loader):
        img_cuda = data.to(device)
        outputs = model(img_cuda)
        predicted = torch.max(outputs.data, 1)[1]
        #print('id={},out={}'.format(idx, predicted[0]))
        ans.write('{},{}\n'.format(idx, predicted[0]))
ans.close()

print('finish')
