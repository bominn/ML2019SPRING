import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_file')

a = parser.parse_args()

width = height = 48
f = open(a.train_file)
data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
data = np.array(data)
image = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1,width, height, 1)).astype('uint8')
label = data[::width*height+1].astype('int')

mean = image.mean()/255
std = image.std()/255

train_image, valid_image = image[:-1000], image[-1000:]
train_label, valid_label = label[:-1000], label[-1000:]

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
    transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    transforms.RandomRotation(15, resample=False, expand=False, center=None),
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

train_dataset = experimental_dataset(train_image,train_label,transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

valid_dataset = experimental_dataset(valid_image,valid_label,valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

accuracy = 0.67
device = torch.device('cuda')
model = Net()
optimizer = Adam(model.parameters(), lr=0.001)

#state = torch.load('model_fit_5_227.pth')
#model.load_state_dict(state['state_dict'])
model.to(device)
#optimizer.load_state_dict(state['optimizer'])

#optimizer = Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()
print(model)
num_epoch = 1
for epoch in range(num_epoch):
    model.train()
    train_loss = []
    train_acc = []
    for batch_idx, (data, target) in enumerate(train_loader):
        img_cuda = data.to(device)
        target_cuda = target.to(device)
        optimizer.zero_grad()
        
        output = model(img_cuda)
        loss = loss_fn(output, target_cuda)
        loss.backward()
        optimizer.step()
        predict = torch.max(output, 1)[1]
        acc = np.mean((target_cuda == predict).cpu().numpy())

        train_acc.append(acc)
        train_loss.append(loss.item())
       
    #print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
    
    print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
    
    model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_acc = []
        for idx,(data, target) in enumerate(valid_loader):
            img_cuda = data.to(device)
            target_cuda = target.to(device)
            output = model(img_cuda)
            loss = loss_fn(output, target_cuda)
            predict = torch.max(output, 1)[1]
            acc = np.mean((target_cuda == predict).cpu().numpy())
            valid_loss.append(loss.item())
            valid_acc.append(acc)
        print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
       
    if np.mean(valid_acc) > accuracy:
        #save model
        accuracy = np.mean(valid_acc)
        checkpoint_path = 'model_'+str(epoch+1)+'.pth'
        state = {'state_dict': model.state_dict(),
                 'optimizer' : optimizer.state_dict()}
        torch.save(state, checkpoint_path)
        print('model saved to %s' % checkpoint_path)

print('finish')