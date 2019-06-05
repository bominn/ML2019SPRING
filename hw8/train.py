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
    transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1), shear=0, fillcolor=0),
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

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(negative_slope=0.05)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.LeakyReLU(negative_slope=0.05),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(negative_slope=0.05),
            )

        self.model = nn.Sequential(
            conv_bn(  1,  64, 1), 
            conv_dw( 64,  128, 2),
            conv_dw( 128, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 256, 2),
            nn.AvgPool2d(3),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(100, 7)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

train_dataset = experimental_dataset(train_image,train_label,transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

valid_dataset = experimental_dataset(valid_image,valid_label,valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

accuracy = 0.6
device = torch.device('cuda')
model = Net()
optimizer = Adam(model.parameters(), lr=0.001)
model.to(device)
loss_fn = nn.CrossEntropyLoss()

#print(model)

num_epoch = 50000
for epoch in range(num_epoch):
    model.train()
    train_loss = []
    train_acc = []
    for batch_idx, (data, target) in enumerate(train_loader):
        #data = data.half()
        img_cuda = data.to(device)
        target_cuda = target.to(device)
        optimizer.zero_grad()
        
        output = model(img_cuda)
        #print(output.size())
        loss = loss_fn(output, target_cuda)
        loss.backward()
        optimizer.step()
        predict = torch.max(output, 1)[1]
        acc = np.mean((target_cuda == predict).cpu().numpy())

        train_acc.append(acc)
        train_loss.append(loss.item())
        break
    #print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
    if (epoch+1)%100 == 0:
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
        if (epoch+1)%100 == 0:
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
       
    if np.mean(valid_acc) > accuracy:
        #save model
        accuracy = np.mean(valid_acc)
        print(accuracy)
        checkpoint_path = 'model_6_'+str(epoch+1)+'.pth'
        state = model.state_dict()
                 
        torch.save(state, checkpoint_path)
        print('model saved to %s' % checkpoint_path)
    
print('finish')