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
        label = self.label[idx]
        return item, label

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
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

test_dataset = experimental_dataset(test_image,test_label,valid_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

ans = open(a.output_file,'w')
ans.write('id,label\n')

f= open('mobilenet.pkl','rb')
state = pickle.load(f)

for k,v in state.items():
    state[k] = torch.from_numpy(v)

device = torch.device('cuda')
model = Net()
model.load_state_dict(state)
model.to(device)

model.eval()
with torch.no_grad():
    for idx, (data, target)in enumerate(test_loader):
        img_cuda = data.to(device)
        outputs = model(img_cuda)
        predicted = torch.max(outputs.data, 1)[1]
        ans.write('{},{}\n'.format(idx, predicted[0]))
ans.close()

print('finish')