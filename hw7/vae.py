import torch
import numpy as np
from skimage.io import imread, imsave
import torch.nn as nn
import pandas as pd
import pickle
import glob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('test_case')
parser.add_argument('output_file')
a = parser.parse_args()

class hw7dataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.fname = sorted(glob.glob(os.path.join(root,'*.jpg')))
    def __len__(self):
        return len(self.fname)
    def __getitem__(self,idx):
        img = imread(self.fname[idx])
        img = self.transform(img)
        
        return img

# vae model
class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        
        self.latent_size = latent_size
        self.conv_stage = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.fcMean = nn.Linear(1024, self.latent_size)
        self.fcStd = nn.Linear(1024, self.latent_size)
        
        self.fcDecode = nn.Linear(self.latent_size,1024)
        
        self.trans_conv_stage = nn.Sequential(

            nn.ConvTranspose2d(256, 128, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),
 
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
 
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.ConvTranspose2d(32, 3, 4, 2, padding=1,bias=False)
        )
        # final output activation function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def encode(self, x):
        conv_output = self.conv_stage(x).view(-1, 1024)
        return self.fcMean(conv_output), self.fcStd(conv_output)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        #eps = Variable(eps).cuda()
        #eps = eps.cuda()
        eps.requires_grad=True
        eps = eps.cuda()
        #print(eps.requires_grad)
        return eps.mul(std).add_(mu)


    def decode(self, z):
        fc_output = self.fcDecode(z).view(-1, 256, 2, 2)
        trans_conv_output = self.trans_conv_stage(fc_output)
        return self.tanh(trans_conv_output)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# vae loss
def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    
    loss = nn.L1Loss(reduction='sum')
#     MSE = F.mse_loss(recon_x, x, size_average=False)
    l1_loss = loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return l1_loss + KLD, KLD, l1_loss

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

latent_size = 100
model = VAE(latent_size)

load_checkpoint('vae4.pth', model)
model.cuda()
model.eval()

dataset = hw7dataset(root = a.img_path ,transform=transforms.ToTensor())


latent_sapce = []
new_loader = DataLoader(dataset,batch_size=256, shuffle=False)
for idx, (img) in enumerate(new_loader):
    _,mu,var = model(img.cuda())
    mu = mu.detach().cpu().numpy()
    for i in range(mu.shape[0]):
        latent_sapce.append(mu[i])
   
print('latent_space finish')
latent_space = np.asarray(latent_sapce)


pca = PCA(n_components=100, copy=False, whiten=True, svd_solver='full')
latent_vec = pca.fit_transform(latent_space)

kmeans = KMeans(n_clusters=2, random_state=2, max_iter=1000).fit(latent_vec)

#load kmeans for reproduce
#f = open('kmean.pkl','rb')
#kmeans = pickle.load(f)
#f.close()

#kmeans.labels_ = kmeans.predict(latent_vec)


#read test case
test = pd.read_csv(a.test_case)
t = test.iloc[:,[1,2]]
t = np.asarray(t)

#predict answer
ans = open(a.output_file,'w')
ans.write('id,label\n')
for i in range(t.shape[0]):
    l1 = kmeans.labels_[t[i][0]-1]
    l2 = kmeans.labels_[t[i][1]-1]
    
  
    if l1 == l2:
        ans.write('{},{}\n'.format(str(i),str(1)))
    else:
        ans.write('{},{}\n'.format(str(i),str(0)))
ans.close()    

print('finish')