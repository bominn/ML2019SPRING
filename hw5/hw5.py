import torch
import numpy as np
import pandas as pd
import copy
import torchvision.models as models
#import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
parser.add_argument('output_dir')
a = parser.parse_args()
#label = pd.read_csv('hw5_data/labels.csv')

#label = label.iloc[:,3]

#label = np.asarray(label)
#np.save('label.npy', label)

label = np.load('label.npy')


vgg16 = models.resnet101(pretrained=True)
#print(vgg16)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root
        self.label = torch.from_numpy(label).long()
        self.transforms = transforms
        self.fnames = []

        for i in range(200):
             self.fnames.append('{:03}'.format(i))
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root,self.fnames[idx]+'.png'))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return 200

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std, inplace=False)
])
image_root = a.input_dir
image_dataset = adverdataset(image_root, label, transform)
train_loader = DataLoader(image_dataset, batch_size=1, shuffle=False)

def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    
    return recreated_im

class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.cuda()
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        if not os.path.isdir(a.output_dir):
            os.makedirs(a.output_dir)

    def generate(self, original_image, im_label, idx):

        img = original_image
        label = im_label
        ce_loss = nn.CrossEntropyLoss()
        # Start iteration
        for i in range(20):
           
            img.requires_grad_()
            img.grad = None
            # Forward pass
            out = self.model(img.cuda())
            # Calculate CE loss
            pred_loss = ce_loss(out, label.cuda())
            # Do backward pass
            pred_loss.backward()
           
            # Create Noise
           
          
            adv_noise = self.alpha * torch.sign(img.grad.data)
            # Add Noise to processed image
            img.data = img.data + adv_noise
            new_out = self.model(img.cuda())

        '''
            if new_out.data.max(1)[1] != label.cuda():
            
                new_img = recreate_image(img)
                new_img = Image.fromarray(new_img)
                path = a.output_dir+'{:03}'.format(idx)+'.png'
                if os.path.isfile(path):
                    os.remove(path)
                new_img.save(path)
                new_img.close()
        # check
                tmp = Image.open(a.output_dir+'{:03}'.format(idx)+'.png')
                tmp = transform(tmp)
                tmp = tmp.view(1,-1,224,224)
                o = vgg16(tmp.cuda())
        
                if o.data.max(1)[1] != label.cuda():
                    #print('?')
                    return 1
        #else:
        #    print('{} not success'.format(str(idx)))
        '''
        new_img = recreate_image(img)
        new_img = Image.fromarray(new_img)
        path = os.path.join(a.output_dir,'{:03}'.format(idx)+'.png')
        if os.path.isfile(path):
            os.remove(path)
        new_img.save(path)
        new_img.close()
        return 0        

ans = 0
for idx,(img, target) in enumerate(train_loader):
        if (idx+1)%10==0:
            print(idx)
        z = FastGradientSignUntargeted(vgg16, 0.01)
        k = z.generate(img, target, idx)
        ans+=k
       

print(ans/200)    