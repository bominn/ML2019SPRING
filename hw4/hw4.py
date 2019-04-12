import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from torchvision import transforms, utils
from torch.optim import Adam
from skimage.segmentation import slic
from skimage.color import gray2rgb, rgb2gray
from lime import lime_image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_file')
parser.add_argument('output_file')
a = parser.parse_args()

width = height = 48
f = open(a.train_file)
data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
data = np.array(data)
image = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1,1,width, height)).astype('uint8')
label = data[::width*height+1].astype('int')

print('finish load data')
image = image/255
images = image.copy()
labels = label.copy()

image = torch.FloatTensor(image)
label = torch.LongTensor(label)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 3*3*512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

train_set = TensorDataset(image, label)

model = Net()
state = torch.load('model_fit_2_430.pth')
model.load_state_dict(state['state_dict'])
model.cuda()

# saliency map
def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()
    y_pred = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()
    saliency = x.grad.abs().squeeze().data
    return saliency
def show_saliency_maps(x, y, model):
    x_org = x.squeeze().numpy()
   
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(x, y, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.detach().cpu().numpy()
    
    num_pics = x_org.shape[0]
  
    for i in range(num_pics):
        #print(i)
        # You need to save as the correct fig names
        plt.imshow(x_org[i], cmap=plt.cm.gray)
        #plt.savefig('pic_'+ str(i)+'.jpg')
        plt.imshow(saliency[i],cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig(a.output_file+'fig1_'+ str(i)+'.jpg')
        plt.close()

find = [0, 299, 2, 7, 3, 15, 4]  #each class one image in data 

show_saliency_maps(image[find], label[find], model)
print('saliency map finish')

#filter visualization, output of conv1
#use forward hook
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook
model.conv1[0].register_forward_hook(get_activation('conv1'))
model.eval()
output = model(image[0].view(-1,1,48,48).cuda())

act = activation['conv1'].detach().squeeze().cpu()

plt.figure(figsize=(20,10))
for i in range(8):
    for j in range(8):
        plt.subplot(8,8,i*8+j+1)
        plt.imshow(act[i*8+j], cmap = plt.cm.gray)
plt.savefig(a.output_file+'fig2_2.jpg')
print('conv1 output finish')
#filter visualization, Gradient Ascent : Magnify the filter response
#call a small model that only have conv1
class Netsmall(nn.Module):
    def __init__(self):
        super(Netsmall, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            
            nn.Dropout2d(0.25)
        )

    def forward(self, x):
        x = self.conv1(x)

        return x

model2 = Netsmall()
model2_dict = model2.state_dict()
state = torch.load('model_fit_2_430.pth')
pretrained_dict = state['state_dict']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model2_dict}
model2_dict.update(pretrained_dict)
model2.load_state_dict(model2_dict)

#random input
random_image = np.uint8(np.random.uniform(0, 255, (48, 48, 1)))
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
x = transform(random_image)
x = x.view(-1,1,48,48)
x.requires_grad_()
model2.cuda()
model2.eval()
optimizer = Adam([x], lr=0.01)

plt.figure(figsize=(20,10))
for i in range(64):
    random_image = np.uint8(np.random.uniform(0, 255, (48, 48, 1)))
    x = transform(random_image)
    x = x.view(-1,1,48,48)
    x.requires_grad_()
    optimizer = Adam([x], lr=0.01)
    for j in range(200):
        optimizer.zero_grad()
        output = model2(x.cuda())
        loss = -torch.sum(output[0][i])
        loss.backward()
        optimizer.step()
    plt.subplot(8,8,i+1)
    plt.imshow(x.squeeze().detach().numpy(), cmap = plt.cm.gray)
    plt.savefig(a.output_file+'fig2_1.jpg')
print('gradient accent finish')
#LIME
# two functions that lime image explainer requires
plt.figure()
def predict(input):

    # Input: image tensor
    # Returns a predict function which returns the probabilities of labels ((7,) numpy array)
    # ex: return model(data).numpy()
    # TODO:
    # return ?
    model.eval()
    input = rgb2gray(input)
    input = torch.from_numpy(input).float().view((-1,1,48,48))
    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    # Input: image numpy array
    # Returns a segmentation function which returns the segmentation labels array ((48,48) numpy array)
    # ex: return skimage.segmentation.slic()
    # TODO:
    # return ?
    return slic(input,  n_segments=100, compactness=1, sigma=1)

def explain(instance, predict_fn, segmentation_fn):
    np.random.seed(16)
    return explainer.explain_instance(image = instance, classifier_fn = predict_fn, segmentation_fn = segmentation_fn)

for idx, num in enumerate(find):
    x = gray2rgb(images[num])
    x = x.squeeze()
    explainer = lime_image.LimeImageExplainer()
    explaination = explain(x,predict,segmentation)
    image_3, mask = explaination.get_image_and_mask(
                                label=labels[num],
                                positive_only=False,
                                hide_rest=False,
                                num_features=5,
                                min_weight=0.05
                            )

# save the image
    plt.imshow(image_3)
    plt.savefig(a.output_file+'fig3_'+str(idx)+'.jpg')
print('LIME finish')
