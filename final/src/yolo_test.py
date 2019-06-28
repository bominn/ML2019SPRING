import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from yolo_models import Yolov1_vgg16bn, vgg19_bn
from yololoss import yoloLoss
from yolo_dataset import yolodataset, testdataset
from yolo_predict import decoder
import numpy as np


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def main():
    use_gpu = torch.cuda.is_available()

    test_root = 'test/'



    batch_size = 16

    model = vgg19_bn()
    print('load pre-trained model')

    load_checkpoint('models/yolo.pth', model)



    criterion = yoloLoss(7,2,5,0.5)

    if use_gpu:
        model.cuda()

    model.eval()

    valid_dataset = testdataset(root=test_root, transform = transforms.ToTensor())
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    print('start test')
    f = open('pred.csv','w')
    f.write('patientId,x,y,width,height,Target\n')

    with torch.no_grad():
        for idx, (images,name) in enumerate(valid_loader):
            images = Variable(images)
            if use_gpu:
                images = images.cuda()
            
            pred = model(images)

            keep = decoder(pred.cpu())
            
            pred = pred.squeeze().cpu()
            if keep == []:
                f.write('{},,,,,{}\n'.format(str(name[0]),0))
            else:
                for i in range(len(keep)):
                    num_cell = keep[i][0]
                    xmin, weight = (keep[i][1][0]), (keep[i][1][2]-keep[i][1][0])
                    ymin, height = (keep[i][1][1]), (keep[i][1][3]-keep[i][1][1])
                    xmin += 0.05*weight
                    ymin += 0.05*height
                    weight*=0.9
                    height*=0.9
                    f.write('{},{},{},{},{},{}\n'.format(str(name[0]),xmin,ymin,weight,height,1))
            
        f.close()     
        print('finish')

if __name__=='__main__':
    main()
