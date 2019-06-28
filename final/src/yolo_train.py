import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from yolo_models import Yolov1_vgg16bn, vgg19_bn
from yololoss import yoloLoss
from yolo_dataset import yolodataset
import timeit
from yolo_predict import decoder
import numpy as np

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


def main():
    use_gpu = torch.cuda.is_available()

    train_root = 'train/'
    
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 8

    model = vgg19_bn()
    optimizer = torch.optim.SGD([{"params":model.parameters()}], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    print('load pre-trained model')

    
    vgg = models.vgg19_bn(pretrained=True)
    
    new_state_dict = vgg.state_dict()
    dd = model.state_dict()
    for k in new_state_dict.keys():
        #print(k)
        if k in dd.keys() and k.startswith('features'):
            #print('yes')
            dd[k] = new_state_dict[k]
    model.load_state_dict(dd)
    

    '''
    load_checkpoint('yolov1/model_19_3_13.pth', model, optimizer)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    '''

    criterion = yoloLoss(7,2,5,0.5)
    
    if use_gpu:
        model.cuda()

    model.train()



    train_dataset = yolodataset(root=train_root, train=True, transform = transforms.ToTensor(), label='./src/yolo_train.txt')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    valid_dataset = yolodataset(root=train_root, train=False, transform = transforms.ToTensor(), label='./src/yolo_valid.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2)
    best_test_loss = 1.0

    print('start train')

    for epoch in range(num_epochs):
        epochTic = timeit.default_timer()
        model.train()
            
        if epoch > 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        
        total_loss = 0.
        
        for i,(images,target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            
            pred = model(images)
            yoloss = criterion(pred,target)
            loss = yoloss 
            
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\rTraining... Progress: %.1f %%'
                % (100*(i+1)/len(train_loader)),end='')
            
            
        print('\rEpoch [%d/%d], Training loss: %.4f'
            % (epoch + 1, num_epochs, total_loss/len(train_loader)),end='\n')
        
        validation_loss = 0.0
        
        model.eval()
        with torch.no_grad():
            for i,(images,target) in enumerate(valid_loader):
                images = Variable(images)
                target = Variable(target)
                if use_gpu:
                    images,target = images.cuda(),target.cuda()
                
                pred = model(images)
                yoloss = criterion(pred,target)
                loss = yoloss 
                validation_loss += loss.item()          
                print('\rValidation... Progress: %.1f %%'
                    % (100*(i+1)/len(valid_loader)),end='')
        
        validation_loss /= len(valid_loader)
        print('\rEpoch [%d/%d], valid loss: %.4f'
                % (epoch + 1, num_epochs, validation_loss),end='\n')
        
        
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            path ='models/model_yolo_'+str(epoch+1)+'.pth'
            save_checkpoint(path, model, optimizer)

        epochToc = timeit.default_timer()
        (t_min,t_sec) = divmod((epochToc-epochTic),60)
        print('Elapsed time is: %d min: %d sec' % (t_min,t_sec))

      
    

    print('finish')

if __name__=='__main__':
    main()