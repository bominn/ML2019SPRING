import time
import os
import copy
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
import timeit
import retinanet_model
from anchors import Anchors
from retinanet_dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
import pickle

def main():
    dataset_val = CSVDataset(train_file='test.csv', class_list='label.csv', transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    #dataset_train = CSVDataset(train_file='train.csv', class_list='label.csv', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    retinanet = retinanet_model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    path = './models/retinanet.pth'
    state = torch.load(path)
    retinanet.load_state_dict(state)

    retinanet = retinanet.cuda()
    retinanet.eval()
    all_detections = [[None for i in range(dataset_val.num_classes())] for j in range(len(dataset_val))]
    print(len(dataset_val))
    score_threshold=0.5
    max_detections=5
    with torch.no_grad():

        for index in range(len(dataset_val)):
            data = dataset_val[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
         
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset_val.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset_val.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset_val)), end='\r')
            #if index>20:
            #    break
    
    f = open('detection.pkl','wb')
    pickle.dump(all_detections,f)
    


if __name__=='__main__':
    main()
