import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class yoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(yoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        #print(box1)
        #print(box2)
        """
        computing IoU
        :param box: (left, top, right, bottom)
        :return: scala value of IoU
        """
            # computing area of each rectangles
        S_rec1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        S_rec2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
    
        # find the each edge of intersect rectangle
        left_line = max(box1[0], box2[0])
        right_line = min(box1[2], box2[2])
        top_line = max(box1[1], box2[1])
        bottom_line = min(box1[3], box2[3])
    
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return intersect / (sum_area - intersect)

            
        
    def forward(self,pred_tensor,target_tensor):

        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0   #confidence
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,11)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]  #size 98*5
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,11)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,11)
        noo_target = target_tensor[noo_mask].view(-1,11)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c, reduction='sum')

        #compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        cell_size =448./7
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]*cell_size - 0.5*box1[:,2:4]*448    #xmin, ymin
            box1_xyxy[:,2:4] = box1[:,:2]*cell_size +0.5*box1[:,2:4]*448   #xmax, ymax
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]*cell_size - 0.5*box2[:,2:4]*448
            box2_xyxy[:,2:4] = box2[:,:2]*cell_size +0.5*box2[:,2:4]*448
            # compute iou
            iou1 = self.compute_iou(box1_xyxy[0,:4], box2_xyxy[:,:4].squeeze())
            iou2 = self.compute_iou(box1_xyxy[1,:4], box2_xyxy[:,:4].squeeze())
            if iou1 > iou2:
                max_iou = iou1
                max_index = 0
            else:
                max_iou = iou2
                max_index = 1

            #max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            #max_iou = np.asarray(max_iou)
            #max_iou = torch.FloatTensor(max_iou)
            #print(max_iou)
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = max_iou
        box_target_iou = Variable(box_target_iou).cuda()

        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4], reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2], reduction='sum') +\
                   F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]), reduction='sum')
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0

        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4], reduction='sum')

        # class loss

        class_loss = F.mse_loss(pred_tensor[:,:,:,10:], target_tensor[:,:,:,10:], reduction='sum')
        
        total_loss = self.l_coord*loc_loss + 1*contain_loss + 0.5*not_contain_loss + self.l_noobj*nooobj_loss + class_loss

        return total_loss/N

