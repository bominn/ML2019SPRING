import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import jieba
import re
import torch.nn.init as init
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_x')
parser.add_argument('train_y')
parser.add_argument('test_file')
parser.add_argument('jieba_file')
#parser.add_argument('output_file')

a = parser.parse_args()

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

jieba.load_userdict(a.jieba_file)
#load sentance
t = pd.read_csv(a.train_x)
t = t.iloc[:,1]
train_text = np.asarray(t)
#load label
l = pd.read_csv(a.train_y)
l = l.iloc[:,1]
label = np.asarray(l)

seq_train_text = []
for i in range(train_text.shape[0]):
    word_list = jieba.lcut(train_text[i])
    seq_train_text.append(word_list)
print('jieba finish')
#load word dictionary
f = open('word_dict.pkl','rb')
vocab = pickle.load(f)
dummy_index = len(vocab)

ff = open('word_matrix.pkl','rb')
wv_matrix = pickle.load(ff)

dummy_col = np.random.uniform(size = wv_matrix.shape[1]).reshape(1, -1)
dummy_index = wv_matrix.shape[0]
wv_matrix = np.vstack((wv_matrix, dummy_col,dummy_col,dummy_col))

vec = []
#dummy_index = unk, +1 EOF +2 pad
punctuation_search = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——\>\<！，。?？、\-～~@#￥%……&*（）：]+")
for text in seq_train_text:
    num = dummy_index  #num of word
    sentence = text
    clean_list = []
    for word in sentence:
        check = punctuation_search.match(word,0)
        if type(check)== type(None):
            try:
                clean_list.append(vocab[word])
            except:
                
                pad = dummy_index
                clean_list.append(pad)  #UNK
    if len(clean_list) > 128:
        clean_list = clean_list[:128-1]
        clean_list.append(dummy_index+1) #EOF
    while len(clean_list) < 128:
        clean_list.append(dummy_index+2) #pad
   
    clean_list = np.array(clean_list)
    vec.append(clean_list)

vec =  np.array(vec)
#deal with train data 
vec = vec[0:119017]
label = label[0:119017]

#split train data
train_vec, valid_vec = vec[0:100000], vec[100000:]
train_label, valid_label = label[0:100000], label[100000:]

#model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = 2

        self.embedding = nn.Embedding(input_size, 300)

        self.gru = nn.GRU(300, hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=True, batch_first=True, dropout=0.5)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 1*2, 128),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        embed_input = self.embedding(input.long())
        output, (hidden) = self.gru(embed_input)
        a,b,c,d = hidden[0], hidden[1], hidden[2], hidden[3]
        hidden = torch.cat((c,d),1)
    
        label = self.fc3(self.fc1(hidden))
        return label.squeeze()
    def init_embedding(self, matrix):
        self.embedding.weight = matrix
        print(self.embedding.weight.requires_grad)

class DcardDataset(Dataset):
    def __init__(self, data, train=True, label=None):
        self.data = torch.Tensor(data)
        if train:
            self.label = torch.from_numpy(label).float()
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.train:
            sample = {'data': self.data[idx],
                      'label': self.label[idx]}
        else:
            sample = {'data': self.data[idx]}

        return sample

input_size = dummy_index+3
batch_size = 32
n_epochs = 2
print_every = 1
hidden_size = 200
lr = 0.001

train_dataset = DcardDataset(data=train_vec,
                             label=train_label,
                             train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataset = DcardDataset(data=valid_vec,
                             label=valid_label,
                             train=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

#train
model = RNN(input_size, hidden_size, batch_size).cuda()
model.apply(weight_init)
model.init_embedding(torch.nn.Parameter(torch.Tensor(wv_matrix).cuda(),
                                        requires_grad = True) )

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
criterion = nn.BCELoss()

model.train()

ltrain_loss = []
lvalid_loss = []
ltrain_acc = []
lvalid_acc = []
print('start train')
for epoch in range(1, n_epochs + 1):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, sample in enumerate(train_loader):
        x = sample['data'].cuda()
        label = sample['label'].cuda()
        
        optimizer.zero_grad()
        #print(x[0])
        output_label = model(x)
        loss = criterion(output_label, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
        #_, preds = torch.max(output_label.data, 1)
        preds = (output_label>0.5).float()
        epoch_acc += torch.sum(preds == label)

    if epoch % print_every == 0:
        model.eval()
        with torch.no_grad():
            valid_acc = 0
            valid_loss = 0
            for i, sample in enumerate(valid_loader):
                x = sample['data'].cuda()
                label = sample['label'].cuda()
                optimizer.zero_grad()
                output_label = model(x)
                loss = criterion(output_label, label)
                #_, preds = torch.max(output_label.data, 1)
                valid_loss += criterion(output_label, label)
                preds = (output_label>0.5).float()
                valid_acc += torch.sum(preds == label)

        print('[ (%d %d%%), Loss:  %.3f, train_Acc: %.5f, valid_Loss: %.3f, valid_Acc: %.5f]' %
              (
               epoch,
               epoch / n_epochs * 100,
               epoch_loss/len(train_loader),
               float(epoch_acc) / len(train_loader) / batch_size,
               valid_loss/len(valid_loader),
               float(valid_acc) / len(valid_loader) / batch_size))
        ltrain_loss.append(epoch_loss/len(train_loader))
        lvalid_loss.append(valid_loss/len(valid_loader))
        ltrain_acc.append(float(epoch_acc) / len(train_loader) / batch_size)
        lvalid_acc.append(float(valid_acc) / len(valid_loader) / batch_size)
        epoch_loss = epoch_acc = 0
        if valid_loss/len(valid_loader) < 0.47:
            break

#load test data
test = pd.read_csv(a.test_file)
test = test.iloc[:,1]
test_text = np.asarray(test)

seq_test_text = []
for i in range(test_text.shape[0]):
    word_list = jieba.lcut(test_text[i])
    seq_test_text.append(word_list)

test_vec = []
punctuation_search = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——\>\<！，。?？、\-～~@#￥%……&*（）：]+")
for text in seq_test_text:
    #7127
    #13678
    num = dummy_index  #num of word
    sentence = text
    clean_list = []
    for word in sentence:
        check = punctuation_search.match(word,0)
        if type(check)== type(None):
            try:
                clean_list.append(vocab[word])
            except:
                
                pad = dummy_index
                clean_list.append(pad)
    
    if len(clean_list) > 128:
        clean_list = clean_list[:128-1]
        clean_list.append(dummy_index+1) #EOF
    while len(clean_list) < 128:
        clean_list.append(dummy_index+2) #pad
    clean_list = np.array(clean_list)
    test_vec.append(clean_list)

test_dataset = DcardDataset(data=test_vec,
                            train=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#start test
model.eval()
f = open('pred.csv', "w")
f.write('id,label\n')
ans = []
for i, sample in enumerate(test_loader):
    x = sample['data'].cuda()
    output_label = model(x)
    preds = (output_label>0.5).cpu().numpy()

    for i in range(len(preds)):
        ans.append(preds[i])
    
    
for j in range(len(ans)):
    f.write(str(j)+','+str(ans[j])+'\n')
f.close()
print('finish')