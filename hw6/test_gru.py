import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import jieba
import re
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test_file')
parser.add_argument('jieba_file')
parser.add_argument('output_file')

a = parser.parse_args()

jieba.load_userdict(a.jieba_file)

f = open('word_dict.pkl','rb')

vocab = pickle.load(f)

dummy_index = len(vocab)

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

input_size = dummy_index+3
hidden_size = 200
batch_size = 128

model1 = RNN(input_size, hidden_size, batch_size).cuda()
state = torch.load('word_lstm_20.pth')
model1.load_state_dict(state)
model2 = RNN(input_size, hidden_size, batch_size).cuda()
state = torch.load('word_lstm_22.pth')
model2.load_state_dict(state)
model3 = RNN(input_size, hidden_size, batch_size).cuda()
state = torch.load('word_lstm_26.pth')
model3.load_state_dict(state)

test_dataset = DcardDataset(data=test_vec,
                            train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model1.eval()
model2.eval()
model3.eval()
f = open(a.output_file, "w")
f.write('id,label\n')
ans = []
for i, sample in enumerate(test_loader):
    x = sample['data'].cuda()
    output1 = model1(x)
    output2 = model2(x)
    output3 = model3(x)
    #_, preds = torch.max(output_label.data, 1)
    #preds = preds.to(torch.device('cpu'))
    preds = (output1+output2+output3)/3
    preds = (preds>0.495).cpu().numpy()
    #print(preds)
    for a in preds:
        ans.append(a)

for i in range(len(ans)):
    f.write(str(i)+','+str(ans[i])+'\n')
f.close()

print('test finish')