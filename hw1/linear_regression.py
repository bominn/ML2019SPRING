import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')
#parser.add_argument('mode')
a = parser.parse_args()
# train
'''
data = pd.read_csv('train.csv', encoding= 'big5')

#data preprocess
data = data.values
feature = data[0:18,2]

data = data[:, 3:]
data[data == 'NR'] = 0.0
data = data.astype(float)


temp = [[] for i in range(18)]
for i in range(data.shape[0]):
    index = i%18
    for ele in data[i]:
        temp[index].append(ele)
temp = np.asarray(temp)

x_data = []
y_data = []
for i in range(temp.shape[1]-9):
    if i%480 >= 471:    #20days*24hour per month
        continue
    x = temp[:, i:i+9]  #choose x
    x = x.flatten()
    x_data.append(x)
    y_data.append(temp[9,i+9]) #Pm2.5 every 10hours
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

ma = np.max(x_data, axis=0)
mi = np.min(x_data, axis=0)
#x_data = (x_data-mi)/(ma-mi+1e-20)

#linear regression, use Adagrad
num = x_data.shape[1] 
b = 0.0
w = np.ones(num)
lr = 1.5
epoch = 80000

b_lr = 0
w_lr = np.zeros(num)

for e in range(epoch):
    error = y_data - b - np.dot(x_data, w)
    b_grad = -2*np.sum(error)  
    w_grad = -2*np.dot(error, x_data)

    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2
    mse = np.mean(np.square(error))

    b = b - lr/np.sqrt(b_lr)*b_grad
    w = w - lr/np.sqrt(w_lr)*w_grad

    if (e+1)%10000 == 0:
        print('epoch:{}\nloss{}'.format(e+1, np.sqrt(mse)))
        print(w_grad.shape)
np.save('model', w)
np.save('bias', b)
'''
# test 
test = pd.read_csv(a.input_file, encoding = 'big5', header = None)
test = test.values
test = test[:, 2:]
test[test=='NR'] = 0.0
test = test.astype(float)

length = test.shape[0]/18 

test_feature = []
for i in range(int(length)):
    x = test[i*18:i*18+18, :]
    x = x[:,:]
    x = x.flatten()
    test_feature.append(x)
test_feature = np.asarray(test_feature)
#test_feature = (test_feature-mi)/(ma-mi+1e-20)
f = open(a.output_file, 'w')
f.write('id,value\n')
w = np.load('model.npy')
b = np.load('bias.npy')
for i in range(len(test_feature)):
    ans = np.dot(w, test_feature[i]) + b
    f.write('id_{},{}\n'.format(i, ans))
f.close()

