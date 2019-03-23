import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_file')
parser.add_argument('train_label')
parser.add_argument('test_file')
parser.add_argument('output_file')

a = parser.parse_args()
# data processs
x_train = pd.read_csv(a.train_file)
y_train = pd.read_csv(a.train_label)

x_train = x_train.values
y_train = y_train.values

y_train = y_train.reshape(-1)  # check difference between (x,1) and (x,)

mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis = 0)
x_train = (x_train-mean)/(std+1e-100)

def sigmoid(z):
    return 1/(1+np.exp(-z))


# logistic regression
num = x_train.shape[1]
b = 0.0
w = np.ones(num)
lr = 0.025
epoch = 2000
b_lr = 0
w_lr = np.ones(num)

for e in range(epoch):
    z = np.dot(x_train, w) + b
    ans = sigmoid(z)
    loss = y_train - ans

    b_grad = -1*np.sum(loss)
    w_grad = -1*np.dot(loss, x_train)

    b_lr += b_grad**2
    w_lr += w_grad**2


    b = b-lr/np.sqrt(b_lr)*b_grad
    w = w-lr/np.sqrt(w_lr)*w_grad

    if(e+1)%500 == 0:
        loss = -1*np.mean(y_train*np.log(ans+1e-100) + (1-y_train)*np.log(1-ans+1e-100))
        ans[ans>=0.5] = 1
        ans[ans<0.5]  = 0
        acc = y_train - ans
        acc[acc==0] = 2
        acc[acc!=2] = 0
        acc[acc==2] = 1   
        print('epoch:{}\nloss:{}\naccuracy:{}%'.format(e+1,loss, np.sum(acc)*100/acc.shape[0]))


#test
x_test = pd.read_csv(a.test_file)
x_test = x_test.values

f = open(a.output_file,'w')
f.write('id,label\n')
for i in range(x_test.shape[0]):
    temp = x_test[i]
    temp = (temp-mean)/(std+1e-100)
    z = np.dot(w, temp) + b
    if z>0:
        f.write('{},{}\n'.format(i+1,1))
    else:
        f.write('{},{}\n'.format(i+1, 0))
f.close()
print('finish')

