import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train_file')
parser.add_argument('train_label')
parser.add_argument('test_file')
parser.add_argument('output_file')

a = parser.parse_args()
# calculate P(x|C)
def condition(x, u, conv):
    tmp = -0.5*np.dot(np.dot((x-u).T, np.linalg.inv(conv)), (x-u))
    cond = (1/(np.linalg.det(conv)**0.5))*np.exp(tmp)
    return cond
# data processs
x_train = pd.read_csv(a.train_file)
y_train = pd.read_csv(a.train_label)

x_train = x_train.values
y_train = y_train.values
y_train = y_train.reshape(-1)

mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis = 0)
x_train = (x_train-mean)/(std+1e-100)

# class A : > 50K
A_cls = (y_train == 1)
A_train = x_train[A_cls, :]

B_cls = (y_train == 0)
B_train = x_train[B_cls, :]

A_prior = A_train.shape[0]/x_train.shape[0]
B_prior = B_train.shape[0]/x_train.shape[0]


mean_A = np.mean(A_train, axis=0)
con_A = np.dot((A_train-mean_A).T, (A_train-mean_A))/A_train.shape[0]

mean_B = np.mean(B_train, axis=0)
con_B = np.dot((B_train-mean_B).T, (B_train-mean_B))/B_train.shape[0]

conv = A_prior*con_A + B_prior*con_B


accuracy = 0
for i in range(x_train.shape[0]):
    temp = x_train[i]

    A_cond = condition(temp, mean_A, conv)

    B_cond = condition(temp, mean_B, conv)
    
    prob = A_cond * A_prior / (A_cond * A_prior + B_cond * B_prior + 1e-100)
    
    if prob > 0.5:
        if y_train[i] == 1:
            accuracy += 1
    else:
        if y_train[i] == 0:
            accuracy +=1
print(accuracy/x_train.shape[0])


#test
x_test = pd.read_csv(a.test_file)
x_test = x_test.values

f = open(a.output_file,'w')
f.write('id,label\n')
for i in range(x_test.shape[0]):
    temp = x_test[i]
    temp = (temp-mean)/(std+1e-100)
    A_cond = condition(temp, mean_A, conv)
    B_cond = condition(temp, mean_B, conv)

    prob = A_cond * A_prior / (A_cond * A_prior + B_cond * B_prior + 1e-100)

    if prob > 0.5:
        f.write('{},{}\n'.format(i+1,1))
    else:
        f.write('{},{}\n'.format(i+1,0))
f.close()

print('finish')