import os
import numpy as np 
from skimage.io import imread, imsave
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('img_name')
parser.add_argument('recon_name')
a = parser.parse_args()

path = a.img_path
k = 5

filelist =  sorted(glob.glob(os.path.join(path,'*jpg')))
img_shape = imread(filelist[0]).shape 
print('read image finish')
img_data = []
for filename in filelist:
    tmp = imread(filename)  
    img_data.append(tmp.flatten())

training_data = np.array(img_data).astype('float32')
mean = np.mean(training_data, axis = 0) 

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

#problem 1-a
#imsave('average.jpg.jpg',process(mean).reshape(img_shape))

#problem 1-b
center = training_data - mean
U, S, V = np.linalg.svd(center.T, full_matrices=False)
print('SVD finish')
#for i in range(5):
#    eigenface = process(U[:, i].reshape(img_shape))
#    imsave('eigen_face'+str(i+1)+'.jpg',eigenface)

#reconstruct image  
picked_img = imread(os.path.join(path,a.img_name))
X = picked_img.flatten().astype('float32')
X -= mean
weight = np.dot(X, U[:, :5])
reconstruct = process(mean + np.dot(weight, U[:, :5].T))
imsave(a.recon_name, reconstruct.reshape(img_shape))

print('reconstruct finish')
#problem 1-d
#for i in range(5):
#    number = S[i] * 100 / sum(S)
#    print(number)
