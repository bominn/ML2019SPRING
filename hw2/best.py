import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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

x_test = pd.read_csv(a.test_file)
x_test = x_test.values
x_test = (x_test-mean)/(std+1e-100)

'''
clf = GradientBoostingClassifier(loss = 'deviance',n_estimators = 150 , max_depth = 5, random_state = 0)
clf.fit(x_train, y_train)
scores = cross_val_score(clf, x_train, y_train, cv=3)
print(clf.score(x_train, y_train))
print(np.mean(scores))


f = open('model.pickle','wb')
pickle.dump(clf,f)
f.close()

k = clf.predict(x_test)
'''
ff = open('model.pickle','rb')
clf2 = pickle.load(ff)
ff.close()
k = clf2.predict(x_test)
f = open(a.output_file,'w')
f.write('id,label\n')
for i in range(len(k)):
    f.write('{},{}\n'.format(i+1,k[i]))
f.close()
print('finish')