import numpy as np
import csv
import glob
import os
#generate data format that support retinanet code, include train.csv, calid.csv, label.csv, test.csv
f = open('label.csv', 'w')

f.write('diease,0')
f.close()


f = open('train_labels.csv', 'r')
csv_reader = csv.reader(f, delimiter=',')

new = open('all.csv','w')
line_count = 0
for row in csv_reader:
    if line_count == 0:
        line_count+=1
        
    else:
        if row[5] == '0':
            new.write('{},,,,,\n'.format('train/'+row[0]))
        else:
            #print(row)
            xmax, ymax = (int(float(row[1]))+int(float(row[3]))), (int(float(row[2]))+int(float(row[4])))
            new.write('{},{},{},{},{},{}\n'.format('train/'+row[0], int(float(row[1])), int(float(row[2])), str(xmax), str(ymax), 'diease'))
f.close()
new.close()

f = open('all.csv','r')
csv_reader = csv.reader(f, delimiter=',')

train = open('train.csv','w', newline = '')
t = csv.writer(train, delimiter=',')

valid = open('valid.csv','w', newline = '')
v = csv.writer(valid, delimiter=',')

line_count = 0
for row in csv_reader:
    if line_count < 20000:
        t.writerow(row)
    else:
        v.writerow(row)
    line_count+=1
f.close()
train.close()
valid.close()
os.remove('all.csv')

fname = sorted(glob.glob('test/'+'*.png'))
#print(os.path.basename(fname[0]))

f = open('test.csv','w')
for i in range(len(fname)):
    f.write('{},,,,,\n'.format('test/'+os.path.basename(fname[i])))
f.close()