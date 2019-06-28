import pandas as pd
import numpy as np
import os
import torch
import csv

def yolo_format():
    with open('../train_labels.csv', 'r') as f:
        file = open('a.txt','w')
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        tmp = ''
        for row in csv_reader:
            if line_count > 0:
                if row[5] == '0':
                    tmp = row[0]
                    file.write('\n{}'.format(row[0]))
                else:
                    xmax, ymax = (float(row[1])+float(row[3])), (float(row[2])+float(row[4])) 
                    if row[0] == tmp:
                        tmp = row[0]
                        file.write(' {} {} {} {}'.format(row[1], row[2], str(xmax), str(ymax)))
                    else:
                        tmp = row[0]
                        file.write('\n{} {} {} {} {}'.format(row[0], row[1], row[2], str(xmax), str(ymax)))
            else:
                line_count+=1    



def yolo_split():
    f = open('a.txt', 'r')
    lines = f.readlines()
    print(len(lines))

    t = open('train.txt','w')
    v = open('valid.txt','w')
    for i in range(1, len(lines)-1):
        if i > 17500:
            v.write(lines[i])
        else:
            t.write(lines[i])

yolo_format()
yolo_split()

os.remove('a.txt')
