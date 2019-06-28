import pickle
import glob
import os
fname = sorted(glob.glob('test/'+'*.png'))

f = open('detection.pkl','rb')

z = pickle.load(f)
f.close()
for i in range(20):
    print(z[i][0])

#    print(len(z[i][0]))
f = open('pred.csv','w')
f.write('patientId,x,y,width,height,Target\n')

for i in range(len(fname)):
    if len(z[i][0]) == 0:
        f.write('{},,,,,{}\n'.format(os.path.basename(fname[i]), 0))
    else:
        for j in range(len(z[i][0])):
            box = z[i][0][j]
            xmin, ymin = box[0], box[1]
            w,h = min(1024,box[2])-box[0], min(1024,box[3])-box[1]
            xmin = xmin+w*0.075
            ymin = ymin+h*0.075
            w = w*0.85
            h = h*0.85
            f.write('{},{},{},{},{},{}\n'.format(os.path.basename(fname[i]),xmin,ymin,w,h, 1))
f.close()