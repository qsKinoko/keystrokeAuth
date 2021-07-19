# -*- coding: utf-8 -*-  
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve
from sklearn import preprocessing
import pickle
import kbtest

INPUTS_NUM = 5#一次输入的长度
STEPS_NUM = 30#输入的次数
BATCH_SIZE =1
CLASS_NUM=2

DATA_NUM=512
START_FLAG=200000
NUM = 4

def modelTest(mode,xs,thres):
	resl = kbtest.C_RNNtest(xs,mode) + thres
	return resl

			
filename = "./data/thres.data"
f = open(filename, 'rb')
mode_thresh = pickle.load(f)
f.close()

xs = np.fromfile('test_data.bin',np.float)
ys = np.fromfile('test_ans.bin',np.int32)

xs.shape = -1,STEPS_NUM,INPUTS_NUM
ys.shape = -1,2
#print(xs)
all = len(ys)
half_all = all/2
far = 0 #错误接受率
frr = 0 #错误拒绝率
		
#scores = np.zeros(xs.shape[0],np.int32)
resl = np.zeros(xs.shape[0])

for mode in range(NUM):	
	resl = resl + modelTest(mode,xs,mode_thresh[mode])
resl = resl/NUM
'''
for i in range(100):
    for j in range(all):
        a = resl[j,i]
        if a>0.5:
            a = 1
            #scores[j] = scores[j]+1
        else:
            a = 0
        if a != ys[j][0]:
            if ys[j][0]==1:
                frr = frr+1
            else:
                far = far+1				
'''
for i,y in zip(resl,ys):
	#print(i)
	if i < 0.5 and y[0]==1:
		frr = frr + 1
	if i > 0.5 and y[0]==0:
		far = far + 1
print ("FAR:"+str(far)+"/"+str(half_all))
print ("FRR:"+str(frr)+"/"+str(half_all))
#y_lable = []
#for i in ys:
#    y_lable.append(i[0])

#eer = 1
min = 0.1
		
fpr, tpr, thresholds = roc_curve(np.array(np.transpose(ys)[0]),resl,pos_label=None,sample_weight=None,drop_intermediate=True)
	
for i in range(len(thresholds)):
        
    if (abs(fpr[i]-(1-tpr[i]))<min): #frr和far值相差很小时认为相等
        eer = abs(fpr[i]+1-tpr[i])/2
        min = abs(fpr[i]-(1-tpr[i]))
print("EER:"+str(eer))

plt.plot(fpr,1-tpr,c='#280f91') 
plt.plot([0,1],[0,1],c='y')
    
plt.scatter([eer],[eer],c='g')
plt.text(eer+0.03, eer+0.01, "EER="+str(eer), size = 10)
plt.ylabel("True Negative Rate")
plt.xlabel("False Positive Rate")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('./roc.png')


