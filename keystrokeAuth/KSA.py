# -*- coding: utf-8 -*- 

import os,time
import numpy as np
import random
from sklearn import preprocessing
import pickle
import datadeal
import kbtest
import kbtrain
import warnings

INPUTS_NUM = 5
STEPS_NUM = 30

def shuffle(X,Y):
	datax = X.tolist()
	datay = Y.tolist()
	data = []
	ans = []
	while(len(datax)>0):
		x = random.randint(0, len(datax)-1)
		data.append(datax[x])
		ans.append(datay[x])
		del datax[x]
		del datay[x]
	return np.array(data),np.array(ans)
	
def getScaler(X_posi,X_nega):
	for i in range(len(X_nega)):
		x = random.randint(0, len(posi)-1)
		X_nega.append(X_posi[x])
	
	
	arrData = np.array(X_nega).reshape(-1,INPUTS_NUM)
	scaler = preprocessing.StandardScaler().fit(arrData)
	return scaler
		

def BalanceCascade(X_train, y_train, X_test, num):	
	negnum = y_train[y_train == 0].shape[0]#负样本容量
	posnum = y_train[y_train == 1].shape[0]#正样本容量
	neg_index = np.argwhere(y_train == 0).reshape(negnum, )#argwhere返回索引，负样本索引
	pos_index = np.argwhere(y_train == 1).reshape(posnum, )#正样本索引
	pos_train = X_train[pos_index ,:]#训练集中的正样本
	FP = pow(posnum/negnum, 1/(num-1))#FP：False Positive假阳性，错误接受率，不过num应该是子集数量
	#classifiers = {}#分类器
	thresholds = {}#阈值
	test_prob = np.empty((X_test.shape[0], num))#用于记录预测结果
	for i in range(num):
		#classifiers[i] = AdaBoostClassifier()
		if len(neg_index)<posnum:			
			neg_index = np.repeat(neg_index,posnum,axis=0).reshape(-1)[0:posnum]
		neg_train_index = np.random.permutation(neg_index)[:posnum]
		neg_train = X_train[neg_train_index, :]#以上两步，对负样本进行洗牌，取出与正样本等量的负样本
		cur_X_train = np.r_[pos_train, neg_train]#按列连接两个矩阵，先正样本后负样本
		cur_y_train = np.r_[y_train[pos_index], y_train[neg_train_index]]#按列连接相应标签
		
		#classifiers[i].fit(cur_X_train, cur_y_train)#使用分类器训练模型
		cur_X_train,cur_y_train = shuffle(cur_X_train,cur_y_train)
		
		kbtrain.train(cur_X_train, cur_y_train, i,0.5)
		print("train_over!")
		#predict_result = classifiers[i].predict_proba(X_train[neg_index, :])[:,-1]#训练集中的所有负样本进行预测
		predict_result = kbtest.C_RNNtest(X_train[neg_index, :],i)
		print("test_over!")
		thresholds[i] = np.sort(predict_result)[int(neg_index.shape[0]*(1-FP))] - 0.5#建立一个阈值？（什么阈值？大概是用于分辨正负类的阈值
		print(thresholds[i])
		neg_index = np.argwhere(predict_result >= (thresholds[i] + 0.5)).reshape(-1, )
		#neg_index = np.intersect1d(np.argwhere(predict_result >= ( 0.5)).reshape(-1, ),neg_index)#选出分错的？
		print(len(neg_index))
		test_prob[:,i] = kbtest.C_RNNtest(X_test,i) + thresholds[i]#那个阈值大概类似于一个偏移#记录验证集验证结果
		#print(test_prob[:,i])
		print("No.{} Classifier Training Finished".format(i))
	test_prob_result = np.average(test_prob, axis=1)
	return test_prob_result,thresholds
	
	#return test_prob_result


random.seed(1)
warnings.filterwarnings("ignore")
print("数据收集环节...")
########用户名##########
print("用户名：")
username = input()
rootdir = "./data/usr/"

hook_exe = "hook_Service.exe"

########密码X3##########
print("输入密码_第一次：")
os.system(hook_exe)
print("输入密码_第二次：")
os.system(hook_exe)

########自由文本########
print("自由文本收集...")
ft = open("./data/cnews.txt",'r', encoding="utf-8")
content = ft.readlines()
ft.close()
freetxt = content[random.randint(0, len(content)-1)]
del content
print(freetxt.split('\t')[1][0:210])
os.system(hook_exe)

########制作数据集######
print("数据集制作环节...")
datadeal.datadeal(rootdir+username)

files = os.listdir(rootdir)
alld = np.zeros(0)
alld.shape = -1,INPUTS_NUM
userd = np.zeros(0)
negaset = np.zeros(0)
negaset.shape = -1,INPUTS_NUM

for file in files:
	arr = np.fromfile(rootdir+file,np.int32)
	userd = np.concatenate((userd,arr))

negaset = np.fromfile("./data/counter/negaSet.bin",np.int32)
userd.shape = -1,STEPS_NUM,INPUTS_NUM
negaset.shape = -1,STEPS_NUM,INPUTS_NUM
posi = userd.tolist()
nega = negaset.tolist()


data = []
ans = []
num = 0

for i in range(64):
	x = random.randint(0, len(posi)-1)
	data.append(posi[x])
	ans.append(1)
	num = num+1

for i in nega:
	data.append(i)
	ans.append(0)

arrData = np.array(data)
arrAns =  np.array(ans)

arrData.shape = -1,INPUTS_NUM
scaler = getScaler(posi,nega)
#print(scaler.mean_)
#print(scaler.var_)
arrData = scaler.transform(arrData)

f = open("./data/scaler.data", 'wb')
pickle.dump(scaler, f)
f.close()

arrData.tofile('data.bin')
arrAns.tofile('ans.bin')

#print(len(arrAns))
#print(num)
print("数据集制作完成！")

##########模型训练############
print("模型训练环节...")
arrData.shape = -1,STEPS_NUM,INPUTS_NUM
arrAns.shape = -1
xs = np.fromfile('test_data.bin',np.float)
ys = np.fromfile('test_ans.bin',np.int32)
xs.shape = -1,STEPS_NUM,INPUTS_NUM
ys.shape = -1,2
test_resl,thres = BalanceCascade(arrData, arrAns, xs, 4)

print (thres)

f = open("./data/thres.data", 'wb')
pickle.dump(thres, f)
f.close()

frr = 0
far = 0
thres = 0.5
#print(thres)
for i,j in zip(ys,test_resl):
	if i[0]>0.5 and j>thres:
		frr = frr + 1
	elif i[0]<0.5 and j<thres:
		far = far + 1
	#else:
		#print(i,j)
#print(frr/132)
#print(far/132)
##########模型测试############

print("---END---")
