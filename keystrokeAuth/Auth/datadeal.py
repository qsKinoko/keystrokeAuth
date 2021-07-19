import kbAlltest
import os
import numpy as np
import pickle
import warnings

INPUTS_NUM = 5#一次输入的长度
STEPS_NUM = 30#输入的次数
LENTH = 30

def mtSort(m):
	while True:
		tmp = []
		for i in range(len(m)-1):
			if float(m[i][1])>float(m[i+1][1]):
				tmp = m[i]
				m[i] = m[i+1]
				m[i+1] = tmp
		if tmp == []:
			break
	return m

def getVector(keys):
	resl = []
	for i in range(1,len(keys)):
		resl.append([
				int(keys[i-1][0]),int(keys[i][0]),
				int(float(keys[i-1][2])-float(keys[i-1][1]))/1000,
				int(float(keys[i][2])-float(keys[i][1]))/1000,
				int(float(keys[i][1])-float(keys[i-1][1]))/1000])
	return np.array(resl)


def datadeal(data):
	warnings.filterwarnings("ignore")
	data = mtSort(data)

	vector = getVector(data)
	filename = "./data/scaler.data"
	f = open(filename, 'rb')
	scaler = pickle.load(f)
	
	#vector = scaler.transform(vector) 
	if len(vector)!=STEPS_NUM:
		#print "wrong lenth!"
		arr = np.zeros([STEPS_NUM, INPUTS_NUM],np.int32) 
		arr[0:len(vector)] = vector
		padding_array = np.zeros([STEPS_NUM, INPUTS_NUM]) 
		padding_array[0:len(vector)]=vector
		padding_array = scaler.transform(padding_array) 
		#print padding_array
		resl = kbAlltest.dataTest(padding_array)
			
	else:
		arr = np.zeros([STEPS_NUM, INPUTS_NUM],np.int32) 
		arr[0:LENTH] = vector
		
		arr = scaler.transform(arr) 
		arr.tofile("try.bin")
		#print(arr)
		resl = kbAlltest.dataTest(arr)
		
	return resl


