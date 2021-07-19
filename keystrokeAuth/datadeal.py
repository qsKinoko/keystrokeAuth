import os
import numpy as np
import pickle

INPUTS_NUM = 5
STEPS_NUM = 30

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
				#int(float(keys[i][1])-float(keys[i-1][2]))/1000])
				int(float(keys[i][1])-float(keys[i-1][1]))/1000])
				#int(float(keys[i][2])-float(keys[i-1][2]))/1000])
	return np.array(resl)


def datadeal(dataname):
	
	num = 0
	arr = []
	
	rootdir = './data/usr_txt/'
	files = os.listdir(rootdir)
	
	for file in files:
		path = os.path.join(rootdir,file)
		if os.path.isfile(path):
			matrix = []
			f = open(path,'r')
			content = f.readlines()
			f.close()
			for line in content:
				a = line.split('\t')
				if(len(a) == 3):
					matrix.append(a)
		
			matrix = mtSort(matrix)
			vector = getVector(matrix)
	
			n = STEPS_NUM	
				
			while(n<len(vector)):
							
				arr = np.array(vector[n-STEPS_NUM:n],np.int32)
				arr.tofile(dataname+"_"+str(num)+".bin")
				n = n+STEPS_NUM
				num = num+1
			
		
			if n>=len(vector) and n-STEPS_NUM<len(vector):
				padding_array = np.zeros([STEPS_NUM, INPUTS_NUM],np.int32) 
				padding_array[0:len(vector[n-STEPS_NUM:])]=vector[n-STEPS_NUM:]
				padding_array.tofile(dataname+"_"+str(num)+".bin")
				num = num+1
