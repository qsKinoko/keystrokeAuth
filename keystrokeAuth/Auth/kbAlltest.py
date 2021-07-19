# -*- coding: utf-8 -*-  
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pickle
import kbtest

NUM = 4

def modelTest(mode,xs,thres):
	resl = kbtest.C_RNNtest(xs,mode) + thres
	return resl

def dataTest(xs):
	filename = "./data/thres.data"
	f = open(filename, 'rb')
	mode_thresh = pickle.load(f)
	f.close()

	resl = np.zeros(xs.shape[0])

	for mode in range(NUM):	
		resl = resl + modelTest(mode,xs,mode_thresh[mode])
	resl = resl/NUM
	print (resl[0])
	thre = 0.8
	if resl[0] < thre:
		return False
	elif resl[0] > thre:
		return True
