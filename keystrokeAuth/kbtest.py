# -*- coding: utf-8 -*-  
#import matplotlib
#matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import gc

#import matplotlib.pyplot as plt 
#from sklearn import preprocessing
#import pickle

INPUTS_NUM = 5#一次输入的长度
STEPS_NUM = 30#输入的次数
BATCH_SIZE = 1
CLASS_NUM=1

def C_RNNtest(xs,snum): 
    Xs=xs
    resl = np.empty((10,Xs.shape[0]))
    th = np.multiply(np.ones([len(Xs),CLASS_NUM]),0.5)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model'+str(snum)+'/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./model/model'+str(snum)+'/'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input_x:0')
        outputs = graph.get_tensor_by_name('outputs:0')
        batch_size = graph.get_tensor_by_name('batch_size:0')
        update_num = graph.get_tensor_by_name('batch_size_update:0')
        #correct = graph.get_tensor_by_name('correct:0')

        Xs.shape = -1,STEPS_NUM,INPUTS_NUM# = xx.reshape([-1,STEPS_NUM,INPUTS_NUM]) 
        sess.run(batch_size,feed_dict={update_num:np.array(len(Xs)).reshape(1)})
        
        for i in range(10):	
            #print(i)		
            resl[i,:]=np.less(0.5,sess.run(outputs,feed_dict={x:Xs})).reshape(-1)

        return np.mean(resl.astype(np.int32), axis=0)

    