# -*- coding: utf-8 -*- 
import tensorflow as tf 
import numpy as np
import gc,os
import warnings

lr = 0.001

BATCH_SIZE = 128
EPOCH = 3000

INPUTS_NUM = 5#一次输入的长度
INPUTS_NUM2 = 32#CNN后的长度/CNN卷积核数量
STEPS_NUM = 30#输入的次数
CLASS_NUM = 1#结果类的数量
HIDDEN_LAYERS = [32,32]#隐藏层数量
KERNEL = 2


def train(xs, ys, snum, threshold):
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
    tf.reset_default_graph()
    graph = tf.Graph()
    tf.set_random_seed(1)
    with graph.as_default() as g:
        tf.set_random_seed(1)
        x = tf.placeholder(tf.float32,shape=[None,STEPS_NUM,INPUTS_NUM],name='input_x') #输入
        y = tf.placeholder(tf.float32,shape=[None,CLASS_NUM],name='input_y') #输出
        thres = tf.placeholder(tf.float32,shape=[None,CLASS_NUM],name='threshold') #阈值

        weights = {
            'in':tf.Variable(tf.random_normal([INPUTS_NUM2,HIDDEN_LAYERS[0]])),     
            'out':tf.Variable(tf.random_normal(([HIDDEN_LAYERS[0],CLASS_NUM]))) ,
            'w_conv': tf.Variable(tf.random_normal([KERNEL,INPUTS_NUM, 1, 32]))
        } 

        biases = { 
            'in':tf.Variable(tf.random_normal([HIDDEN_LAYERS[0],])),     
            'out':tf.Variable(tf.random_normal(([CLASS_NUM,]))) ,
            'b_conv':tf.Variable(tf.random_normal(([32]))) 
        }

        #计算结果
        bs = tf.get_variable("bs",initializer=BATCH_SIZE)
        bs_update = tf.placeholder(tf.int32,shape=[None],name='batch_size_update') #输入
        update=tf.assign(bs,bs_update[0],name="batch_size")

        pred = Rnn(x,weights,biases,bs.value()) 

        #赋予结果名称以便于寻找
        b = tf.constant(value=1,dtype=tf.float32)
        outputs = tf.multiply(pred,b,name='outputs')

        #计算损失
        ori_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs,labels=y)
        #tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs,labels=y)
        #-tf.reduce_mean(y * tf.log(tf.clip_by_value(outputs,1e-10,1.0)))
        #tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels= y)
        ####惩罚####
        #penalty = tf.constant(value=14,dtype=tf.float32)
        #ori_loss = tf.multiply(tf.multiply(tf.multiply(tf.cast(tf.less(thres,outputs),dtype=tf.float32),1-y),penalty)+b,ori_loss)
        ############
        cost = tf.reduce_mean(ori_loss + tf.contrib.layers.l2_regularizer(0.001)(tf.Variable(tf.random_normal([1,128]), dtype=tf.float32))) 

        #train_op = tf.train.AdagradOptimizer(0.1,name = "train_op").minimize(cost)#还可以，就是有的时候有点跳脱
        train_op = tf.train.AdamOptimizer(0.01,name = "train_op").minimize(cost)#还可以，就是有的时候有点跳脱

        
        correct = tf.less(thres,outputs)#tf.equal(tf.argmax(outputs,1),tf.argmax(y,1))
        cor = tf.multiply(tf.cast(correct,dtype=tf.float32),b,name='correct')
        frr = tf.reduce_sum(tf.multiply(tf.cast(correct,dtype=tf.float32),y)) / tf.reduce_sum(y)
        far = tf.reduce_sum(tf.multiply((tf.cast(correct,dtype=tf.float32)-1)*(-1),((y-1)*(-1)))) / (BATCH_SIZE-tf.reduce_sum(y))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(correct,dtype=tf.float32),y),tf.float32))


        init = tf.initialize_all_variables() 
        saver=tf.train.Saver()
	
        with tf.Session(graph=g) as sess:	
            os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
            warnings.filterwarnings("ignore")
            model_path='./model/model'+str(snum)+'/model.ckpt'
            tf.set_random_seed(1)
            sess.run(init)
		
            print("start...")
        
            xs.shape = -1,STEPS_NUM,INPUTS_NUM
            ys.shape = -1,1

            th = np.multiply(np.ones([BATCH_SIZE,CLASS_NUM]),threshold)
            test_xs = np.fromfile('test_data.bin',np.float)
            test_ys = np.fromfile('test_ans.bin',np.int32)

            test_xs.shape = -1,STEPS_NUM,INPUTS_NUM
            test_ys.shape = -1,2
            test_x = test_xs[0:BATCH_SIZE].reshape([BATCH_SIZE,STEPS_NUM,INPUTS_NUM])
            test_y = np.transpose(test_ys)[0][0:BATCH_SIZE].reshape([-1,1])

            train_acc = []
            test_acc = []	
            frrs = []
            fars = []
            eer = 0
            for n in range(EPOCH): 
                num = 0			
                while (num+1)*BATCH_SIZE <len(xs)+1:
                    batch_xs = xs[num*BATCH_SIZE:num*BATCH_SIZE+BATCH_SIZE]
                    batch_ys = ys[num*BATCH_SIZE:num*BATCH_SIZE+BATCH_SIZE].reshape([-1,1])
                    if len(batch_xs)>0:
                        batch_xs = batch_xs.reshape([BATCH_SIZE,STEPS_NUM,INPUTS_NUM])  
                    else:
                        continue
                    sess.run('train_op',feed_dict={x:batch_xs,y:batch_ys,thres:th})  
                    num = num+1
                    if n%10 == 0:
                        ac = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,thres:th})
                        train_acc.append(ac)
                        print("train:"+str(ac))
                        #print(sess.run(correct,feed_dict={x:test_x,y:test_y,thres:th}))
                        #print(test_y)
                        acc = sess.run(accuracy,feed_dict={x:test_x,y:test_y,thres:th})
                        fr = sess.run(frr,feed_dict={x:test_x,y:test_y,thres:th})
                        fa = sess.run(far,feed_dict={x:test_x,y:test_y,thres:th})
                        er = (fr+fa)/2
                        test_acc.append(acc)
                        fars.append(fa)
                        frrs.append(fr)
                        #print("test:"+str(acc))
                        #print("frr:"+str(fr))
                        #print("far:"+str(fa))
                        #print("eer:"+str(er))
                        if er > eer:
                            eer = er
                            #saver.save(sess,model_path)
                            #print("SAVE!")
                        #if ac > 0.999:
                        #    break
                #if ac > 0.999 and n>1000:
                #    break
            #saver=tf.train.Saver()
            saver.save(sess,model_path)
    gc.collect()
		

def Rnn(X,weights,biases,batch_size):
    warnings.filterwarnings("ignore")
    X = tf.reshape(X,[batch_size,-1,INPUTS_NUM,1])
   
	###加入卷积###
    conv1 = tf.nn.conv2d(X,  filter=weights['w_conv'], strides=[1, 1,1,1], padding='VALID')
    conv1 = tf.layers.batch_normalization(conv1)
    relu = tf.nn.relu(tf.nn.bias_add(conv1, biases['b_conv']))
    relu = tf.nn.dropout(relu, 0.5, noise_shape = None)
    pool1 = tf.nn.max_pool(relu, ksize=[1, 2, 1, 1], strides=[1, 1, 1, 1], padding='VALID') 
    ###卷积结束###
	
    X = tf.reshape(pool1,[-1,INPUTS_NUM2])

    X_in = tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.nn.dropout(X_in, 0.5, noise_shape = None) ###	
    
    X_in = tf.reshape(X_in,[-1,STEPS_NUM-KERNEL,HIDDEN_LAYERS[0]])#

    cells = [tf.nn.rnn_cell.GRUCell(hlayer) for hlayer in HIDDEN_LAYERS]
    #cells[0] = tf.nn.rnn_cell.DropoutWrapper(cells[0], output_keep_prob=0.5)
    cells[1] = tf.nn.rnn_cell.DropoutWrapper(cells[1], input_keep_prob=0.5)
    mul_cells=tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=True)
    _init_state = mul_cells.zero_state(batch_size,tf.float32)
    
    outputs,states = tf.nn.dynamic_rnn(mul_cells,X_in,initial_state=_init_state,time_major=False)
    
    ###########################
    result = tf.matmul(states[1],weights['out'])+biases['out']
    return result