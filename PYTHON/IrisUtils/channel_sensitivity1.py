# -*- coding: utf-8 -*-
"""
This file is used to train and test the DetNet architecture in the hard decision output scenario.
The constellation used is QPSK and the channel is complex
all parameters were optimized and trained over the 20X30 iid channel, changing the channel might require parameter tuning
Notice that the run time analysis presented in the paper was made on a numpy version of the tensorflow network.
writen by Neev Samuel based on the paper:
    "Learning to detect, Neev Samuel,Tzvi Diskin,Ami Wiesel"
contact by neev.samuel@gmail.com
"""
import tensorflow.compat.v1 as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl
import matplotlib.pyplot as plt

"""
Parameters
K - size of x
N - size of y
snrdb_low - the lower bound of noise db used during training
snr_high - the higher bound of noise db used during training
L - number of layers in DetNet
v_size = size of auxiliary variable at each layer
hl_size - size of hidden layer at each DetNet layer (the dimention the layers input are increased to
startingLearningRate - the initial step size of the gradient descent algorithm
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor
train_iter - number of train iterations
train_batch_size - batch size during training phase
test_iter - number of test iterations
test_batch_size  - batch size during testing phase
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the same weight 
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article)
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread between snrdb_low_test and snrdb_high_test 
"""

###start here
tf.compat.v1.disable_eager_execution()
sess = tf.InteractiveSession()

if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")
   

K = 20
N = 30
snrdb_low = 7.0
snrdb_high = 14.0
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
#n0=np.expand_dims(0.5,1)
res_alpha=0.9
L=30
v_size = 1*(2*K)
hl_size = 4*(2*K)
test_iter= 200
test_batch_size=1000

train_iter = 200000
train_batch_iter = 3000  # train batch size

num_snr = 8
snrdb_low_test=8.0
snrdb_high_test=15.0
snrdb_list = np.linspace(snrdb_low_test,snrdb_high_test,num_snr)
print(snrdb_list)
snr_list = 10.0 ** (snrdb_list/10.0)
startingLearningRate = 0.0003
decay_factor = 0.97
decay_step = 1000

print('DetNet QPSK')
print(K)
print(N)
print(snrdb_low)
print(snrdb_high)
print(snr_low)
print(snr_high)
print(L)
print(v_size)
print(hl_size)
print(startingLearningRate)
print(decay_factor)
print(decay_step)
print(train_iter)
print(train_batch_iter)
print(test_iter)
print(test_batch_size)
print(res_alpha)
print(num_snr)
print(snrdb_low_test)
print(snrdb_high_test)



"""Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""

def generate_data_iid_test(B,K,N,snr_low,snr_high):
    H_R = np.random.randn(B,N,K)
    H_I = np.random.randn(B,N,K)
    
    channel_noise = np.random.randn(B,N,K)
    H_R_noisy = np.zeros(H_R.shape)
    H_I_noisy = np.zeros(H_I.shape)
    
    H_  = np.zeros([B,2*N,2*K])
    HN_  = np.zeros([B,2*N,2*K])

    x_R = np.sign(np.random.rand(B,K) - 0.5)
    x_I = np.sign(np.random.rand(B,K) - 0.5)
    x_  = np.concatenate((x_R , x_I) , axis = 1)

    y_  = np.zeros([B,2*N])

    w_R = np.random.randn(B,N)
    w_I = np.random.randn(B,N)
    w   = np.concatenate((w_R , w_I) , axis = 1)

    Hy_ = np.zeros([B,2*K])
    HyN_ = np.zeros([B,2*K])

    HH_ = np.zeros([B,2*K,2*K])
    HHN_ = np.zeros([B,2*K,2*K])
    
    SNR_ = np.zeros([B])
    x_ind = np.zeros([B,K,4])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,3] =  1
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H = np.concatenate((np.concatenate((H_R[i,:,:], -1*H_I[i,:,:]), axis=1) , np.concatenate((H_I[i,:,:] , H_R[i,:,:]), axis=1) ), axis=0)
        tmp_snr=(H.T.dot(H)).trace()/(2*K)
        H_R_noisy[i,:,:] = H_R[i,:,:] + channel_noise[i,:,:]/np.sqrt(2*SNR)
        H_I_noisy[i,:,:] = H_I[i,:,:] + channel_noise[i,:,:]/np.sqrt(2*SNR)
        H_noisy = np.concatenate((np.concatenate((H_R_noisy[i,:,:], -1*H_I_noisy[i,:,:]), axis=1) , np.concatenate((H_I_noisy[i,:,:] , H_R_noisy[i,:,:]), axis=1) ), axis=0)
        H_[i,:,:]= H
        HN_[i,:,:] = H_noisy
        y_[i,:]=(H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HyN_[i,:] = H_noisy.T.dot(y_[i,:])
        HH_[i,:,:]= H.T.dot( H_[i,:,:])
        HHN_[i,:,:] = H_noisy.T.dot( HN_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,HN_,Hy_,HyN_,HH_,HHN_,x_,SNR_, H_R, H_R_noisy, H_I, H_I_noisy, x_R, x_I, w_R, w_I, x_ind



def generate_data_train(B,K,N,snr_low,snr_high):
    H_R = np.random.randn(B,N,K)
    H_I = np.random.randn(B,N,K)
    H_  = np.zeros([B,2*N,2*K])
    

    x_R = np.sign(np.random.rand(B,K) - 0.5)
    x_I = np.sign(np.random.rand(B,K) - 0.5)
    x_  = np.concatenate((x_R , x_I) , axis = 1)

    y_  = np.zeros([B,2*N])

    w_R = np.random.randn(B,N)
    w_I = np.random.randn(B,N)
    w   = np.concatenate((w_R , w_I) , axis = 1)

    Hy_ = np.zeros([B,2*K])

    HH_ = np.zeros([B,2*K,2*K])
    SNR_ = np.zeros([B])
    x_ind = np.zeros([B,K,4])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,3] =  1
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H   = np.concatenate((np.concatenate((H_R[i,:,:], -1*H_I[i,:,:]), axis=1) , np.concatenate((H_I[i,:,:] , H_R[i,:,:]), axis=1) ), axis=0)
        tmp_snr=(H.T.dot(H)).trace()/(2*K)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I,x_ind

def generate_data_iid_validate(B,K,N,snr_low,snr_high):
    H_R = np.random.randn(B,N,K)
    H_I = np.random.randn(B,N,K)
    H_  = np.zeros([B,2*N,2*K])
    

    x_R = np.sign(np.random.rand(B,K) - 0.5)
    x_I = np.sign(np.random.rand(B,K) - 0.5)
    x_  = np.concatenate((x_R , x_I) , axis = 1)

    y_  = np.zeros([B,2*N])

    w_R = np.random.randn(B,N)
    w_I = np.random.randn(B,N)
    w   = np.concatenate((w_R , w_I) , axis = 1)

    Hy_ = np.zeros([B,2*K])

    HH_ = np.zeros([B,2*K,2*K])
    SNR_ = np.zeros([B])
    x_ind = np.zeros([B,K,4])
    for i in range(B):
        for ii in range(K):
            if x_R[i,ii]==-1 and x_I[i,ii] == -1:
                x_ind[i,ii,0] =  1
            if x_R[i,ii]==-1 and x_I[i,ii] == 1:
                x_ind[i,ii,1] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == -1:
                x_ind[i,ii,2] =  1
            if x_R[i,ii]==1 and x_I[i,ii] == 1:
                x_ind[i,ii,3] =  1
    for i in range(B):
        SNR = np.random.uniform(low=snr_low,high=snr_high)
        H   = np.concatenate((np.concatenate((H_R[i,:,:], -1*H_I[i,:,:]), axis=1) , np.concatenate((H_I[i,:,:] , H_R[i,:,:]), axis=1) ), axis=0)
        tmp_snr=(H.T.dot(H)).trace()/(2*K)
        H_[i,:,:]=H
        y_[i,:]=(H.dot(x_[i,:])+w[i,:]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        Hy_[i,:]=H.T.dot(y_[i,:])
        HH_[i,:,:]=H.T.dot( H_[i,:,:])
        SNR_[i] = SNR
    return y_,H_,Hy_,HH_,x_,SNR_, H_R, H_I, x_R, x_I, w_R, w_I,x_ind

def piecewise_linear_soft_sign(x):
    t = tf.Variable(0.1)
    y = -1+tf.nn.relu(x+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x-t)/(tf.abs(t)+0.00001)
    return y

def affine_layer(x,input_size,output_size,Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))

    y = tf.matmul(x, W)+w
    return y

def relu_layer(x,input_size,output_size,Layer_num):
    y = tf.nn.relu(affine_layer(x,input_size,output_size,Layer_num))
    return y

def sign_layer(x,input_size,output_size,Layer_num):
    y = affine_layer(x,input_size,output_size,Layer_num)
    return y


HY = tf.placeholder(tf.float32,shape=[None,2*K])
X = tf.placeholder(tf.float32,shape=[None,2*K])
HH = tf.placeholder(tf.float32,shape=[None, 2*K , 2*K])
X_IND = tf.placeholder(tf.float32,shape=[None, K , 4])
batch_size = tf.shape(HY)[0]


x_real = X[:,0:K]
x_imag = X[:,K:2*K]


S1=[]
S1.append(tf.zeros([batch_size,2*K]))
S2=[]
S2.append(tf.zeros([batch_size,4*K]))
V=[]
V.append(tf.zeros([batch_size,v_size]))
LOSS=[]
LOSS.append(tf.zeros([]))
BER=[]
BER.append(tf.zeros([]))
delta = tf.Variable(tf.zeros(L*2,1))

for i in range(1,L):
    temp1 = tf.matmul(tf.expand_dims(S1[-1],1),HH)
    temp1= tf.squeeze(temp1,1)
    Z1 = S1[-1] - delta[(i-1) * 2]*HY + delta[(i-1) * 2 + 1]*temp1
    Z = tf.concat([Z1, V[-1]], 1)
    ZZ = relu_layer(Z,(2*K) + v_size , hl_size,'relu'+str(i))
    S2.append(sign_layer(ZZ , hl_size , 4*K,'sign'+str(i)))
    S2[i]=(1-res_alpha)*S2[i]+res_alpha*S2[i-1]
    S2[i] = tf.clip_by_value(S2[i],0,1)
    V.append(affine_layer(ZZ , hl_size , v_size,'aff'+str(i)))
    V[i]=(1-res_alpha)*V[i]+res_alpha*V[i-1]
    
    S3 = tf.reshape(S2[i],[batch_size,K,4])
    temp_0 = S3[:,:,0]
    temp_1 = S3[:,:,1]
    temp_2 = S3[:,:,2]
    temp_3 = S3[:,:,3]
    
    S1_real = -1.0*temp_0  +\
              -1.0*temp_1  +\
              1.0*temp_2  +\
              1.0*temp_3

    S1_im =   -1.0*temp_0  +\
              1.0*temp_1  +\
              -1.0*temp_2  +\
               1.0*temp_3
    S1.append(tf.concat([S1_real, S1_im], 1))
    x_ind_reshaped = tf.reshape(X_IND,[batch_size,4*K])
    LOSS.append(np.log(i)*tf.reduce_mean(tf.reduce_mean(tf.square(x_ind_reshaped - S2[-1]),1)))
    BER.append(tf.reduce_mean(tf.cast(tf.logical_or(tf.not_equal(x_real,tf.sign(S1[-1][:,0:K])),tf.not_equal(x_imag,tf.sign(S1[-1][:,K:2*K]))), tf.float32)))
    
Max_Val = tf.reduce_max(S3,axis=2, keep_dims=True)
Greater = tf.greater_equal(S3,Max_Val)
BER2 = tf.round(tf.cast(Greater,tf.float32))
BER3 = tf.not_equal(BER2, X_IND)
BER4 = tf.reduce_sum(tf.cast(BER3,tf.float32),2)
BER5 = tf.cast(tf.greater(BER4,0),tf.float32)
SER =  tf.reduce_mean(BER5)    
TOTAL_LOSS=tf.add_n(LOSS)

saver = tf.train.Saver()

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(startingLearningRate, global_step, decay_step, decay_factor, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(TOTAL_LOSS)
init_op=tf.global_variables_initializer()

train_flag = False
if train_flag:
    sess.run(init_op)
    train_loss = []
    val_loss = []
    ser = []
    for i in range(train_iter): #num of train iter

        batch_Y, batch_H, batch_HY, batch_HH, batch_X , SNR1 , H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_train(train_batch_iter,K,N,snr_low,snr_high)
        train_step.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})

        if i % 100== 0 :
            print(TOTAL_LOSS.eval(feed_dict={
                HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}
            ))
            train_loss.append(TOTAL_LOSS.eval(feed_dict={
                HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}))
            batch_Y, batch_H, batch_HY, batch_HH, batch_X ,SNR1, H_R, H_I, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_validate(train_batch_iter,K,N,snr_low,snr_high)
            results = sess.run([TOTAL_LOSS,BER[L-1],S1[L-1]], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind})
            print_string = [i]+results[:2]
            print(' '.join('%s' % x for x in print_string))
            sys.stderr.write(str(i)+' ')
            val_loss.append(results[0])
            ser.append(results[1])

saver.restore(sess, "./DetNet_HD_QPSK/QPSK_HD_model.ckpt")
plot_flag = False
if plot_flag == True:
    bers = np.zeros((1,num_snr))
    bers_noisy = np.zeros((1,num_snr))
    tmp_bers = np.zeros((1,test_iter))
    tmp_bers_noisy = np.zeros((1,test_iter))
    tmp_ber_iter=np.zeros([L,test_iter])
    ber_iter=np.zeros([L,num_snr])
    bers_zf = np.zeros((1,num_snr))
    bers_zf_noisy = np.zeros((1,num_snr))
    tmp_bers_zf = np.zeros((1,test_iter))
    tmp_bers_zf_noisy = np.zeros((1,test_iter))
    
    
    for j in range(num_snr):
        for jj in range(test_iter):
            print(j)
            print(jj)
            batch_Y, batch_H, batch_HN, batch_HY, batch_HYN, batch_HH, batch_HHN, batch_X ,SNR1, H_R, H_R_noisy, H_I, H_I_noisy, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(test_batch_size,K,N,snr_list[j],snr_list[j])
            results = np.array(sess.run([BER[L-1],S1[L-1]], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}))#/(test_batch_size*K)
            tmp_bers[0,jj] = results[0]
            tmp_bers_noisy[0,jj] = np.array(sess.run(BER[L-1], {HY: batch_HYN, HH: batch_HHN, X: batch_X,X_IND:x_ind}))#/(test_batch_size*K)
            
            # Zero Forcing
            x_zf = np.zeros((test_batch_size,K),dtype = complex)
            x_zf_noisy = np.zeros((test_batch_size,K),dtype = complex)
            y = batch_Y[:,:N] + 1j * batch_Y[:,N:]
            H = H_R + 1j * H_I
            H_noisy = H_R_noisy + 1j * H_I_noisy
            for batch in range(test_batch_size):
                x_zf[batch,:] = np.dot(np.linalg.pinv(H[batch,:,:]),y[batch,:])
                x_zf_noisy[batch,:] = np.dot(np.linalg.pinv(H_noisy[batch,:,:]),y[batch,:])
            x_zf_R, x_zf_I = x_zf.real, x_zf.imag
            x_zf_R_noisy, x_zf_I_noisy = x_zf_noisy.real, x_zf_noisy.imag
            tmp_bers_zf[0,jj] = np.mean(np.logical_or(np.not_equal(x_R,np.sign(x_zf_R)),np.not_equal(x_I,np.sign(x_zf_I))).astype(int))
            tmp_bers_zf_noisy[0,jj] = np.mean(np.logical_or(np.not_equal(x_R,np.sign(x_zf_R_noisy)),np.not_equal(x_I,np.sign(x_zf_I_noisy))).astype(int))
        if j == 0 or j == 3 or j == 5:
            plt.scatter(x_R[:,0],x_I[:,0],color = 'red')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.title('True Symbols')
            plt.savefig('True' + str(snrdb_list[j])+'.jpg')
            plt.close()
            plt.scatter(results[1][:,0],results[1][:,K],color = 'blue')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.title('Dnet predictions')
            plt.savefig('Dnet' + str(snrdb_list[j])+'.jpg')
            plt.close()
            plt.scatter(x_zf_R[:,0],x_zf_I[:,0],color = 'yellow')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.title('ZF predictions')
            plt.savefig('ZF' + str(snrdb_list[j])+'.jpg')
            plt.close()
            
        bers[0][j] = np.mean(tmp_bers[0])
        bers_noisy[0][j] = np.mean(tmp_bers_noisy[0])
        #ber_iter[:,j]=np.mean(tmp_ber_iter,1)
        bers_zf[0][j] = np.mean(tmp_bers_zf[0])
        bers_zf_noisy[0][j] = np.mean(tmp_bers_zf_noisy[0])
        
    
    print('snrdb_list')
    print(snrdb_list)
    print('bers detnet')
    print(bers)
    print('bers detnet noisy')
    print(bers_noisy)
    print('bers zero forcing')
    print(bers_zf)
    print('bers zero forcing with noisy channel')
    print(bers_zf_noisy)
    #save_path = saver.save(sess, "./DetNet_HD_QPSK/QPSK_HD_model.ckpt")
    
    #d = N/K
    #snrd = [snr**(-d) for snr in snr_list]
    plt.semilogy(snrdb_list,bers[0],label = 'Dnet')
    plt.semilogy(snrdb_list,bers_noisy[0],label = 'Dnet with noisy channel')
    plt.semilogy(snrdb_list,bers_zf[0],label = 'ZF')
    plt.semilogy(snrdb_list,bers_zf_noisy[0],label = 'ZF with noisy channel')
    #plt.semilogy(snrdb_list,snrd,label = 'SNR^-d')
    plt.xlabel('SNR dB')
    plt.ylabel('SER')
    plt.title('SER Vs SNR')
    plt.legend()
    plt.savefig('channel_sensitivity_official_origfullrange.png')
    plt.close()

    # =============================================================================
    # plt.semilogy(snrdb_list,bers[0],label = 'Dnet')
    # plt.semilogy(snrdb_list,bers_zf[0],label = 'ZF')
    # plt.semilogy(snrdb_list,snrd,label = 'SNR^-d')
    # plt.xlabel('SNR dB')
    # plt.ylabel('SER')
    # plt.title('SER Vs SNR')
    # #plt.yscale("log")
    # plt.legend()
    # plt.savefig('dnetvszf_official_origk.png')
    # plt.close()
    # =============================================================================

    plt.semilogy(snrdb_list,bers_noisy[0],label = 'Dnet with noisy channel')
    plt.semilogy(snrdb_list,bers_zf_noisy[0],label = 'ZF with noisy channel')
    #plt.semilogy(snrdb_list,snrd,label = 'SNR^-d')
    plt.xlabel('SNR dB')
    plt.ylabel('SER')
    plt.title('SER Vs SNR')
    #plt.yscale("log")
    plt.legend()
    plt.savefig('noisydnetvszf_official_origfullrange.png')
    plt.close()


num_snr = 1

snrdb_list = [30.0]
snr_list = [10.0 ** (snrdb_list[0]/10.0)]

# =============================================================================
# bers = np.zeros((1,num_snr))
# bers_noisy = np.zeros((1,num_snr))
# =============================================================================
tmp_bers = []
tmp_bers_noisy = []

# =============================================================================
# bers_zf = np.zeros((1,num_snr))
# bers_zf_noisy = np.zeros((1,num_snr))
# =============================================================================
tmp_bers_zf = []
tmp_bers_zf_noisy = []

errors = 0


while errors < 10:
    batch_Y, batch_H, batch_HN, batch_HY, batch_HYN, batch_HH, batch_HHN, batch_X ,SNR1, H_R, H_R_noisy, H_I, H_I_noisy, x_R, x_I, w_R, w_I,x_ind= generate_data_iid_test(test_batch_size,K,N,snr_list[0],snr_list[0])
    results = np.array(sess.run([BER[L-1],S1[L-1]], {HY: batch_HY, HH: batch_HH, X: batch_X,X_IND:x_ind}))#/(test_batch_size*K)
    tmp_bers.append(results[0])
    #print(tmp_bers)
    results_noisy = np.array(sess.run([BER[L-1]], {HY: batch_HYN, HH: batch_HHN, X: batch_X,X_IND:x_ind}))#/(test_batch_size*K)
    tmp_bers_noisy.append(results_noisy[0])
    #print(tmp_bers_noisy)
    errors += results[0]*K*test_batch_size   
    #Zero Forcing
    x_zf = np.zeros((test_batch_size,K),dtype = complex)
    x_zf_noisy = np.zeros((test_batch_size,K),dtype = complex)
    y = batch_Y[:,:N] + 1j * batch_Y[:,N:]
    H = H_R + 1j * H_I
    H_noisy = H_R_noisy + 1j * H_I_noisy
    for batch in range(test_batch_size):
        x_zf[batch,:] = np.dot(np.linalg.pinv(H[batch,:,:]),y[batch,:])
        x_zf_noisy[batch,:] = np.dot(np.linalg.pinv(H_noisy[batch,:,:]),y[batch,:])
    x_zf_R, x_zf_I = x_zf.real, x_zf.imag
    x_zf_R_noisy, x_zf_I_noisy = x_zf_noisy.real, x_zf_noisy.imag
    tmp_bers_zf.append(np.mean(np.logical_or(np.not_equal(x_R,np.sign(x_zf_R)),np.not_equal(x_I,np.sign(x_zf_I))).astype(int)))
    #print(tmp_bers_zf)
    tmp_bers_zf_noisy.append(np.mean(np.logical_or(np.not_equal(x_R,np.sign(x_zf_R_noisy)),np.not_equal(x_I,np.sign(x_zf_I_noisy))).astype(int)))
    #print(tmp_bers_zf_noisy)
bers = sum(tmp_bers)/len(tmp_bers)
bers_noisy = sum(tmp_bers_noisy)/len(tmp_bers_noisy)
bers_zf = sum(tmp_bers_zf)/len(tmp_bers_zf)
bers_zf_noisy = sum(tmp_bers_zf_noisy)/len(tmp_bers_zf_noisy)


print(bers)
print(bers_noisy)
print(bers_zf)
print(bers_zf_noisy)

