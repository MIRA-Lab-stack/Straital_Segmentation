from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import os
from scipy.io import loadmat
from scipy.ndimage import rotate
from random import randint
import time
import pickle
import random
import nibabel as nib

tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"]="1"

PATCH_SIZE = 72
NUM_PAT    = 5
NUM_PATCH  = 6 #each batch will have NUM_PATCH patches from each of NUM_PAT
LEARNING_RATE = 1e-5
N_CLASS = 6
NUM_ITER = 100000

def conv_block(x,KERNEL_SIZE, FILTERS):
    x = tf.layers.conv3d(x,FILTERS,KERNEL_SIZE,padding='same')
    return tf.nn.elu(x)

def unet(x,N_CLASS):
    with tf.name_scope('layer1_enc'):
        x1 = conv_block(x,3,16)
        x1 = conv_block(x1,3,16)
        x1 = conv_block(x1,3,16)
        x1p = tf.layers.max_pooling3d(x1, 2, 2)
        print(x1p.shape)
    with tf.name_scope('layer2_enc'):
        x2 = conv_block(x1p,3,32)
        x2 = conv_block(x2,3,32)
        x2 = conv_block(x2,3,32)
        x2p = tf.layers.max_pooling3d(x2, 2, 2)
        print(x2p.shape)
    with tf.name_scope('layer3_enc'):
        x3 = conv_block(x2p,3,64)
        x3 = conv_block(x3,3,64)
        x3 = conv_block(x3,3,64)
        x3p = tf.layers.max_pooling3d(x3, 2, 2)
        print(x3p.shape)
    with tf.name_scope('bottom'):
        x4 = conv_block(x3p,3,128)
        x4 = conv_block(x4,3,128)
        x4 = conv_block(x4,3,128)
        
        x4t = tf.layers.conv3d_transpose(x4,64,3,strides=2,padding='same')
        print('p0')
        print(x4t.shape)
    with tf.name_scope('layer3_dec'):
        x3 = tf.concat((x4t,x3),axis=-1)
        x3 = conv_block(x3,3,64)
        x3 = conv_block(x3,3,64)
        x3 = conv_block(x3,3,64)
        x3t = tf.layers.conv3d_transpose(x3,32,3,strides=2,padding='same')  
        print(x3t.shape)
    with tf.name_scope('layer2_dec'):
        x2 = tf.concat((x3t,x2),axis=-1)
        x2 = conv_block(x2,3,32)
        x2 = conv_block(x2,3,32)
        x2 = conv_block(x2,3,32)
        x2t = tf.layers.conv3d_transpose(x2,16,3,strides=2,padding='same')   
    with tf.name_scope('layer1_dec'):
        x1 = tf.concat((x2t,x1),axis=-1)
        x1 = conv_block(x1,3,16)
        x1 = conv_block(x1,3,16)
        x1 = conv_block(x1,3,16)
        x1 =  tf.layers.conv3d(x1, N_CLASS, 1)  
        return(x1)

def choose_patients(data_dir, n_pat):
    a=glob.glob(data_dir + '*')
    random.shuffle(a)
    return(a[0:n_pat])
    
def choose_patch(pid,patch_size,n_patches,force, N_CLASS):
    mri_patches   = np.zeros((len(pid) * n_patches, patch_size, patch_size, patch_size, 1))
    label_patches = np.zeros((len(pid) * n_patches, patch_size, patch_size, patch_size, 1))
    counter=0
    for ii in pid:
        os.chdir(pid[counter])
        mask=nib.load('kds_mask_multiclass.hdr').get_data()
        mri=glob.glob('*rot*hdr')
        mri=nib.load(mri[0]).get_data()
        if mri.shape[-1]>1:
            mri = np.expand_dims(mri,-1)
        if mask.shape[-1]>1:
            mask=np.expand_dims(mask,-1)
        if force == 'positive':
            idx = np.argwhere(mask>0)
        elif force == 'negative':
            idx = np.argwhere(mask==0)
        elif force == 'PATCH_SIZE':
            idx = np.argwhere(mask>-1)
        else:
            raise Exception('You messed up! Choose one of the following for $force: positive, negative, PATCH_SIZE-- CASE SENSITIVE')
        for jj in range(n_patches):
            waiting = 1
            while waiting:
                try:
                    idxx=random.choice(idx)
                    mri_patches[(counter * n_patches) +jj,:,:,:,:]   = mri[max(0,idxx[0]-patch_size/2):min(mri.shape[0],idxx[0]+patch_size/2), max(0,idxx[1]-patch_size/2):min(mri.shape[1],idxx[1]+patch_size/2), max(0,idxx[2]-patch_size/2): min(mri.shape[2],idxx[2]+patch_size/2),:]
                    label_patches[(counter * n_patches) +jj,:,:,:,:]  = mask[max(0,idxx[0]-patch_size/2):min(mask.shape[0],idxx[0]+patch_size/2), max(0,idxx[1]-patch_size/2):min(mask.shape[1],idxx[1]+patch_size/2), max(0,idxx[2]-patch_size/2): min(mask.shape[2],idxx[2]+patch_size/2),:]
                    waiting=0
                except:
                    continue
        counter+=1
        if random.random()>0.7:
            mri_patches=np.flip(mri_patches,1)
            label_patches=np.flip(label_patches,1)
    return mri_patches, label_patches
            
        
        
train_dir = '/media/mira/Data/karl/striatum/in_data/training_Mario/'
val_dir   = '/media/mira/Data/karl/striatum/in_data/validation/'

X = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1],   name='X')  #input
Y = tf.placeholder(tf.int32, shape = [None, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1], name='Y')  #'labels'


def run_model():
    pred = unet(X,N_CLASS)
    cost = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits = pred)
    minLoss=10000000000
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)   
    with tf.name_scope("training"):
            tf.summary.scalar("training_cost", cost, collections=['training'])
 
    with tf.name_scope("validation"):
            tf.summary.scalar("validation_cost", cost, collections=['validation'])
            
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        train_merge      = tf.summary.merge_all(key='training')
        validation_merge = tf.summary.merge_all(key='validation')
        train_writer = tf.summary.FileWriter( graphDir + '/training/', sess.graph)
        validation_writer = tf.summary.FileWriter( graphDir + '/testing/', sess.graph)
        print('Beginning Session!')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('Running Model!')
        
        GLOBAL_STEP=0
        while GLOBAL_STEP <= NUM_ITER:

            if GLOBAL_STEP % 50 != 0:
                print(GLOBAL_STEP)
                x,y=choose_patch(choose_patients(train_dir,NUM_PAT),PATCH_SIZE,NUM_PATCH,'positive',N_CLASS)
                _, acc, c = sess.run([optimizer, train_merge, cost], feed_dict = {X: x, Y: y})
                print(c)
                train_writer.add_summary(acc, GLOBAL_STEP)                    
            else:
                x,y,=choose_patch(choose_patients(val_dir,NUM_PAT-2),PATCH_SIZE,NUM_PATCH,'positive',N_CLASS)
                acc, c= sess.run([validation_merge, cost], feed_dict = {X: x, Y: y})
                validation_writer.add_summary(acc, GLOBAL_STEP)
                print(c)
                if c < minLoss:
                        save_path=saver.save(sess, out_dir + '/model')
                        minLoss=c                        
            GLOBAL_STEP+=1

timestr = time.strftime("%Y%m%d-%H%M%S")
out_dir  = '/media/datadrive/karl/striatum/patches/output_Mario/' + 'new_data_aug/'
try:
    os.mkdir(out_dir)
except:
    pass
graphDir = out_dir
try:
    os.mkdir(graphDir + '/training/')
    os.mkdir(graphDir + '/testing/')
except:
    pass
run_model()

