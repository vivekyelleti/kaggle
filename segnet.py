
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply, Concatenate
from keras.utils import np_utils

from Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
#from generator import data_gen_small
#import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import numpy as np
import argparse
import json
import pandas as pd
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize
from skimage.io import imread ,call_plugin
print("How is this ?")
#def batch_generator(x_gen,y_gen):
   # xx_gen=iter(x_gen)
    #yy_gen=iter(y_gen)
    #while xx_gen is not None:
     #   try:
      #      
       #     xx_gen=next(x_gen)
        #    print(xx_gen)
         #   yy_gen=next(y_gen)
          #  yield(xx_gen,yy_gen)
        #except StopIteration:
         #   break
 #   for (xx_gen,yy_gen) in np.nditer(x_gen,y_gen)
  #    yield (xx_gen,yy_gen)
            
            
def CreateSegNet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    #pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

#    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(conv_4)#(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

  #  pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(conv_7)#(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

   # pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(conv_10)#(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

   # pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    #unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(conv_13)#(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

   # unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(conv_16)#(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

  #  unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(conv_19)#(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

#    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(conv_22)#(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

  #  unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(conv_24)#(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    segnet = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return segnet

#generator
def binarylab(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            #print(i+j)
            x[i, j, (labels[i][j]%5)]=1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x



def data_gen_small(img_dir, mask_dir,lists, batch_size, dims, n_labels):
        print(len(lists))
        x=len(lists)
        d=0
        while d!=x:
            ix = np.random.choice(np.arange(len(lists)), batch_size)
            d=d+1
            imgs = []
            labels = []
           # print("data_gen_small was called")
            for i in ix:
                # images
                #print("data_is _ reading")
                #print(cv2.imread(img_dir + lists.iloc[i, 0]+'.jpg')
                #print(img_dir+lists.iloc[i, 0])
                toriginal_img = cv2.imread(img_dir + lists.iloc[i, 0])[:, :, ]
                #print(dims+[3])
                # print(toriginal_img)
                #resized_img = cv2.resize(toriginal_img, dims+[3])
                array_img = img_to_array(toriginal_img)/255
                imgs.append(array_img)
                # masks
                original_mask = cv2.imread(mask_dir + lists.iloc[i, 0])
                resized_mask = ce, n_labelsv2.resize(original_mask, (dims[0], dims[1]))
                array_mask = binarylab(resized_mask[:, :, 0], dims, n_labels)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
        print(imgs)
        print(labels)
        return imgs, labels

def prep_data(mode):
#    assert mode in {'test', 'train'}, \
 #       'mode should be either \'test\' or \'train\''
    img_h=256
    img_w=256
    n_labels=8
    path='kaggle/'
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
  #  n = n_train if mode == 'train' else n_test
    n=3
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(img)
        #label.append(label_map(gt))
        label.append(binarylab(gt,(img_h,img_w),8))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print (mode + ': OK')
    print ('\tshapes: {}, {}'.format(data.shape, label.shape))
    print ('\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print ('\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label



def start():
    # set the necessary list
    print("main method started")
    train_list = pd.read_csv('./kaggle/train.csv',header=None)
    val_list = pd.read_csv('./kaggle/val.csv',header=None)

    # set the necessary directories
    trainimg_dir = 'kaggle/'
    trainmsk_dir = 'kaggle/'
    valimg_dir = 'kaggle/'
    valmsk_dir = 'kaggle/'
    batch_size=1
  #  train_list='kaggle/train/trainimg.txt'
   # val_list='kaggle/val/valimg.txt'
    n_labels=8
    ipshape=(256,256,1)
    pool_size=(2,2)
    kernel=3
   # print("train gen done")
   # labels=[]
    #train_img,train_labels = data_gen_small(trainimg_dir, trainmsk_dir,train_list,batch_size,[ipshape[0], ipshape[1]],n_labels)
    train_img,train_labels=prep_data('train')
    train_data_new = train_img[:,:,:,:,0].transpose((0,2,3,1))
    print("train gen done")
    print("val gen called")
    #val_img,val_labels = data_gen_small(valimg_dir, valmsk_dir,val_list,batch_size, [ipshape[0],ipshape[1]], n_labels)
    val_img,val_labels=prep_data('val')
    print("val gen done")
    segnet = CreateSegNet(ipshape, n_labels, kernel, pool_size, Activation('softmax'))
    print(segnet.summary())
   
    print("compilation started")
    segnet.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
    print("compiltion succesfull")
#    fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    nb_epoch=5
    #istory=segnet.fit(train_img,train_labels,epochs=2,verbose=1,callbacks=None,validation_split=0.6,validation_data=None,shuffle=True,class_weight=None,sample_weight=None,initial_epoch=0)#steps_per_epoch=2,validation_steps=3)
    history = segnet.fit(train_data_new, train_labels,batch_size=batch_size, epochs=nb_epoch, verbose=1)                   
    print("Trained")
    segnet.save_weights("../SegNet"+str(n_epochs)+".hdf5")
    print("sava weight done..")

    json_string = segnet.to_json()
    open("../SegNet.json", "w").write(json_string)
start()




