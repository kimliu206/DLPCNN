from keras.layers import Input,Conv2D, MaxPooling2D,Flatten,Dense,Embedding,Lambda
from keras.models import Model
from keras import backend as K
import keras
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import numpy as np
from keras.applications import imagenet_utils
import Generate_data
import tensorflow as tf
from tensorflow.python.ops import math_ops

def KNN_loss(y_true, y_pred,lam=0.03,samples=891,k=3):
    for i in range(samples):
        dist = np.array([np.sqrt(sum((y_pred[i] -y_pred[s]) ** 2) ) for s in range(samples)])
        ndx =dist.argsort()
        for j in range(k):
            y_pred[i]+=y_pred[ndx[j]]
        y_pred[i]=y_pred[i]/k
    return lam*K.mean(math_ops.abs(y_pred - y_true), axis=-1)+K.categorical_crossentropy(y_true, y_pred)

if __name__ == '__main__':
    y_true=np.random.rand(891,1)
    y_pred=np.random.rand(891,1)
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    samples=891
    k=3
    for i in range(samples):
        dist = np.array([np.sqrt(sum((y_pred[i] -y_pred[s]) ** 2) ) for s in range(samples)])
        ndx =dist.argsort()
        for j in range(k):
            y_pred[i]+=y_pred[ndx[j]]
        y_pred[i]=y_pred[i]/k
    print(y_pred.shape)
    KNN_loss(y_true,y_pred)