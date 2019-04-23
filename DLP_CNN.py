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
from Generate_data import load_dataset
#一些参数
batch_siz = 128
num_classes = 7
nb_epoch = 60
img_size=90
root_path='./'
feature_size = 2000

#载入数据
x_images=[]
y_labels=[]
x_images,y_labels=load_dataset('Train/',x_images,y_labels,img_size)
x_val_images=[]
y_val_labels=[]
x_val_images,y_val_labels=load_dataset('Val/',x_val_images,y_val_labels,img_size)
x_images=tf.convert_to_tensor(x_images)
x_val_images=tf.convert_to_tensor(x_val_images)
y_labels=tf.convert_to_tensor(y_labels)
y_val_labels=tf.convert_to_tensor(y_val_labels)

# input_shape=(img_size, img_size, 1)

#创建网络模型
input_image = Input(shape=(img_size,img_size,1))
cnn = Conv2D(64, (3, 3), strides=1, padding='same')(input_image)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)
cnn = Conv2D(96,(3,3), strides=1, padding='same')(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)
cnn = Conv2D(128, (3, 3), strides=1, padding='same')(cnn)
cnn = Activation('relu')(cnn)
cnn = Conv2D(128, (3, 3), strides=1, padding='same')(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)
cnn = Conv2D(256, (3, 3), strides=1, padding='same')(cnn)
cnn = Activation('relu')(cnn)
cnn = Conv2D(256, (3, 3), strides=1, padding='same')(cnn)
cnn = Activation('relu')(cnn)
cnn = Flatten()(cnn)
feature = Dense(feature_size, activation='relu')(cnn)
predict = Dense(num_classes, activation='softmax', name='softmax')(feature) #至此，得到一个常规的softmax分类模型


input_target = Input(shape=(7,))
centers = Embedding(num_classes, feature_size)(input_target) #Embedding层用来存放中心
l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([feature,centers])

model_train = Model(inputs=[input_image,input_target], outputs=[predict,l2_loss])
model_train.compile(optimizer='adam', loss=['sparse_categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,0.2], metrics={'softmax':'accuracy'})

model_predict = Model(inputs=input_image, outputs=predict)
model_predict.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_train.fit([x_images, y_labels], epochs=50, steps_per_epoch=800/(batch_siz/32))
# score = model_predict.evaluate([y_labels, y_val_labels],batch_size=128)