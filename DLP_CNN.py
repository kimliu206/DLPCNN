from keras.layers import Input,Conv2D, MaxPooling2D,Flatten,Dense,Embedding,Lambda
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.optimizers import SGD
import os
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import EarlyStopping
# from keras.optimizers import SGD
import keras
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# from keras.applications import imagenet_utils
import Generate_data
import tensorflow as tf
from tensorflow.python.ops import math_ops
from sklearn.neighbors import KNeighborsClassifier
from Generate_data import load_dataset
#一些参数
batch_siz = 128
num_classes = 7
nb_epoch = 1000
img_size=90
root_path='./'
feature_size = 2000
samples=891
k=3
lam=0.03
# # sess=tf.Session()
m=np.random.rand(891)
sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#载入数据
x_images=[]
y_labels=[]
x_images,y_labels=load_dataset('Train/',x_images,y_labels,img_size)
x_val_images=[]
y_val_labels=[]
x_val_images,y_val_labels=load_dataset('Val/',x_val_images,y_val_labels,img_size)
random_y=np.random.rand(len(y_labels),1)
# x_images=tf.convert_to_tensor(x_images)
# x_val_images=tf.convert_to_tensor(x_val_images)
# y_labels=tf.convert_to_tensor(y_labels)
# y_val_labels=tf.convert_to_tensor(y_val_labels)
x_feature=x_images.reshape(891,-1)
print('feature shape:',x_feature.shape)
# def charbonnier(I_x, I_y, I_t, U, V, e)
#     def loss_fun(y_true, y_pred):
#         loss = K.sqrt(K.pow((U*I_x + V*I_y + I_t), 2) + e)
#         return K.sum(loss)
#     return loss_fun
model_path="model/"
enc = OneHotEncoder()
enc.fit([[0],[1],[2],[3],[4],[5],[6]])
knn_loss=0

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

def KNN_loss(y_true, y_pred,Lambda=0.003,k=3,feature=x_feature):
    # feature=x_feature
    # submodel = KNeighborsClassifier(n_neighbors=k)
    # submodel.fit(feature,y_labels)
    # y_pred_onehot=props_to_onehot(y_pred)
    # y_pred_real = enc.inverse_transform(y_pred_onehot)
    # socre=submodel.score(feature,y_pred_real)
    # loss = 1-socre
    loss1=K.categorical_crossentropy(y_true, y_pred)
    # print(K.is_keras_tensor(y_pred))
#     # KNeighborsClassifier.kneighbors(y_pred,n_neighbors=k)
#     center=K.eval(y_pred)
#     # for i in range(samples):
#     #     dist = np.array([np.sqrt(sum((center[i] -center[s]) ** 2) ) for s in range(samples)])
#     #     ndx =dist.argsort()
#     #     for j in range(k):
#     #         center[i]+=center[ndx[j]]
#     #     center[i]=center[i]/k
#     # center = tf.convert_to_tensor(center)lam*K.mean(math_ops.abs(center - y_true), axis=-1)
    return loss1+Lambda*knn_loss

def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
    return (1-e)*loss1 + e*loss2




#创建网络模型
input_image = Input(shape=(img_size,img_size,1))
cnn = Conv2D(64, (3, 3), strides=1, padding='same',activation='relu')(input_image)
cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)
cnn = Conv2D(96,(3,3), strides=1, padding='same',activation='relu')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)
cnn = Conv2D(128, (3, 3), strides=1, padding='same',activation='relu')(cnn)
cnn = Conv2D(128, (3, 3), strides=1, padding='same',activation='relu')(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=2)(cnn)
cnn = Conv2D(256, (3, 3), strides=1, padding='same',activation='relu')(cnn)
cnn = Conv2D(256, (3, 3), strides=1, padding='same',activation='relu')(cnn)
cnn = Flatten()(cnn)
feature = Dense(feature_size, activation='relu')(cnn)
predict = Dense(num_classes,activation='softmax', name='softmax')(feature) #至此，得到一个常规的softmax分类模型



# input_target = Input(shape=(7,))
# centers = Embedding(num_classes, feature_size)(input_target) #Embedding层用来存放中心
# l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([feature,centers])
# if os.path.exists(keras.models.load_model("model.h5")):
#     model_train=keras.models.load_model("model.h5")
# else:
model_train = Model(inputs=input_image, outputs=predict)
model_train.compile(optimizer=sgd, loss=KNN_loss, metrics=['accuracy'])
    # model_predict = Model(inputs=input_image, outputs=predict)
    # model_predict.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

for epo in range(nb_epoch):
    print("epoch is" + str(epo))
    cost = model_train.fit(x_images, y_labels, epochs=1)
    strModel = 'my_model' + str(epo) + '.h5'
    # while (epo+10)%10 == 0:
    #     model_train.save(model_path+strModel)
    predict = model_train.predict(x_images)
    predict_onehot = props_to_onehot(predict)
    predict_real = enc.inverse_transform(predict_onehot)
    feature = x_feature
    submodel = KNeighborsClassifier(n_neighbors=k)
    submodel.fit(feature, y_labels)
    score=submodel.score(feature, predict_real)
    knn_loss=1-score
    print("KNN_loss:", knn_loss)
# save
print('test before save: ', model_train.predict(x_val_images[0:2]))

model_train.save(model_path+"model.h5")
#
# model_train.fit(x_images,y_labels,epochs=1, batch_size=64)
loss, accuracy = model_train.evaluate(x_val_images, y_val_labels,batch_size=64)
# predict=model_train.predict(x_val_images)
print('\ntest loss',loss)
print('accuracy',accuracy)
# print('predict',predict)
# print('predict shape',predict.shape)
# predict_onehot=props_to_onehot(predict)
# print(predict_onehot.shape)
# # predict_real = np.array([np.argmax(one_hot)for one_hot in predict_onehot])
# predict_real=enc.inverse_transform(predict_onehot)
# # print(predict_real)
# print('predict shape: ',predict_real.shape)
# feature=x_feature
# submodel = KNeighborsClassifier(n_neighbors=k)
# submodel.fit(feature,y_labels)
# score=submodel.score(feature,predict_real)
# knn_loss=1-score
# print(knn_loss)






# input_target = Input(shape=(1,))
# centers = Embedding(num_classes, feature_size)(input_target) #Embedding层用来存放中心
# l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([feature,centers])
#
# model_train = Model(inputs=[input_image,input_target], outputs=[predict,l2_loss])
# model_train.compile(optimizer='adam', loss=['sparse_categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,0.2], metrics={'softmax':'accuracy'})
#
# model_predict = Model(inputs=input_image, outputs=predict)
# model_predict.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model_train.fit([x_images,y_labels], [y_labels,random_y], epochs=60)
# loss, accuracy = model_predict.evaluate(x_val_images, y_val_labels,batch_size=128)
# print('\ntest loss',loss)
# print('accuracy',accuracy)