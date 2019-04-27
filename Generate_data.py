import os
import numpy as np
import cv2
from keras.utils import to_categorical
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
images=[]
labels=[]

image_size=90
def read_path(path_name,images,labels,image_size):
    for dir in os.listdir(path_name):                       #进入第一级目录
        subdir=os.path.abspath(os.path.join(path_name,dir))
        for dir_item in os.listdir(subdir):                 #获取标签
            full_path=os.path.abspath(os.path.join(subdir,dir_item))
            image=cv2.imread(full_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            images.append(image)

            labels.append(dir)
    return images,labels

def load_dataset(path_name,images,labels,image_size):
    images,labels=read_path(path_name,images,labels,image_size)
    images=np.array(images,dtype='float32')/255
    # images = np.transpose(images)
    images = np.expand_dims(images, axis=3)
    # print(images)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = np.array([labels]).T
    # labels = keras.utils.categorical(labels,7)
    # labels = np.array(labels,dtype='float32')
    # labels = np.transpose(labels)
    # labels = np.expand_dims(labels, axis=1)
    print('label shape:',labels.shape)
    print('image shape:',images.shape)
    # print(images)
    # print(labels)
    # print(labels.shape[0])
    return(images,labels)


if __name__ == '__main__':
    images,labels=load_dataset('Train/',images,labels,image_size)