import os
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
images=[]
labels=[]
image_size=90
def read_path(path_name,images,labels,image_size):
    for dir in os.listdir(path_name):
        subdir=os.path.abspath(os.path.join(path_name,dir))
        for dir_item in os.listdir(subdir):
            full_path=os.path.abspath(os.path.join(subdir,dir_item))
            image=cv2.imread(full_path)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            images.append(image)
            labels.append(dir)
    return images,labels

def load_dataset(path_name,images,labels,image_size):
    images,labels=read_path(path_name,images,labels,image_size)
    images=np.array(images,dtype='float')
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(images.shape)
    print(labels)
    return(images,labels)


if __name__ == '__main__':
    images,labels=load_dataset('Train/',images,labels)