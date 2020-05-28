#!/usr/bin/env python
# coding: utf-8

# In[8]:


# keras imports for the dataset and building our neural network
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
import os

def trained_model(n):
    # loading the dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # building the input vector from the 32x32 pixels
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255
    
    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)
    
    # building a linear stack of layers with the sequential model
    model = Sequential()
    
    # convolutional layer
    model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    # convolutional layer
    model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    # convolutional layer
    model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    # convolutional layer
    model.add(Conv2D(250, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    if n>1:
        # convolutional layer
        model.add(Conv2D(300, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
    
    # flatten output of conv
    model.add(Flatten())
    
    # hidden layer
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))
    # output layer
    model.add(Dense(10, activation='softmax'))
    
    print(model.summary())
    
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    # training the model for 20 epochs
    history = model.fit(X_train, Y_train,
              batch_size=128,
              epochs=20,
              validation_data=(X_test, Y_test))
    
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    a=score[1]*100
    model.save("MY_CIFAR10_AlexNet.h5")
    os.system("mv /MY_CIFAR10_AlexNet.h5/project")
    return a
no_layer=1
accuracy_trained_model=trained_model(no_layer)
f = open("accuracy.txt","w+")
f.write(str(accuracy_trained_model))
f.close()
os.system("mv /accuracy.txt /project")


# In[ ]:




