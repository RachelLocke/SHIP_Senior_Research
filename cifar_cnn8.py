'''
    Rachel Locke
    Updated: 11.10.2021
    CSC 499
    Mentors: Dr. Alice Armstrong & Dr. C. Dudley Girard
'''
import numpy as np
import tensorflow as tf
import keras
import pandas as pd

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.python.keras.layers.core import Dropout
from sklearn.metrics import confusion_matrix, classification_report

def load_data():
    training_file = open("train_reduced8.csv")
    training_data = np.loadtxt(training_file, delimiter=",")
    # <class 'numpy.ndarray'> Shape (50000, 768)
    
    label_file = open("training_labels.csv")
    label_data = np.loadtxt(label_file, delimiter="\n")
    # <class 'numpy.ndarray'> Shape (50000,)

    test_file = open("test_reduced8.csv")
    testing_data = np.loadtxt(test_file, delimiter=",")

    # All the features and are for the 32x32 dimensions. Just retrieve the testing labels
    (X_unused, y_unused), (X_testUnused, y_test) = cifar10.load_data()

    return training_data, label_data, testing_data, y_test

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']

def normalize_dataset(features):
    features /= 255
    return features

'''
\**************************************************************************************************
* one_hot_encoder
**************************************************************************************************/
*   
*   @param: classes - list of labels
*   @return: encoded - one hot encoded binary matrix to represent the input
===================================================================================================
'''
def one_hot_encoder(classes):
    encoded = to_categorical(classes, num_classes=10)
    return encoded

def cnn():
    drop_rate = 0.5   # Dropout rate of 0.5 for now
    model = models.Sequential()
    # First Conv layer
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(8,8,3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(drop_rate))   
    # Second Conv layer
    model.add(layers.Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))
    '''
    # Third Conv layer
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))
    # Flattening layer
    '''
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(drop_rate))
    # Output layer
    model.add(layers.Dense(units=10, activation='softmax'))

    return model

'''
\**************************************************************************************************
* main
**************************************************************************************************/
* Purpose: acts as the point of execution for the program.
===================================================================================================
'''
def main():
    # Load in the reduced 8x8 image cifar10 dataset
    x_train, y_train, x_test, y_test = load_data()

    # Reshape the 50,000 image row vectors into 8x8x3
    x_train = x_train.reshape((x_train.shape[0],8,8,3))#.transpose(0,2,3,1)
    print(x_train.shape)

    x_test = x_test.reshape((x_test.shape[0],8,8,3))#.transpose(0,2,3,1)
    print(x_test.shape)

    # Reshape the 50,000 training labels
    y_train = y_train.reshape((50000,1))
    print(y_train.shape)
    print(y_test.shape)

    # Normalize the features or image vectors
    x_train = x_train.astype('float32')
    x_train = normalize_dataset(x_train)
    x_test = x_test.astype('float32')
    x_test = normalize_dataset(x_test)
    
    # Encode the labels from label indices to one-hot encoded vectors
    y_train = one_hot_encoder(y_train)
    y_test = one_hot_encoder(y_test)
    print(y_train.shape)
    print(y_test.shape)

    # Create a simple model and print the achitecture summary. keep_prob = dropout rate
    model = cnn()
    model.summary()

    # Loss and Optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    # Evaluate the model; model.evaluate returns a score with indicies 0 (loss) & 1 (Accuracy)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % (acc * 100.0))
    print('Test Loss: ', loss)

    pred = model.predict(x_test)
    pred = np.argmax(pred,axis=1)
    y_true = np.argmax(y_test,axis=1)
    conf = confusion_matrix(y_true, pred)

    FP = conf.sum(axis=0) - np.diag(conf)
    FN = conf.sum(axis=1) - np.diag(conf)
    TP = np.diag(conf)
    TN = conf.sum() - (FP +FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    print(f"{FP} -> False Positives")
    print(f"{FN} -> False Negatives")
    print(f"{TP} -> True Positives")
    print(f"{TN} -> True Negatives")

    conf_df = pd.DataFrame(conf, 
                            columns=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck'],
                            index = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck'])
    
    report = classification_report(y_true, pred)

    print(conf_df)
    print("\n")
    print(report)
    print('\nDone in main!')
    

if __name__=='__main__':
    main()