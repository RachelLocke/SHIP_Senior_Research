'''
*    Rachel Locke
*    Updated: 11.6.2021
*    CSC 499
*    Mentors: Dr. Alice Armstrong & Dr. C. Dudley Girard
'''

import csv    # Import the CSV library
import math
import numpy as np
#import pandas as pd

def average_pixels(cell1, cell2, cell3, cell4):
    avg = math.trunc((int(cell1)+int(cell2)+int(cell3)+int(cell4))/4)
    return avg

def main():
    
    train_file = open("train.csv")
    training_data = np.loadtxt(train_file, delimiter=",", skiprows=1)
    # <class 'numpy.ndarray'> shape: (50000, 3073)

    test_file = open("test.csv")
    test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
    #<class 'numpy.ndarray'> Shape (10000, 3072)

    sixteen_training = ([])
    sixteen_test=([])
    training_labels = ([])
    s = 32   # 32x32 source row length is 32
        
    #--------------------------------------------------------------------------------------------------------------  
    for x in range(training_data.shape[0]):
        reducedRow = ([])
        
        print(f"I'm on row: {x}")
        print (f"red averages")
        for i in range(256):
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            reducedRow.insert(i, average_pixels((training_data[x,c1]), (training_data[x,c2]), (training_data[x,c3]), (training_data[x,c4])))
                    
        print (f"green averages")
        for i in range(256,512):
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            reducedRow.insert(i, average_pixels((training_data[x,c1]), (training_data[x,c2]), (training_data[x,c3]), (training_data[x,c4])))
        
        print (f"blue averages")
        for i in range(512,768):
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            reducedRow.insert(i, average_pixels((training_data[x,c1]), (training_data[x,c2]), (training_data[x,c3]), (training_data[x,c4])))
            #if i == 768:
            #    training_labels.insert(i, training_data[3073])   # Add the image label
        
        #training_labels = np.array(training_labels)
        sixteen_training = np.array(sixteen_training)   # Convert the list to a numpy array
        sixteen_training = np.append(sixteen_training, reducedRow)
    
    sixteen_training = sixteen_training.reshape(50000,768)    
    print(sixteen_training.shape)
    #print(sixteen)

    with open('train_reduced16.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(sixteen_training)
    
   # with open('training_labels.csv', 'w', newline='') as file:
   #     mywriter = csv.writer(file, delimiter=',')
   #     mywriter.writerows(training_labels)
    
    
    # Reduce the test data
    #---------------------------------------------------------------------------------------------------------------
    # Transpose testing set 32x32 to 16x16
    for x in range(test_data.shape[0]):
        reducedRow = ([])
        
        print(f"I'm on row: {x}")
        print (f"red averages")
        for i in range(256):
            
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            reducedRow.insert(i, average_pixels((test_data[x,c1]), (test_data[x,c2]), (test_data[x,c3]), (test_data[x,c4])))

        print (f"green averages")
        for i in range(256,512):
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            reducedRow.insert(i, average_pixels((test_data[x,c1]), (test_data[x,c2]), (test_data[x,c3]), (test_data[x,c4])))
        
        print (f"blue averages")
        for i in range(512,768):
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            reducedRow.insert(i, average_pixels((test_data[x,c1]), (test_data[x,c2]), (test_data[x,c3]), (test_data[x,c4])))
        
        sixteen_test = np.array(sixteen_test)   # Convert the list to a numpy array
        sixteen_test = np.append(sixteen_test, reducedRow)
    
    sixteen_test = sixteen_test.reshape(10000,768)
    print(sixteen_test.shape)
    
    with open('test_reduced16.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(sixteen_test)
    
    #------------------------------------------------------------------------------------------------------------------    
    print("done")

if __name__=='__main__':
    main()