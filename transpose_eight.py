'''
*    Rachel Locke
*    Updated: 11.8.2021
*    CSC 499
*    Mentors: Dr. Alice Armstrong & Dr. C. Dudley Girard
'''

import csv
import math
import numpy as np

def average_pixels(cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8):
    avg = math.trunc((int(cell1)+int(cell2)+int(cell3)+int(cell4)+
    int(cell5)+int(cell6)+int(cell7)+int(cell8))/8)
    return avg

def main():
    
    train_file = open("train.csv")
    train_data = np.loadtxt(train_file, delimiter=",", skiprows=1)
    # <class 'numpy.ndarray'> shape: (50000, 3073)

    test_file = open("test.csv")
    test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
    #<class 'numpy.ndarray'> Shape (10000, 3072)
  
    eight_training = ([])
    eight_test = ([])
    training_labels = ([])
    training_labels = np.array(training_labels)

    s = 32   # 32x32 source row length is 32
    avg = 0

    '''
    # Create a csv file for the training labels
    for x in range(train_data.shape[0]):
        training_labels = np.append(training_labels,train_data[x,3072])

    training_labels = training_labels.reshape(50000,1)
    
    with open('training_labels.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(training_labels)
    '''
    
    #--------------------------------------------------------------------------------------------------------------  
    # Transpose training set 32x32 to 8x8
    for x in range(train_data.shape[0]):
        reducedRow = ([])
        print(f"I'm on row: {x}")
        print (f"red averages")
        for i in range(64):
            #print(train_data[x])
            c1 = (i*4)
            c2 = ((i*4)+1)
            c3 = ((i*4)+2)
            c4 = ((i*4)+3)
            c5 = ((i*4)+s)
            c6 = ((i*4)+s+1)
            c7 = ((i*4)+s+2)
            c8 = ((i*4)+s+3)

            avg = average_pixels((train_data[x,c1]), (train_data[x,c2]), (train_data[x,c3]), (train_data[x,c4]),
                (train_data[x,c5]), (train_data[x,c6]), (train_data[x,c7]), (train_data[x,c8]))
            
            reducedRow.insert(i,avg)

        print (f"green averages")
        for i in range(64, 128):
            c1 = (i*4)
            c2 = ((i*4)+1)
            c3 = ((i*4)+2)
            c4 = ((i*4)+3)
            c5 = ((i*4)+s)
            c6 = ((i*4)+s+1)
            c7 = ((i*4)+s+2)
            c8 = ((i*4)+s+3)

            avg = average_pixels((train_data[x,c1]), (train_data[x,c2]), (train_data[x,c3]), (train_data[x,c4]),
                (train_data[x,c5]), (train_data[x,c6]), (train_data[x,c7]), (train_data[x,c8]))
            
            reducedRow.insert(i,avg)

        print (f"blue averages")
        for i in range(128,192):
            c1 = (i*4)
            c2 = ((i*4)+1)
            c3 = ((i*4)+2)
            c4 = ((i*4)+3)
            c5 = ((i*4)+s)
            c6 = ((i*4)+s+1)
            c7 = ((i*4)+s+2)
            c8 = ((i*4)+s+3)

            avg = average_pixels((train_data[x,c1]), (train_data[x,c2]), (train_data[x,c3]), (train_data[x,c4]),
                (train_data[x,c5]), (train_data[x,c6]), (train_data[x,c7]), (train_data[x,c8]))
            
            reducedRow.insert(i,avg)

        eight_training = np.array(eight_training)
        eight_training = np.append(eight_training, reducedRow)

    eight_training = eight_training.reshape(50000,192)

    with open('train_reduced8.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(eight_training)

    #---------------------------------------------------------------------------------------------------------------    
    # Transpose testing set 32x32 to 8x8
    for x in range(test_data.shape[0]):
        reducedRow = ([])
        print(f"I'm on row: {x}")
        print (f"red averages")
        for i in range(64):
            #print(test_data[x])
            c1 = (i*4)
            c2 = ((i*4)+1)
            c3 = ((i*4)+2)
            c4 = ((i*4)+3)
            c5 = ((i*4)+s)
            c6 = ((i*4)+s+1)
            c7 = ((i*4)+s+2)
            c8 = ((i*4)+s+3)

            avg = average_pixels((test_data[x,c1]), (test_data[x,c2]), (test_data[x,c3]), (test_data[x,c4]),
                (test_data[x,c5]), (test_data[x,c6]), (test_data[x,c7]), (test_data[x,c8]))
            
            reducedRow.insert(i,avg)

        print (f"green averages")
        for i in range(64, 128):
            c1 = (i*4)
            c2 = ((i*4)+1)
            c3 = ((i*4)+2)
            c4 = ((i*4)+3)
            c5 = ((i*4)+s)
            c6 = ((i*4)+s+1)
            c7 = ((i*4)+s+2)
            c8 = ((i*4)+s+3)

            avg = average_pixels((test_data[x,c1]), (test_data[x,c2]), (test_data[x,c3]), (test_data[x,c4]),
                (test_data[x,c5]), (test_data[x,c6]), (test_data[x,c7]), (test_data[x,c8]))
            
            reducedRow.insert(i,avg)

        for i in range(128, 192):
            c1 = (i*4)
            c2 = ((i*4)+1)
            c3 = ((i*4)+2)
            c4 = ((i*4)+3)
            c5 = ((i*4)+s)
            c6 = ((i*4)+s+1)
            c7 = ((i*4)+s+2)
            c8 = ((i*4)+s+3)
            
            avg = average_pixels((test_data[x,c1]), (test_data[x,c2]), (test_data[x,c3]), (test_data[x,c4]),
                (test_data[x,c5]), (test_data[x,c6]), (test_data[x,c7]), (test_data[x,c8]))
            
            reducedRow.insert(i,avg)
        
        eight_test = np.array(eight_test)
        eight_test = np.append(eight_test, reducedRow)

    eight_test = eight_test.reshape(10000,192)
    print(eight_test.shape)
    
    with open('test_reduced8.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(eight_test)

    #------------------------------------------------------------------------------------------------------------------    

    print("done")

if __name__=='__main__':
    main()