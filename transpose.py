'''
*    Rachel Locke
*    Updated: 10.28.2021
*    CSC 499
*    Mentors: Dr. Alice Armstrong & Dr. C. Dudley Girard
'''

import csv    # Import the CSV library
import math
import numpy as np

def average_pixels(cell1, cell2, cell3, cell4):
    avg = math.trunc((int(cell1)+int(cell2)+int(cell3)+int(cell4))/4)
    return avg

def main():
    
    train_file = open("train.csv")
    train_data = np.loadtxt(train_file, delimiter=",", skiprows=1)
    # <class 'numpy.ndarray'> shape: (50000, 3073)

   # test_file = open("test.csv")
    #test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
    #<class 'numpy.ndarray'> Shape (10000, 3072)
    '''
    print(type(train_data))
    print(train_data.shape)
    print(type(test_data))
    print(test_data.shape)
    print("\n")

    print(train_data[0])
    print(train_data[0,0])
    print("\n")

    print(train_data.shape[0])
    print(test_data.shape[0])
    print("\n")
    '''
    sixteen = ([])
    sixteen_test=([])
    s = 32   # 32x32 source row length is 32
    k = 0
    avg = 0

    #--------------------------------------------------------------------------------------------------------------  
    # Transpose training set 32x32 to 16x16
    for x in range(train_data.shape[0]):
        print(f"I'm on row: {x}")
        for i in range(256):
            #print(train_data[x])
            #print(f"I'm on column: {i}")
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            print(f"I'm on row: {x}")
            print(f"{c1} {c2} {c3} {c4}")
            avg = average_pixels((train_data[x,c1]), (train_data[x,c2]), (train_data[x,c3]), (train_data[x,c4]))
            sixteen = np.append(sixteen, avg)
            print(avg)
    
    sixteen = sixteen.reshape(50000,256)    
    print(sixteen.shape)
    #print(sixteen)

    with open('train_16.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(sixteen)

    '''
    # Transpose testing set 32x32 to 16x16
    for x in range(test_data.shape[0]):
        #for iy in range(255):
        for i in range(256):
            print(test_data[x])
            c1 = (i*2)
            c2 = ((i*2)+1)
            c3 = ((i*2)+s)
            c4 = ((i*2) + s + 1)
            print(f"{c1} {c2} {c3} {c4}")
            avg = average_pixels((test_data[x,c1]), (test_data[x,c2]), (test_data[x,c3]), (test_data[x,c4]))
            print(avg)
            sixteen_test = np.append(sixteen_test, avg)
    
    sixteen_test.reshape(10000,256)
    with open('test_16.csv', 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(sixteen)
    '''
    #------------------------------------------------------------------------------------------------------------------    
    print("done")

if __name__=='__main__':
    main()