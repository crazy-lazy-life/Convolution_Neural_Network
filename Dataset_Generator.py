import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img_data = []

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def data_set_gen(set_type, num_classes):
    if set_type is "train":
        read_data = pd.read_csv("./datasets/emnist-letters-train.csv", header = None)
        print("Dataset read.")
        #Loading class Labels
        class_labels = read_data[0].values
        #Converting to one-hot vector
        ret_class_labels = dense_to_one_hot(class_labels,num_classes)
        print("Lables acquired")
        data = read_data.loc[:,1:28*28].values
        print("Data acquired")
        print ("length of the dataset: ", len(data))
        #Acquiring only the first 5000 images
        print("Converting data to desired datatype")
        for i in range(len(data)):
            X = data[i]
            #Reshaping the data into suitable format
            X_img = X.reshape((28,28,1))
            X_img = np.transpose(X_img, (1,0,2))
            #Adding data to the list
            img_data.append(X_img)

        #Returning the acquired data as an array
        ret_data = np.asarray(img_data)
        np.save("train_label_file.npy", ret_class_labels)
        np.save("train_data_file.npy", ret_data)
        print("Completed writing to train_file.")
        #return ret_class_labels, ret_data

    elif set_type is "test":
        img_data=[]
        read_data = pd.read_csv("./datasets/emnist-letters-test.csv", header = None)
        print("Dataset read.")
        #Loading class Labels
        class_labels = read_data[0].values
        #Converting to one-hot vector
        ret_class_labels = dense_to_one_hot(class_labels,num_classes)
        print("Lables acquired")
        data = read_data.loc[:,1:28*28].values
        print("Data acquired")
        #Acquiring only the first 5000 images
        print("Converting data to desired datatype")
        for i in range(len(data)):
            X = data[i]
            #Reshaping the data into suitable format
            X_img = X.reshape((28,28,1))
            X_img = np.transpose(X_img, (1,0,2))
            #Adding data to the list
            img_data.append(X_img)

        #Returning the acquired data in a file
        ret_data = np.asarray(img_data)
        np.save("test_label_file.npy", ret_class_labels)
        np.save("test_data_file.npy", ret_data)
        print("Completed writing to test file.")
        #return ret_class_labels, ret_data
'''
def main():
    data_set_gen("train", 37)

if __name__ == "__main__":
    main()
'''
