#Importing Dependencies annd Libraries required for training the CNN
print("Importing Dependencies...")
import tensorflow as tf
import numpy as np
import ConvolutionNetwork as cn
import Simple_CNN as Simple_CNN
import Dataset_Generator as Dataset_Generator
import image_preprocess_wCV2 as imp
import time
import os
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
print("Dependencies Imported")
#-------------------------------------------------------------------------------
#Initializing few variables and the hyperparameters
classes=[None,'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
num_class = 37 #Determines the no of classes to predict
num_epoch = 10 #Determines the no of times the training is carried out on the database
BATCH_SIZE = 100 #Determines the size of each batch of training data passed
path = './new_trained_CNN' #Specifies the folder containing the trained CNN
#X will hold the original image tensor from the dataset
X = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
#X_img will hold the 2D representation of the X tensor
X_img = tf.reshape(X, [-1, 28,28,1])
#Y will hold the truth value of the class of the images
Y_Truth = tf.placeholder(tf.float32, shape=[None, num_class], name='Y_Truth')
#Defining the global step and the learning rate during the training
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
learning_rate = 1e-4 #Keeping it defined for the time being
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Building the model
print("Building the model\n\n")
#Storing the model and the one hot prediction in 'model' and 'predict' respectively
model, predict = Simple_CNN.build_model(X_img, 37)
#-------------------------------------------------------------------------------
#Defining Cost, Optimizer and the Accuracy
with tf.name_scope("Optimization") as scope:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cost_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y_Truth)
    cost = tf.reduce_mean(cost_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

with tf.name_scope('Accuracy') as scope:
    correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(Y_Truth, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Adding the cost and accuracy to summary
tf.summary.scalar('Cost', cost)
tf.summary.scalar('Accuracy', accuracy)
#-------------------------------------------------------------------------------
#SAVER
merged_summary = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(path, sess.graph)
#Loading pre-computed values
try:
    print("Restoring last checkpoint ...")
    last_chk_point = tf.train.latest_checkpoint(checkpoint_dir='./')
    saver.restore(sess, save_path=last_chk_point)
    print("Restored checkpoint from:", last_chk_point)
except ValueError:
    print("Failed to restore checkpoint. Initializing variables.")
    sess.run(tf.global_variables_initializer())
#-------------------------------------------------------------------------------
def train():
    #Loading the images from the dataset
    #For now we are iterating on a set of 5000 images and their labels just for testing.
    print("Loading the dataset")
    #If the pre formatted data file exist, we read from that File
    if os.path.exists('./train_data_file.npy'):
        print("File exist. Loading data...")
        train_x = np.load('train_data_file.npy')
        train_y = np.load('train_label_file.npy')
    else:
        print("File does not exist. Running dataset generator program...")
        dg = Dataset_Generator
        #the image-labels and the images are loaded into train_y and train_x
        dg.data_set_gen("train", num_class) #Check the Dataset_Generator class for the details of the function
        train_x = np.load('train_data_file.npy')
        train_y = np.load('train_label_file.npy')
    #Printing certain details about the dataset loaded
    print("Lenght of set: ", len(train_x))

    #Looping over the dataset 'num_epoch' times
    for epoch in range(num_epoch):
        #Estimating the batch size
        batch_size = int(math.ceil(len(train_x) / BATCH_SIZE))

        #Looping over the images and the labels of each batch
        for batch in range(batch_size):
            #Stroing the batches of labels and images in batch_y and batch_x
            batch_x = train_x[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
            batch_y = train_y[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

            #Starting the stopwatch
            start_time = time.time()
            #Running then tensorflow Session
            steps, sumOut, _, error, acu = sess.run([global_step, merged_summary, optimizer, cost, accuracy],
                                            feed_dict={X_img: batch_x, Y_Truth : batch_y})
            duration = time.time()-start_time
            train_writer.add_summary(sumOut, steps)
            #Printing out the resul after training on each batch
            print("Epoch_No. :", epoch, "\nTotal Samples Trained: ", steps*BATCH_SIZE, "\nError: ", error, "\nAccuracy: ", acu)

            #Saving the data after every hundred steps
            if steps%100 == 0:
                print("Saving the model")
                saver.save(sess, path, global_step=steps)

            #End of a Batch-Training
        print("End of epoch no.: ", epoch)
    #End of an Epoch
#-------------------------------------------------------------------------------
def test():
    print("Training completed. Starting Test...")
    accu=0
    #Loading the images from the dataset
    #For now we are iterating on a set of 5000 images and their labels just for testing.
    print("Loading the test dataset")
    #If the pre formatted data file exist, we read from that File
    if os.path.exists('./test_data_file.npy'):
        print("File exist. Loading data...")
        test_x = np.load('test_data_file.npy')
        test_y = np.load('test_label_file.npy')
    else:
        print("File does not exist. Running dataset generator program...")
        dg = Dataset_Generator
        #the image-labels and the images are loaded into train_y and train_x
        dg.data_set_gen("test", num_class) #Check the Dataset_Generator class for the details of the function
        test_x = np.load('test_data_file.npy')
        test_y = np.load('test_label_file.npy')
    #Printing certain details about the dataset loaded
    print("Lenght of set: ", len(test_x))

    #Strting test
    #Estimating the batch size
    batch_size = int(math.ceil(len(test_x) / BATCH_SIZE))
    print("Processing in ",batch_size," no. of batches.")
    #Looping over the images and the labels of each batch
    for batch in range(batch_size):
        #Stroing the batches of labels and images in batch_y and batch_x
        batch_x = test_x[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        batch_y = test_y[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        acc = sess.run(accuracy, feed_dict={X_img: batch_x, Y_Truth: batch_y})
        accu += acc
        print("Completed testing batch: ",batch+1)

    print("Average accuracy of the model with respect to test images: ", (accu/batch_size))
#-------------------------------------------------------------------------------
#Defining the prediction function for a given input images
def predict_final(img_name):
    #Loading the desired image
    img = imp.preprocess(img_name)
    img = img.reshape(1, 28*28*1)
    prediction = sess.run(predict, feed_dict={X: img, Y_Truth:np.array(np.zeros((1, num_class)))})
    prediction = prediction[0]
    high=0
    for i in range(len(prediction)):
        if(prediction[i] > prediction[high]):
            high=i;

    print(classes[high])

#-------------------------------------------------------------------------------
def main():
    ans='y'
    while(ans=='y'):
        print("1. Train\n2. Test\n3. Predict")
        c = int(input("Enter your choice: "))
        if(c==1):
            num_epoch = int(input("Enter Epoch size: ")) #Determined the no of times the training is carried out on the database
            print("Starting training of the CNN:")
            train()

        elif (c==2):
            test()

        else:
            img_name = input("Enter the name of the image: ")
            predict_final(img_name);

        ans = input("Continue? (y/n): ")

    sess.close()
    print("-------------END--------------")

if __name__ == "__main__":
    main()
#-------------------------------------------------------------------------------
