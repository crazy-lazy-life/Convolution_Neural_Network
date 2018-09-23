#Importing essential libraries
import tensorflow as tf
from ConvolutionNetwork import ConvolutionNetwork

def build_model(input, no_class):
    cn = ConvolutionNetwork()
    with tf.name_scope("Main_Model") as scope:
        #Taking the input image
        model = input

        print("-----Creating Convolution Blocks-----")
        #Defining the first Convolution Block
        model = cn.new_conv_layer("Conv11",  model, 1, 5, 32, summary = True)
        model = cn.new_relu_layer(model)
        model = cn.new_conv_layer("Conv12",  model, 32, 5, 32, summary = True)
        model = cn.new_relu_layer(model)
        model = cn.new_pool_layer("Pool1", model)
        print("Created Layer 1")

        #Defining the second Convolution Block
        model = cn.new_conv_layer("Conv21",  model, 32, 5, 64, summary = True)
        model = cn.new_relu_layer(model)
        model = cn.new_conv_layer("Conv22",  model, 64, 5, 64, summary = True)
        model = cn.new_relu_layer(model)
        model = cn.new_pool_layer("Pool2", model)
        print("Created Layer 2")

        #Defining the third Convolution Block
        model = cn.new_conv_layer("Conv31",  model, 64, 5, 128, summary = True)
        model = cn.new_relu_layer(model)
        model = cn.new_conv_layer("Conv32",  model, 128, 5, 128, summary = True)
        model = cn.new_relu_layer(model)
        model = cn.new_pool_layer("Pool3", model)
        print("Created Layer 3")

        print("Creating Final FC Layer")
        #Defining the Fully Connected layer
        #Flattening the output in a single layer
        model = cn.flatten(model)
        #Implementing the Dense layer
        model = cn.new_dense_layer("Dense1", model, 200, summary=True)
        model = cn.new_sigmoid_layer(model)
        model = cn.new_dense_layer("Dense2", model, 200, summary=True)
        model = cn.new_sigmoid_layer(model)
        model = cn.new_dense_layer("Dense3", model, no_class)
        #Implementing Softmax Regression
        predict = cn.new_softmax_layer(model)
        print("Network Created")
        return model, predict
#---------------------------------------------------------------------------------
