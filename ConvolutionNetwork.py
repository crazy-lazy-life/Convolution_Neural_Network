#Lets create a Convolution Network.
import tensorflow as tf

#Lets create a different class.
class ConvolutionNetwork:
  def __init__(self):
      pass

  def new_conv_layer(self, name, input_layer, num_input_channel, filter_size, num_filters,summary = False):
      with tf.name_scope(name) as scope:
          #Shaping the filter weights
          shape = [filter_size, filter_size, num_input_channel, num_filters]
          #Assigning weights to the filter matrix
          weights = tf.Variable(tf.truncated_normal(shape, stddev = 0.05))
          #Creating new biases
          biases = tf.Variable(tf.constant(0.05, shape = [num_filters]))

          #Performing the tensorflow operation
          conv = tf.nn.conv2d(input = input_layer, filter = weights, strides = [1,1,1,1], padding = 'SAME')
          #Adding the biases
          conv+=biases
          return conv

  #Applying RELU Activation function after each
  def new_relu_layer(self, input_layer):
      with tf.name_scope("Activation") as scope:
          return tf.nn.relu(input_layer)

  #Performing pooling operation after convolution
  def new_pool_layer(self, name, input_layer):
      with tf.name_scope(name) as scope:
          #Performing the Pooling operation:
          pool = tf.nn.max_pool(value = input_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
          return pool

  #Flattening the layer after convolutinon layers
  def flatten(self, input_layer):
      with tf.name_scope("Flatten") as scope:
          input_size = input_layer.get_shape().as_list()
          new_size = input_size[-1]*input_size[-2]*input_size[-3]
          return tf.reshape(input_layer, [-1, new_size])

  #Implementing the fully connected network
  def new_dense_layer(self, name, input_layer, size, summary = False):
      with tf.name_scope(name) as scope:
          input_size = input_layer.get_shape().as_list()[-1]
          #Creating new weights and biases
          weights = tf.Variable(tf.truncated_normal([input_size, size], stddev = 0.05), name='dense_weight')
          if summary:
              tf.summary.histogram(weights.name, weights)
          biases = tf.Variable(tf.truncated_normal([size], stddev = 0.05), name='dense_biases')
          #Performing multiplication of the inputs an the weights and adding the biases
          dense = tf.matmul(input_layer, weights) + biases
          return dense

  #Applyingt the sigmoid function to every output of each dense layer
  def new_sigmoid_layer(self, input_layer):
      with tf.name_scope("Activation") as scope:
          return tf.nn.sigmoid(input_layer)

  #Using softmax to normalize the output
  def new_softmax_layer(self, input_layer):
      with tf.name_scope("Activation") as scope:
          return tf.nn.softmax(input_layer)
