import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Traing data size: %d " % mnist.train.num_examples)

print("Validating data size: %d " % mnist.validation.num_examples)

print("Testing data size: %d " % mnist.test.num_examples)

print("Example Trainning data: %s " % mnist.train.images[0])

print("Example Trainning label: %s " % mnist.train.labels[0])




