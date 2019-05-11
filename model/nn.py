from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def highway_layer(input_data, dim, init, name='', reuse=None,
                    activation=tf.nn.relu):
    """ Creates a highway layer
    """
    print("Constructing highway layer..")
    trans = linear(input_data, dim, init,  name='trans_{}'.format(name),
                        reuse=reuse)
    trans = activation(trans)
    gate = linear(input_data, dim, init, name='gate_{}'.format(name),
                        reuse=reuse)
    gate = tf.nn.sigmoid(gate)
    if(dim!=input_data.get_shape()[-1]):
        input_data = linear(input_data, dim, init,name='trans2_{}'.format(name),
                            reuse=reuse)
    output = gate * trans + (1-gate) * input_data
    return output


def projection_layer(inputs, output_dim, name='', reuse=None,
                    activation=None, weights_regularizer=None,
                    initializer=None, dropout=None, use_mode='FC',
                    num_layers=2, mode='', return_weights=False,
                    is_train=False):
    """ Simple Projection layer

    Args:
        x: `tensor`. vectors to be projected
            Shape is [batch_size x time_steps x emb_size]
        output_dim: `int`. dimensions of input embeddings
        rname: `str`. variable scope name
        reuse: `bool`. whether to reuse parameters within same
            scope
        activation: tensorflow activation function
        initializer: initializer
        dropout: dropout placeholder
        use_fc: `bool` to use fc layer api or matmul
        num_layers: `int` number layers of projection

    Returns:
        A 3D `Tensor` of shape [batch, time_steps, output_dim]
    """
    # input_dim = tf.shape(inputs)[2]
    if(initializer is None):
        initializer = tf.contrib.layers.xavier_initializer()
    shape = inputs.get_shape().as_list()
    if(len(shape)==3):
        input_dim = inputs.get_shape().as_list()[2]
        time_steps = tf.shape(inputs)[1]
    else:
        input_dim = inputs.get_shape().as_list()[1]
    with tf.variable_scope('proj_{}'.format(name), reuse=reuse) as scope:
        x = tf.reshape(inputs, [-1, input_dim])
        output = x
        for i in range(num_layers):
            if(dropout is not None and dropout < 1.0):
                output = dropoutz(output, dropout, is_train)
            _dim = output.get_shape().as_list()[1]
            if(use_mode=='FC'):
                weights = tf.get_variable('weights_{}'.format(i),
                              [_dim, output_dim],
                              initializer=initializer)
                zero_init = tf.zeros_initializer()
                bias = tf.get_variable('bias_{}'.format(i), shape=output_dim,
                                            dtype=tf.float32,
                                            initializer=zero_init)
                output = tf.nn.xw_plus_b(output, weights, bias)
            elif(use_mode=='HIGH'):
                output = highway_layer(output, output_dim, initializer,
                                name='proj_{}'.format(i), reuse=reuse)
            else:
                weights = tf.get_variable('weights_{}_{}'.format(i, name),
                              [_dim, output_dim],
                              initializer=initializer)
                output = tf.matmul(output, weights)
            if(activation is not None and use_mode!='HIGH'):
                output = activation(output)


        if(len(shape)==3):
            output = tf.reshape(output, [-1, time_steps, output_dim])


        return output
