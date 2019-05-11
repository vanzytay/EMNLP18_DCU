from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .nn import *
from .cell import *

def DCU(embed, lengths,
        initializer=None, name='', reuse=None,
        dropout=None, widths=[1,2,4,10,25],
        mode='highway', control_mode='',
        is_train=None, dim=None, pooling='avg'):
    """ Implementation of DCU Encoder
    Consist of series of contract-expand
    operations used to learn gating functions.

    inputs are batch size x sequence length x dim
    please ensure sequence length are divisible by width values
    """
    print(embed)
    if(dim is None):
        dim = embed.get_shape().as_list()[2]

    def pulse_op(embed, fuse, reuse=None):
        if(fuse>1):
            if(pooling=='avg'):
                pooled = tf.layers.average_pooling1d(embed, fuse, fuse,
                                                    padding='same')
        else:
            pooled = embed
        proj_pool = projection_layer(pooled, dim,
                        name='fuse{}_{}'.format(fuse, name),
                        reuse=reuse,
                        activation=None,
                        initializer=initializer,
                        dropout=dropout, mode='FC',
                        num_layers=1,
                        is_train=is_train)
        if(fuse>1):
            gate = tf.tile(proj_pool, [1, fuse, 1])
        else:
            gate = proj_pool
        return gate

    fuse_gates = []

    for fuse in widths:
        fuse_gate = pulse_op(embed, fuse, reuse=reuse)
        fuse_gates.append(fuse_gate)

    fuse_gates = tf.concat(fuse_gates, 2)

    batch_size = tf.shape(embed)[0]
    initial_state = tf.tile(tf.Variable(
        tf.zeros([1, dim])), [batch_size, 1])

    if('MLP' in control_mode):
        mid_point = int(len(widths) /2)
        gate = projection_layer(fuse_gates, dim * mid_point,
                        name='combine_gates1_{}'.format(name),
                        reuse=reuse,
                        activation=tf.nn.relu,
                        initializer=initializer,
                        dropout=1.0, mode='FC',
                        is_train=is_train,
                        num_layers=1)
        gate = projection_layer(gate, dim,
                        name='combine_gates2_{}'.format(name),
                        reuse=reuse,
                        activation=None,
                        initializer=initializer,
                        dropout=1.0, mode='FC',
                        is_train=is_train,
                        num_layers=1)
    else:
        gate = projection_layer(fuse_gates, dim,
                        name='combine_gates_{}'.format(name),
                        reuse=reuse,
                        activation=None,
                        initializer=initializer,
                        dropout=1.0, mode='FC',
                        is_train=is_train,
                        num_layers=2)
    gate = tf.nn.sigmoid(gate)
    proj_embed = projection_layer(embed, dim,
                    name='multigran_MLP_{}'.format(name),
                    reuse=reuse,
                    activation=tf.nn.tanh,
                    initializer=initializer,
                    is_train=is_train,
                    dropout=1.0, mode='FC',
                    num_layers=1)
    if(mode=='highway'):
        embed = (embed * gate) + ((1-gate) * proj_embed)
    elif(mode=='recurrent'):
        forget_gate= gate
        pooling = DCU_pooling(dim, 'f')
        initial_state = pooling.zero_state(tf.shape(embed)[0],
                                            tf.float32)
        stack_input = tf.concat([proj_embed, forget_gate], 2)
        embed, _ = tf.nn.dynamic_rnn(pooling, stack_input,
                                initial_state=initial_state,
                                sequence_length=tf.cast(
                                            lengths,tf.int32))
    elif(mode=='dual'):
        """ Dual
        """
        forget_gate = gate
        pooling = DCU_pooling(dim, 'fo')
        # initial_state = pooling.zero_state(tf.shape(embed)[0],
        #                                     tf.float32)
        output_gate = projection_layer(embed, dim,
                        name='output_MLP_{}'.format(name),
                        reuse=reuse,
                        activation=None,
                        initializer=initializer,
                        dropout=dropout, mode='FC',
                        num_layers=1)
        output_gate = tf.nn.sigmoid(output_gate)

        stack_input = tf.concat([proj_embed, forget_gate, output_gate], 2)
        embed, _ = tf.nn.dynamic_rnn(pooling, stack_input,
                                initial_state=initial_state,
                                sequence_length=tf.cast(
                                            lengths,tf.int32))
    elif(mode=='tri'):
        forget_gate = gate
        pooling = DCU_pooling(dim, 'ifo')
        # initial_state = pooling.zero_state(tf.shape(embed)[0],
        #                                     tf.float32)
        output_gate = projection_layer(embed, dim,
                        name='output_MLP_{}'.format(name),
                        reuse=reuse,
                        activation=None,
                        initializer=initializer,
                        dropout=dropout, mode='FC',
                        num_layers=1)
        input_gate = projection_layer(embed, dim,
                        name='input_MLP_{}'.format(name),
                        reuse=reuse,
                        activation=None,
                        initializer=initializer,
                        dropout=dropout, mode='FC',
                        num_layers=1)
        input_gate = tf.nn.sigmoid(input_gate)
        output_gate = tf.nn.sigmoid(output_gate)
        stack_input = tf.concat([proj_embed, input_gate,
                                    forget_gate, output_gate], 2)
        embed, _ = tf.nn.dynamic_rnn(pooling, stack_input,
                                initial_state=initial_state,
                                sequence_length=tf.cast(
                                            lengths,tf.int32))

    return embed, tf.reduce_sum(embed, 1)
