from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import random



import tensorflow as tf
# add the layer #
def add_layer(inputs, wei,bias,activation_function=None):

    Wx_plus_b=tf.add(tf.matmul(inputs,wei),bias)
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

# the autoencoder model #
def auto_model(X):
    en_hl_1=add_layer(X,wei_en_hl_1,bias_en_hl_1,activation_function=tf.nn.sigmoid)
    en_hl_2=add_layer(en_hl_1,wei_en_hl_2,bias_en_hl_2,activation_function=tf.nn.sigmoid)
    encoded=add_layer(en_hl_2,wei_en_hl_3,bias_en_hl_3,activation_function=tf.nn.sigmoid)

    de_hl_1=add_layer(encoded,wei_de_hl_1,bias_de_hl_1,activation_function=tf.nn.sigmoid)
    de_hl_2=add_layer(de_hl_1,wei_de_hl_2,bias_de_hl_2,activation_function=tf.nn.sigmoid)
    decoded=add_layer(de_hl_2,wei_de_hl_3,bias_de_hl_3,activation_function=tf.nn.sigmoid)

    return encoded, decoded
# the trained encoded model connectd a full connection network #
def dnn_model(X):
    dnn_hl_1=add_layer(X,weight_dnn_1_trained,bias_dnn_1_trained,activation_function=tf.nn.sigmoid)
    dnn_hl_2=add_layer(dnn_hl_1,weight_dnn_2_trained,bias_dnn_2_trained,activation_function=tf.nn.sigmoid)
    dnn_encoded=add_layer(dnn_hl_2,weight_dnn_3_trained,bias_dnn_3_trained,activation_function=tf.nn.sigmoid)

    dnn_classifi_1=add_layer(dnn_encoded,wei_dnn_1,bias_dnn_1,activation_function=tf.nn.sigmoid)
    #dnn_classifi_2=add_layer(dnn_classifi_1,wei_dnn_2,bias_dnn_2,activation_function=tf.nn.sigmoid)
    prediction=add_layer(dnn_classifi_1,wei_dnn_3,bias_dnn_3,activation_function=tf.nn.sigmoid)

    return prediction

# Import data from siyuan #

m_6587_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'm_6587.csv')
m_6587 = pd.read_csv(m_6587_path, delimiter = ',')
m_6587_targets_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'm_6587_targets.csv')
m_6587_targets = pd.read_csv(m_6587_targets_path, delimiter = ',')


#Inserted Variables #


x = tf.placeholder("float", [None, 314])
y_ = tf.placeholder("float", [None, 94])

## the trained weights of encoder model
weight_dnn_1_trained=tf.placeholder(tf.float32,shape=[314,256])
weight_dnn_2_trained=tf.placeholder(tf.float32,shape=[256,64])
weight_dnn_3_trained=tf.placeholder(tf.float32,shape=[64,32])

bias_dnn_1_trained=tf.placeholder(tf.float32,shape=[256])
bias_dnn_2_trained=tf.placeholder(tf.float32,shape=[64])
bias_dnn_3_trained=tf.placeholder(tf.float32,shape=[32])

### autoencoder ###

wei_en_hl_1= tf.Variable(tf.random_normal([314,256],mean=0.0, stddev=0.05))
wei_en_hl_2= tf.Variable(tf.random_normal([256, 64],mean=0.0, stddev=0.05))
wei_en_hl_3= tf.Variable(tf.random_normal([64, 32],mean=0.0, stddev=0.05))


wei_de_hl_1= tf.Variable(tf.random_normal([32, 64], mean=0.0, stddev=0.05))
wei_de_hl_2= tf.Variable(tf.random_normal([64, 256], mean=0.0, stddev=0.05))
wei_de_hl_3= tf.Variable(tf.random_normal([256, 314], mean=0.0, stddev=0.05))

bias_en_hl_1=tf.Variable(tf.constant(0.1,shape=[256]))
bias_en_hl_2=tf.Variable(tf.constant(0.1,shape=[64]))
bias_en_hl_3=tf.Variable(tf.constant(0.1,shape=[32]))

bias_de_hl_1=tf.Variable(tf.constant(0.1,shape=[64]))
bias_de_hl_2=tf.Variable(tf.constant(0.1,shape=[256]))
bias_de_hl_3=tf.Variable(tf.constant(0.1,shape=[314]))




encoded, decoded = auto_model(x)


# Cost Function basic term
cross_entropy_auto = -1. * x * tf.log(decoded) - (1. - x) * tf.log(1. - decoded)

# training #
loss_auto = tf.reduce_mean(cross_entropy_auto)
train_step_auto = tf.train.AdagradOptimizer(0.001).minimize(loss_auto)


## the dnn classifier ##

wei_dnn_1= tf.Variable(tf.random_normal([32, 128]))
wei_dnn_2= tf.Variable(tf.random_normal([128, 128]))
wei_dnn_3=tf.Variable(tf.random_normal([128, 94]))

bias_dnn_1=tf.Variable(tf.constant(0.1,shape=[128]))
bias_dnn_2=tf.Variable(tf.constant(0.1,shape=[128]))
bias_dnn_3=tf.Variable(tf.constant(0.1,shape=[94]))

prediction=dnn_model(x)

dnn_cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=prediction))
train_step_dnn = tf.train.AdagradOptimizer(0.001).minimize(dnn_cross_entropy)

# Train
init = tf.initialize_all_variables()


with tf.Session() as sess:

    sess.run(init)

    ## data  transfer and load, created inserted data##
    delete_index_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    delete_index_target = np.array([0])

    m_6587 = np.array(m_6587)
    m_6587_targets = np.array(m_6587_targets)
    data_x = []
    data_y = []


    for i in range(len(np.array(m_6587))):
        data_x.append(np.delete(m_6587[i], delete_index_data))
    data_x = np.array(data_x)

    for i in range(len(np.array(m_6587_targets))):
        data_y.append(np.delete(m_6587_targets[i], delete_index_target))
    data_y = np.array(data_y)

    # random spilt train test data #
    length_sample = len(data_x)
    train_percentage = 0.8
    split_number=int(np.floor(length_sample*0.8))
    print('split number',split_number)
    train_x = data_x[0:split_number]
    train_y=data_y[0:split_number]

    test_x=data_x[split_number:-1]
    test_y=data_y[split_number:-1]
    #print("test_x",test_x[0])
    #print("test_y",test_y[[0,1]])
    #print("test_y",len(test_y[[0,1]]))
    #print("test_x", len(test_x[0]))
    batch_size = 256;

    print('....autoecoder Training....')
    for i in range(15001):

        random_number = random.sample(range(0, len(train_x)), batch_size)

        sample_x = train_x[random_number]
        sample_y = train_y[random_number]

        train_step_auto.run({x: sample_x, y_: sample_y})

        if i % 1000 == 0:
            train_loss = loss_auto.eval({x: sample_x, y_: sample_y})

            print('  step, loss = %6d: %6.3f' % (i, train_loss))

        if i % 1000 == 0:
            test_data = {x: test_x, y_: test_y}

            print('loss (test) = ', loss_auto.eval(test_data))
    print("the decode", decoded.eval({x: test_x[[0, 1]], y_: test_y[[0, 1]]}))
    print("the x...", test_x[[0, 1]])

    print(".....deep neural network classifier training.... ")

    ###  get trained weights from autoencoder ###
    w1, w2, w3, \
    b1, b2, b3 \
        = sess.run([wei_en_hl_1, wei_en_hl_2, wei_en_hl_3,
                    bias_en_hl_1, bias_en_hl_2, bias_en_hl_3])

    for i_dnn in range(200001):

        random_number = random.sample(range(0, len(train_x)), batch_size)
        sample_x = train_x[random_number]
        sample_y = train_y[random_number]
        sess.run(train_step_dnn,feed_dict={x: sample_x, y_: sample_y, weight_dnn_1_trained: w1, weight_dnn_2_trained: w2,
                 weight_dnn_3_trained: w3, bias_dnn_1_trained: b1, bias_dnn_2_trained: b2, bias_dnn_3_trained: b3})

        ## connected the trained weights with a full connection network ##
        if i_dnn % 500 == 0:
            train_loss_dnn = dnn_cross_entropy.eval(
                {x: sample_x, y_: sample_y, weight_dnn_1_trained: w1, weight_dnn_2_trained: w2,
                 weight_dnn_3_trained: w3, bias_dnn_1_trained: b1, bias_dnn_2_trained: b2, bias_dnn_3_trained: b3})

            print('  dnn_step, dnn_loss = %6d: %6.3f' % (i_dnn, train_loss_dnn))
            # print("the prediction",prediction.eval({x:sample_x , y_: sample_y,weight_dnn_1_trained:w1,weight_dnn_2_trained:w2,
            # weight_dnn_3_trained:w3,bias_dnn_1_trained:b1,bias_dnn_2_trained:b2,bias_dnn_3_trained:b3}))
            # print("the sample_y", sample_y)

        if i_dnn % 500 == 0:
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            accuracy_train = sess.run(accuracy, feed_dict={x: sample_x, y_: sample_y, weight_dnn_1_trained: w1,
                                                           weight_dnn_2_trained: w2,
                                                           weight_dnn_3_trained: w3, bias_dnn_1_trained: b1,
                                                           bias_dnn_2_trained: b2, bias_dnn_3_trained: b3})
            error_train = 1 - accuracy_train

            accuracy_test = sess.run(accuracy, feed_dict={x: test_x, y_: test_y, weight_dnn_1_trained: w1,
                                                          weight_dnn_2_trained: w2,
                                                          weight_dnn_3_trained: w3, bias_dnn_1_trained: b1,
                                                          bias_dnn_2_trained: b2, bias_dnn_3_trained: b3})
            error_test = 1 - accuracy_test

            print("....dnn_train_accuracy....", accuracy_train)
            print("....dnn_train_error...", error_train)

            print("....dnn_test_accuracy....", accuracy_test)
