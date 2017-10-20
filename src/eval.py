# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import data_providers
from tensorflow.contrib import predictor
import pandas as pd 
import datetime

from constant import VALID_MALL_ID

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

def eval(mall_id, timestamp):
    exp_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'saved_models')
    checkpoint_dir = os.path.join(exp_dir, timestamp, 'checkpoints')
    # run_stats = np.load(os.path.join(checkpoint_dir, mall_id, 'run.npz'))
    # print(run_stats)
    model_path = os.path.join(checkpoint_dir, mall_id, 'model.ckpt.meta')
    
    eval_data = data_providers.WIFIDataProviderLatLongAdded(mall_id, 'eval', is_eval=True, batch_size=64, shuffle_order=False)

    #First let's load meta graph and restore weights
    print(model_path)

    num_input = eval_data.inputs.shape[1] # WIFI data input 
    num_output = eval_data.num_classes
    num_hidden_1 = 256
    num_hidden_2 = 256

    # Reconstruct the same graph 
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('data'):
            inputs = tf.placeholder(tf.float32, [None, num_input], 'inputs')
            targets = tf.placeholder(tf.float32, [None, num_output], 'targets')
        keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope('fc-layer-1'):
            hidden_1 = fully_connected_layer(inputs, num_input, num_hidden_1)
            hidden_1 = tf.nn.dropout(hidden_1, keep_prob)
        with tf.name_scope('fc-layer-2'):
            hidden_2 = fully_connected_layer(hidden_1, num_hidden_1, num_hidden_2)
        with tf.name_scope('output-layer'):
            outputs = fully_connected_layer(hidden_2, num_hidden_2, num_output, tf.identity)

        with tf.name_scope('error'):
            error = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
                    tf.float32))

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer().minimize(error)
        saver = tf.train.Saver()

    # Starts predictions 
    with tf.Session(graph=graph) as session:
        saver.restore(session,  os.path.join(checkpoint_dir, mall_id, 'model.ckpt'))
        feed_dict = {inputs: eval_data.inputs, keep_prob: 1.}
        predictions = session.run([outputs], feed_dict)
        print(predictions[0].shape)
        shop_id = np.array([eval_data.shop_list[i] for i in np.argmax(predictions[0], 1)])
        out = np.array([eval_data.row_id, shop_id]).T
    return out

def main(argv):
    # 这告诉电脑，去哪个文件夹找训练好的模型, 得根据不同训练的时间改动
    timestamp_from_training = argv 
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    valid_mall_id = VALID_MALL_ID

    # valid_mall_id = ['m_625', 'm_626', 'm_690',]
    pre_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'predictions', timestamp)
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    # generate predictions for each mall_id
    i = 0
    for mall_id in valid_mall_id:
        print("================ Start evaluation for mall: {0} ================".format(mall_id))
        out = eval(mall_id, timestamp_from_training)
        if i == 0:
            all_df = out
        else:
            all_df = np.concatenate([all_df, out])
        print("The predictions for all know has shape of: ", all_df.shape)
        i += 1
        print("================ End evaluation for mall: {0} ================".format(mall_id))

    all_df = pd.DataFrame(all_df, columns=['row_id', 'shop_id'])
    csv_dir = os.path.join(pre_dir, 'all-eval.csv')
    all_df.to_csv(csv_dir, index=False)
    print("================ Successfully combined all predictions ================".format(csv_dir))


if __name__ == '__main__':
    main(sys.argv[1])