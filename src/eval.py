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
    
    eval_data = data_providers.WIFIDataProvider(mall_id, 'eval', is_eval=True, batch_size=64, shuffle_order=False)

    #First let's load meta graph and restore weights
    print(model_path)

    num_input = eval_data.inputs.shape[1] # WIFI data input 
    num_output = eval_data.num_classes
    num_hidden_1 = 256
    num_hidden_2 = 128

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
    valid_mall_id = ['m_690', 'm_6587', 'm_5892', 'm_625', 'm_3839', 'm_3739', 
                            'm_1293', 'm_1175', 'm_2182', 'm_2058', 'm_3871', 'm_3005', 
                            'm_822', 'm_2467', 'm_4406', 'm_909', 'm_4923', 'm_2224', 
                            'm_2333', 'm_4079', 'm_5085', 'm_2415', 'm_4543', 'm_7168', 
                            'm_2123', 'm_4572', 'm_1790', 'm_3313', 'm_4459', 'm_1409', 
                            'm_979', 'm_7973', 'm_1375', 'm_4011', 'm_1831', 'm_4495', 
                            'm_1085', 'm_3445', 'm_626', 'm_8093', 'm_4828', 'm_6167', 
                            'm_3112', 'm_4341', 'm_622', 'm_4422', 'm_2267', 'm_615', 
                            'm_4121', 'm_9054', 'm_4515', 'm_1950', 'm_3425', 'm_3501', 
                            'm_4548', 'm_5352', 'm_3832', 'm_1377', 'm_1621', 'm_1263', 
                            'm_2578', 'm_2270', 'm_968', 'm_1089', 'm_7374', 'm_2009', 
                            'm_6337', 'm_7601', 'm_623', 'm_5154', 'm_5529', 'm_4168', 
                            'm_3916', 'm_2878', 'm_9068', 'm_3528', 'm_4033', 'm_3019', 
                            'm_1920', 'm_8344', 'm_6803', 'm_3054', 'm_8379', 'm_1021', 
                            'm_2907', 'm_4094', 'm_4187', 'm_5076', 'm_3517', 'm_2715', 
                            'm_5810', 'm_5767', 'm_4759', 'm_5825', 'm_7994', 'm_7523', 
                            'm_7800']

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