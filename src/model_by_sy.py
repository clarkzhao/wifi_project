# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import data_providers
import datetime
from constant import VALID_MALL_ID

def train(mall_id, timestamp,
        learning_rate=1e-3, 
        num_epoch = 10, 
        num_hidden_1 = 256,
        num_hidden_2 = 256):
    def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
            'weights')
        biases = tf.Variable(tf.zeros([output_dim]), 'biases')
        outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
        return outputs
        
    with tf.name_scope('input'):
        train_data = data_providers.WIFIDataProviderLatLongAdded(mall_id, 'train', batch_size=64)
        valid_data = data_providers.WIFIDataProviderLatLongAdded(mall_id, 'valid', batch_size=64)

    DROPOUT = 0.80 # Dropout, probability to keep units
    num_input = train_data.inputs.shape[1] # WIFI data input 
    num_output = train_data.num_classes

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

        # Create a summary to monitor cost tensor
        tf.summary.scalar("error", error)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", accuracy)
        # Merge all summaries into a single op
        summary_op = tf.summary.merge_all()
            
        # create objects for writing summaries and checkpoints during training
        
        exp_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'saved_models')
        checkpoint_dir = os.path.join(exp_dir, timestamp, 'checkpoints')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        init = tf.global_variables_initializer()
        
        sess = tf.InteractiveSession(graph=graph)
        
        valid_inputs = valid_data.inputs
        valid_targets = valid_data.to_one_of_k(valid_data.targets)
        sess.run(init)

        # 'Saver' op to save and restore all the variables, only keep the best model to save storage space
        saver = tf.train.Saver(max_to_keep=1)

        # to store model performance for future analysis
        train_accuracy = np.zeros(num_epoch)
        train_error = np.zeros(num_epoch)
        valid_accuracy = np.zeros(num_epoch)
        valid_error = np.zeros(num_epoch)
        time_cost = np.zeros(num_epoch)
        num_valid = valid_data.inputs.shape[0]

        for e in range(num_epoch):
            #Training sets
            best_epoch = 0
            for b, (input_batch, target_batch) in enumerate(train_data):
                index = e * train_data.num_batches + b
                _, batch_error, batch_acc, summary = sess.run(
                    [train_step, error, accuracy, summary_op], 
                    feed_dict={inputs: input_batch, targets: target_batch, keep_prob: DROPOUT})
                train_error[e] += batch_error
                train_accuracy[e] += batch_acc

            train_error[e] /= train_data.num_batches
            train_accuracy[e] /= train_data.num_batches
            # Validation on model when finishing one epoch 
            valid_error[e], valid_accuracy[e] = sess.run([error, accuracy], 
                                            feed_dict={inputs: valid_inputs, 
                                            targets: valid_targets,
                                            keep_prob: 1.})
            if e % 5 == 0:
                print('End of epoch {0:02d}: err(train)={1:.4f} acc(train)={2:.4f}'
                        .format(e + 1, train_error[e], train_accuracy[e]))
                print('                 err(valid)={0:.4f} acc(valid)={1:.4f}'
                        .format(valid_error[e], valid_accuracy[e]))
            
            if valid_error[e] <= np.amin(valid_error[:e+1]):
                print("found better model in epoch: ", e+1)
                print("      validation error: ", valid_error[e])
                print("      validation accuracy: ", valid_accuracy[e])
                saver.save(sess, os.path.join(checkpoint_dir, mall_id, 'model.ckpt'))
                print("saved model")
                # Save model weights to disk

        np.savez_compressed(
            os.path.join(checkpoint_dir, mall_id, 'run.npz'),
            train_error=train_error,
            train_accuracy=train_accuracy,
            valid_error=valid_error,
            valid_accuracy=valid_accuracy,
            time_cost = time_cost,
            num_valid = num_valid
        )
        best_valid_error = np.amin(valid_error)
        correspond_valid_acc = valid_accuracy[np.argmin(valid_error)]
        return best_valid_error, correspond_valid_acc, num_valid

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    valid_mall_id = VALID_MALL_ID
    num_valid_accumulated = 0
    valid_accuracy_accumulated = 1.
    i = 0
    for mall_id in valid_mall_id:
        print("\n")
        print("================ Start training for mall: {0} ================".format(mall_id))
        best_valid_error, correspond_valid_acc, num_valid = train(mall_id, timestamp)
        print("================ End training for mall: {0} ================".format(mall_id))
        print("best valid error: ", best_valid_error)
        print("corresponding valid accuracy: ", correspond_valid_acc)
        if i == 0:
            num_valid_accumulated = num_valid
            valid_accuracy_accumulated = correspond_valid_acc
        else:
            valid_accuracy_accumulated = valid_accuracy_accumulated * num_valid_accumulated + correspond_valid_acc * num_valid
            num_valid_accumulated += num_valid
            valid_accuracy_accumulated /= num_valid_accumulated
        print("Overall valid accuracy for all models trained so far is: {0:.4f}".format(valid_accuracy_accumulated))
        i += 1
    print("Finish training: timestamp for this training is: ")
    print(timestamp)
        