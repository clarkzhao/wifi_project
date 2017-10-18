from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import data_providers

def train(mall_id, 
        learning_rate=1e-3, 
        num_epoch = 50, 
        num_hidden_1 = 256):

    def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
            'weights')
        biases = tf.Variable(tf.zeros([output_dim]), 'biases')
        outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
        return outputs
        
    with tf.name_scope('input'):
        train_data = data_providers.WIFIDataProvider(mall_id, 'train', batch_size=64)
        valid_data = data_providers.WIFIDataProvider(mall_id, 'valid', batch_size=64)

    num_input = train_data.inputs.shape[1] # WIFI data input 
    num_output = train_data.num_classes

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('data'):
            inputs = tf.placeholder(tf.float32, [None, num_input], 'inputs')
            targets = tf.placeholder(tf.float32, [None, num_output], 'targets')
        with tf.name_scope('fc-layer-1'):
            hidden_1 = fully_connected_layer(inputs, num_input, num_hidden_1)
        with tf.name_scope('output-layer'):
            outputs = fully_connected_layer(hidden_1, num_hidden_1, num_output, tf.identity)

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
            
        init = tf.global_variables_initializer()
        
    #     log_dir = os.path.join(os.path.dirname(os.getcwd()), '', 'ccf_first_round_shop_info.csv')
    #     train_writer = tf.summary.FileWriter(log_dir + '/train', graph=graph)
    #     valid_writer = tf.summary.FileWriter(log_dir + '/valid', graph=graph)

        sess = tf.InteractiveSession(graph=graph)
        
        valid_inputs = valid_data.inputs
        valid_targets = valid_data.to_one_of_k(valid_data.targets)
        sess.run(init)
        run_stats = []
        for e in range(num_epoch):
            train_err = 0.
            train_acc = 0.
            #Training sets
            for b, (input_batch, target_batch) in enumerate(train_data):
                index = e * train_data.num_batches + b
                _, batch_error, batch_acc, summary = sess.run(
                    [train_step, error, accuracy, summary_op], 
                    feed_dict={inputs: input_batch, targets: target_batch})
    #             train_writer.add_summary(summary, index)
                train_err += batch_error
                train_acc += batch_acc
    #             if index % 50 == 0:
    #                 valid_summary = sess.run(
    #                 summary_op, feed_dict={inputs: valid_inputs, targets: valid_targets})
    #                 valid_writer.add_summary(valid_summary, index)
            train_err /= train_data.num_batches
            train_acc /= train_data.num_batches
            # Validation on model when finishing one epoch 
            valid_err, valid_acc = sess.run([error, accuracy], 
                                            feed_dict={inputs: valid_inputs, targets: valid_targets})
            if e % 5 == 0:
                print('End of epoch {0:02d}: err(train)={1:.4f} acc(train)={2:.4f}'
                        .format(e + 1, train_err, train_acc))
                print('                 err(valid)={0:.4f} acc(valid)={1:.4f}'
                        .format(valid_err, valid_acc))
            run_stats.append([train_err, train_acc, valid_err, valid_acc])
        return run_stats

if __name__ == '__main__':
    valid_mall_id = ['m_6587', 'm_625', 'm_2182']
    for mall_id in valid_mall_id:
        print("================ Start training for mall: {0} ================".format(mall_id))
        stats = train(mall_id)
        stats = np.array(stats)
        print("================ End training for mall: {0} ================".format(mall_id))
        print("best valid error: ", np.amin(stats[:,2]))
        print("best valid accuracy: ", np.amax(stats[:,3]))
        