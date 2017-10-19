from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import data_providers
import datetime


def train(mall_id, timestamp,
        learning_rate=1e-3, 
        num_epoch = 20, 
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
            
        # create objects for writing summaries and checkpoints during training
        
        exp_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'saved_models')
        checkpoint_dir = os.path.join(exp_dir, timestamp, 'checkpoints')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train-summaries'))
        # valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid-summaries'))

        init = tf.global_variables_initializer()
        
    #     log_dir = os.path.join(os.path.dirname(os.getcwd()), '', 'ccf_first_round_shop_info.csv')
    #     train_writer = tf.summary.FileWriter(log_dir + '/train', graph=graph)
    #     valid_writer = tf.summary.FileWriter(log_dir + '/valid', graph=graph)

        sess = tf.InteractiveSession(graph=graph)
        
        valid_inputs = valid_data.inputs
        valid_targets = valid_data.to_one_of_k(valid_data.targets)
        sess.run(init)

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver(max_to_keep=1)

        train_accuracy = np.zeros(num_epoch)
        train_error = np.zeros(num_epoch)
        valid_accuracy = np.zeros(num_epoch)
        valid_error = np.zeros(num_epoch)
        time_cost = np.zeros(num_epoch)


        for e in range(num_epoch):
            #Training sets
            best_epoch = 0
            for b, (input_batch, target_batch) in enumerate(train_data):
                index = e * train_data.num_batches + b
                _, batch_error, batch_acc, summary = sess.run(
                    [train_step, error, accuracy, summary_op], 
                    feed_dict={inputs: input_batch, targets: target_batch})
    #             train_writer.add_summary(summary, index)
                # train_err += batch_error
                # train_acc += batch_acc
                # train_writer.add_summary(summary, step)
                train_error[e] += batch_error
                train_accuracy[e] += batch_acc
    #             if index % 50 == 0:
    #                 valid_summary = sess.run(
    #                 summary_op, feed_dict={inputs: valid_inputs, targets: valid_targets})
    #                 valid_writer.add_summary(valid_summary, index)
            # train_err /= train_data.num_batches
            # train_acc /= train_data.num_batches
                    # normalise running means by number of batches
            train_error[e] /= train_data.num_batches
            train_accuracy[e] /= train_data.num_batches
            # Validation on model when finishing one epoch 
            valid_error[e], valid_accuracy[e] = sess.run([error, accuracy], 
                                            feed_dict={inputs: valid_inputs, targets: valid_targets})
            if e % 5 == 0:
                print('End of epoch {0:02d}: err(train)={1:.4f} acc(train)={2:.4f}'
                        .format(e + 1, train_error[e], train_accuracy[e]))
                print('                 err(valid)={0:.4f} acc(valid)={1:.4f}'
                        .format(valid_error[e], valid_accuracy[e]))
            
            if valid_error[e] <= np.amin(valid_error[:e+1]):
                print("found better model")
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
            time_cost = time_cost
        )
        return valid_error, valid_accuracy

if __name__ == '__main__':
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
    for mall_id in valid_mall_id:
        print("================ Start training for mall: {0} ================".format(mall_id))
        valid_error, valid_accuracy = train(mall_id, timestamp)
        print("================ End training for mall: {0} ================".format(mall_id))
        print("best valid error: ", np.amin(valid_error))
        print("best valid accuracy: ", np.amax(valid_accuracy))
        