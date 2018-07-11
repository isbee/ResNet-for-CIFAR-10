import tensorflow as tf

from model import build_resnet_cifar10
from utills import get_learning_rate, save_checkpoint, test_acc
from data import get_data_set, get_shuffle_data, dense_to_one_hot
from dataInput import dataInput

from time import time
import math
import os

import numpy as np

if __name__ == '__main__': 

    _BATCH_SIZE = 128
    _EPOCH = 500
    _SAVE_PATH = "./ModelInfo/cifar-10/v1" 
    _CHECKPOINT_PATH = "./ModelInfo/cifar-10"  

    train_x, train_y = get_data_set("train")
    test_x, test_y = get_data_set("test")

    # print("train_y ", train_y.shape)
    # _y = np.argmax(train_y, axis=1).reshape(-1, 1)
    # print("argmax ", _y)
    # data_with_label = np.hstack((train_x, _y))
    # np.random.shuffle(data_with_label)
    # train_x = data_with_label[:,:-1]
    # train_y = np.array(data_with_label[:,-1], dtype=int)
    # print(train_y)
    # print(dense_to_one_hot(train_y))
    
    #dataVisual = dataInput()
    #dataVisual.dataVisuallizationSubplot()
    
    x, y, output, y_pred_cls, global_step, learning_rate, training = build_resnet_cifar10(
                                                    bottleneck=True, resnet_size=50)
    
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08).minimize(loss, global_step=global_step)
        
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Optimization
    # https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu
    os.environ["KMP_BLOCKTIME"] = "0"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 2 # Set thread pool with phsycial core
    config.inter_op_parallelism_threads = 2

    with tf.Session(config=config) as sess:
        loss_summary = tf.summary.scalar("train_loss", loss)
        acc_summary = tf.summary.scalar("train_accuracy", accuracy)
        train_summary = tf.summary.merge([loss_summary, acc_summary])

        test_loss_summary = tf.summary.scalar("validation_loss", loss)
        test_acc_summary = tf.summary.scalar("validation_accuracy", accuracy)
        test_summary = tf.summary.merge([test_loss_summary, test_acc_summary])

        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=24*7)
        writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)  

        print("\n Trying to restore last checkpoint ...")
        lastest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=_CHECKPOINT_PATH)
        
        write_meta_graph = True
        if lastest_checkpoint is not None:
            saver.restore(sess, save_path=lastest_checkpoint)
            write_meta_graph = False
            print("\n Restored checkpoint from:", lastest_checkpoint, "\n")
        else:
            print("\n Failed to restore checkpoint. Initializing variables instead. \n")
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

        # try:
        #     print("\n Trying to restore last checkpoint ...")
        #     last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_CHECKPOINT_PATH)
        #     saver.restore(sess, save_path=last_chk_path)
        #     print("\n Restored checkpoint from:", last_chk_path, "\n")
        # # restore() raise ValueError if available checkpoints aren't exist
        # except ValueError:
        #     print("\n Failed to restore checkpoint. Initializing variables instead. \n")
        #     writer.add_graph(sess.graph)
        #     sess.run(tf.global_variables_initializer())
        
        batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
        for epoch in range(_EPOCH):
            train_x, train_y = get_shuffle_data(train_x, train_y)

            print(str(epoch + 1) + ' epoch')
            for step in range(batch_size):
                batch_xs = train_x[step*_BATCH_SIZE: (step+1)*_BATCH_SIZE]
                batch_ys = train_y[step*_BATCH_SIZE: (step+1)*_BATCH_SIZE]
                #batch_xs, batch_ys = tf.train.shuffle_batch([train_x, train_y], batch_size=_BATCH_SIZE, 
                #                                        capacity=50000, min_after_dequeue=10000)            
                    
                start_time = time()
                _global_step, _, batch_loss, batch_acc, _train_summary = sess.run(
                        [global_step, optimizer, loss, accuracy, train_summary],
                        feed_dict={x: batch_xs, y: batch_ys, 
                        learning_rate: get_learning_rate(epoch), training: True}) 
                duration = time() - start_time        

                if step % 10 == 0:
                    # for print current result
                    percentage = int(round((step/batch_size)*100))
                    bar_len = 29
                    filled_len = int((bar_len*int(percentage))/100)
                    bar = '*' * filled_len + '-' * (bar_len - filled_len)
                    msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
                    print(msg.format(_global_step, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))   

            #predicted_test_data = sess.run(y_pred_cls, feed_dict={x: test_x, y: test_y, training: False})
            #correct_test_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
            #test_accuracy = tf.reduce_mean(tf.cast(correct_test_prediction, tf.float32))
            #test_acc_summary = tf.summary.scalar("test_accuracy", test_accuracy)

            _test_summary, _test_acc, _test_loss = sess.run([test_summary, accuracy, loss], feed_dict={x: test_x, y: test_y, training: False})
            print('Validation acc : ', _test_acc, ' %', ' Validation loss : ', _test_loss)
            writer.add_summary(_test_summary, global_step=epoch)

            writer.add_summary(_train_summary, global_step=_global_step)    
            save_checkpoint(saver, sess, _SAVE_PATH, _global_step, write_meta_graph)

            # if (epoch + 1) % 50 == 0:
            #     acc = test_acc(x, y, test_x, test_y, sess, y_pred_cls, training)
            #     print('Test ACC Rate --> ' + str(acc) + ' %')

        acc = test_acc(x, y, test_x, test_y, sess, y_pred_cls, training)
        print('Test ACC Rate --> ' + str(acc) + ' %')
