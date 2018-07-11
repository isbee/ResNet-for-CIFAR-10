import numpy as np

def save_checkpoint(saver, sess, _SAVE_PATH, _global_step, write_meta_graph):
    saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step, write_meta_graph=write_meta_graph)
    print("\n Checkpoint saved.\n")
          
def acc(est,gnd):
    ## parameter
    total_num = len(gnd)
    acc = 0
    for i in range(total_num):
        if(est[i]==gnd[i]):
            acc = acc + 1
        else:
            acc = acc
    return (acc / total_num)*100

def test_acc(input_x, input_y, test_x, test_y, sess, y_pred_cls, training):
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    predicted_class = sess.run(y_pred_cls, feed_dict={input_x: test_x, input_y: test_y, training: False})
    
    accRate = acc(predicted_class, test_y.argmax(1))
    return accRate

def get_learning_rate(epoch):
    learning_rate = (0.1 ** ((epoch+1)//20 + 1)) * 1e-3

    return learning_rate

#if __name__ == '__main__':
#    print('main')   