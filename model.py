import tensorflow as tf
import numpy as np

def build_resnet_cifar10(bottleneck, resnet_size):
    if resnet_size % 6 != 2:
        # ResNet's architecture for cifar10 --> 1*(conv+max_pool) + 2n*3*(conv) + 1*(average_pool)
        raise ValueError('Resnet size for cifar10 must be 6n+2, but current: ', resnet_size)

    n = (resnet_size - 2) // 6
    
    _BLOCK_SIZE = [2*n] * 3
    _STRIDE = [1, 2, 2] # for feature map [32, 16, 8]
    _IMAGE_WIDTH = 32
    _IMAGE_HEIGHT = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _FILTER_SIZE = 16

    with tf.name_scope('params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_WIDTH * _IMAGE_HEIGHT * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        training = tf.placeholder(tf.bool, name='training')
        image = tf.reshape(x, [-1, _IMAGE_WIDTH, _IMAGE_HEIGHT, _IMAGE_CHANNELS], name='images')
        """ 
        # *** Future work ***
        # Convert data format channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        #
        # If use NCHW, 
        # 1. Set conv/pooling data_format = 'channels_first' (default= 'channels_last')
        # 2. Set batchnormalization axis = 1
        #
        # * NCHW isn't compatible on CPU(even with MKL support)
        
        if has_gpu:
            image = tf.transpose(image, [0,3,2,1])
        """
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('ResNet-CIFAR10'):
        result = build_block_with_pool(input = image, filter_size = _FILTER_SIZE, kernel_size = 3, conv_stride=1,
                                    bottleneck=bottleneck, pool_stride=1, training_placeholder = training, block_name='first_block')

        projection = False 

        filter_size = _FILTER_SIZE
        image_size = _IMAGE_WIDTH
        for i, block_size in enumerate(_BLOCK_SIZE):
            image_size /= _STRIDE[i]
            filter_size = filter_size * (2**i)
            result = build_composite_block(result=result, block_size=block_size, filter_size=filter_size, kernel_size=3, stride=_STRIDE[i], 
                                            projection=projection, training_placeholder=training, bottleneck=bottleneck, 
                                            block_name='composite_block' + str(i+1))
            projection = True
        
        result = build_fully_connected(result= result, final_size= int((image_size**2) * filter_size), bottleneck=bottleneck, 
                                        training_placeholder=training, block_name= 'fully_connected')
        
        result = tf.layers.dense(inputs=result, units=_NUM_CLASSES)
        y_pred_cls = tf.argmax(result, axis=1)

    return x, y, result, y_pred_cls, global_step, learning_rate, training

def build_block_with_pool(input, filter_size, kernel_size, conv_stride, bottleneck, pool_stride, training_placeholder, block_name):
    with tf.variable_scope(block_name):
        result = tf.layers.conv2d(
            inputs=input,
            filters=filter_size,
            kernel_size=[kernel_size, kernel_size],
            strides=conv_stride,
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer()
        )
        result = tf.layers.max_pooling2d(result, pool_size=[2, 2], strides=pool_stride, padding='SAME')

        if bottleneck:
            result = tf.layers.conv2d(
                inputs=result,
                filters=filter_size*4,
                kernel_size=[1, 1],
                strides=conv_stride,
                padding='SAME',
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer()
            )
    return result

def build_block(result, filter_size, kernel_size, projection, first_conv_stride, training_placeholder, block_name):
    with tf.variable_scope(block_name):
        shortcut = result
        result = tf.layers.batch_normalization(inputs= result, training= training_placeholder,
                                                    fused=True, center=True, scale=True, name= block_name + '_batchNorm')
        result = tf.nn.relu(features= result)

        if projection:
            shortcut = tf.layers.conv2d(
                inputs=result,
                filters=filter_size,
                kernel_size=[1, 1],
                strides=first_conv_stride,
                padding='SAME',
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer()
            )

        result = tf.layers.conv2d(
            inputs=result,
            filters=filter_size,
            kernel_size=[kernel_size, kernel_size],
            padding='SAME',
            strides=first_conv_stride,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer()
        )
        result = tf.layers.batch_normalization(inputs= result, training= training_placeholder,
                                                    fused=True, center=True, scale=True, name= block_name + '_batchNorm2')
        result = tf.nn.relu(features= result, name= block_name + '_relu')

        result = tf.layers.conv2d(
            inputs=result,
            filters=filter_size,
            kernel_size=[kernel_size, kernel_size],
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer()
        )

    return result + shortcut

def build_bottleneck_block(result, filter_size, kernel_size, second_conv_stride, projection, training_placeholder, block_name):
    with tf.variable_scope(block_name):
        shortcut = result
        result = tf.layers.batch_normalization(inputs= result, training= training_placeholder,
                                                    fused=True, center=True, scale=True, name= block_name + '_batchNorm')
        result = tf.nn.relu(features= result, name= block_name + '_relu')

        if projection:
            shortcut = tf.layers.conv2d(
                inputs=result,
                filters=filter_size*4,
                kernel_size=[1, 1],
                strides=second_conv_stride,
                padding='SAME',
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer()
            )

        result = tf.layers.conv2d(
            inputs=result,
            filters=filter_size,
            kernel_size=[1, 1],
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer()
        )
        result = tf.layers.batch_normalization(inputs= result, training= training_placeholder,
                                                    fused=True, center=True, scale=True, name= block_name + '_batchNorm2')
        result = tf.nn.relu(features= result, name= block_name + '_relu2')

        result = tf.layers.conv2d(
            inputs=result,
            filters=filter_size,
            kernel_size=[kernel_size, kernel_size],
            padding='SAME',
            strides=second_conv_stride,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer()
        )
        result = tf.layers.batch_normalization(inputs= result, training= training_placeholder, 
                                                    fused=True, center=True, scale=True, name= block_name + '_batchNorm3')
        result = tf.nn.relu(features= result, name= block_name + '_relu3')

        result = tf.layers.conv2d(
            inputs=result,
            filters=filter_size * 4,
            kernel_size=[1, 1],
            padding='SAME',
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer()
        )

    return result + shortcut

def build_fully_connected(result, final_size, bottleneck, training_placeholder, block_name):
    with tf.variable_scope(block_name):
        if bottleneck:
            final_size *= 4
        result = tf.layers.batch_normalization(inputs= result, training= training_placeholder,
                                                    fused=True, center=True, scale=True, name= block_name + '_batchNorm')
        result = tf.nn.relu(features = result, name= block_name + '_relu')
        result = tf.layers.average_pooling2d(result, pool_size=[2, 2], strides=1, padding='SAME')

        result = tf.reshape(result, [-1, final_size])

    return result

def build_composite_block(result, block_size, filter_size, kernel_size, stride, projection, bottleneck, training_placeholder, block_name):
    with tf.variable_scope(block_name):
        for i in range(block_size):
            if bottleneck:
                result = build_bottleneck_block(result= result, filter_size= filter_size, 
                                    kernel_size= kernel_size, second_conv_stride= stride, projection= projection,
                                    training_placeholder= training_placeholder, block_name= 'bottleneck_block' + str(i+1))
            else:
                result = build_block(result= result, filter_size= filter_size, 
                                    kernel_size= kernel_size, first_conv_stride= stride, projection= projection,
                                    training_placeholder= training_placeholder, block_name= 'block' + str(i+1))
            projection = False
            stride = 1
        
    return result