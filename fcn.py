# python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import vgg
from utils import bilinear_upsample_weights

slim = tf.contrib.slim


def fcn(
    image_tensor,
    upsample_factor,
    number_of_classes,
    annotation_tensor
    ):
    # Define the model that we want to use -- specify to use only two classes at the last layer
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(image_tensor,
                                        num_classes=number_of_classes,
                                        spatial_squeeze=False,
                                        fc_conv_padding='SAME')
    downsampled_logits_shape = tf.shape(logits)
    img_shape = tf.shape(image_tensor)

    # Calculate the ouput size of the upsampled tensor
    # The shape should be batch_size X width X height X num_classes
    upsampled_logits_shape = tf.stack([
                                  downsampled_logits_shape[0],
                                  img_shape[1],
                                  img_shape[2],
                                  downsampled_logits_shape[3]
                                  ])
    if upsample_factor == 32:
        upsample_filter_np_x32 = bilinear_upsample_weights(upsample_factor,number_of_classes)
        upsample_filter_tensor_x32 = tf.Variable(upsample_filter_np_x32, name='vgg_16/fc8/t_conv_x32')
        upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x32,
                                                output_shape=upsampled_logits_shape,
                                                strides=[1, upsample_factor, upsample_factor, 1],
                                                padding='SAME')
    elif upsample_factor == 16:
        pool4_feature = end_points['vgg_16/pool4']
        with tf.variable_scope('vgg_16/fc8'):
            aux_logits_16s = slim.conv2d(pool4_feature, number_of_classes, [1, 1],
                                        activation_fn=None,
                                        weights_initializer=tf.zeros_initializer,
                                        scope='conv_pool4')

        # Perform the upsampling
        upsample_filter_np_x2 = bilinear_upsample_weights(2,  # upsample_factor,
                                                        number_of_classes)

        upsample_filter_tensor_x2 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2')

        upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2,
                                                output_shape=tf.shape(aux_logits_16s),
                                                strides=[1, 2, 2, 1],
                                                padding='SAME')


        upsampled_logits = upsampled_logits + aux_logits_16s

        upsample_filter_np_x16 = bilinear_upsample_weights(upsample_factor,
                                                        number_of_classes)

        upsample_filter_tensor_x16 = tf.Variable(upsample_filter_np_x16, name='vgg_16/fc8/t_conv_x16')
        upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x16,
                                                output_shape=upsampled_logits_shape,
                                                strides=[1, upsample_factor, upsample_factor, 1],
                                                padding='SAME')
    elif upsample_factor == 8:
        pool3_feature = end_points['vgg_16/pool3']
        with tf.variable_scope('vgg_16/fc8'):
            aux_logits_8s = slim.conv2d(pool3_feature, number_of_classes, [1, 1],
                                        activation_fn=None,
                                        weights_initializer=tf.zeros_initializer,
                                        scope='conv_pool3')

        pool4_feature = end_points['vgg_16/pool4']
        with tf.variable_scope('vgg_16/fc8'):
            aux_logits_16s = slim.conv2d(pool4_feature, number_of_classes, [1, 1],
                                        activation_fn=None,
                                        weights_initializer=tf.zeros_initializer,
                                        scope='conv_pool4')

        # 对fc8结果做 upsampling，得到16s
        upsample_filter_np_x2 = bilinear_upsample_weights(2,  # upsample_factor,
                                                        number_of_classes)
        upsample_filter_tensor_x2_1 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2_1')
        upsampled_logits = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x2_1,
                                                output_shape=tf.shape(aux_logits_16s),
                                                strides=[1, 2, 2, 1],
                                                padding='SAME')
        # 求和之后再做一次 upsampling，得到8s
        upsampled_logits = upsampled_logits + aux_logits_16s
        upsample_filter_tensor_x2_2 = tf.Variable(upsample_filter_np_x2, name='vgg_16/fc8/t_conv_x2_2')
        upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x2_2,
                                                output_shape=tf.shape(aux_logits_8s),
                                                strides=[1, 2, 2, 1],
                                                padding='SAME')

        upsampled_logits = upsampled_logits + aux_logits_8s

        upsample_filter_np_x8 = bilinear_upsample_weights(upsample_factor,
                                                        number_of_classes)
        upsample_filter_tensor_x8 = tf.Variable(upsample_filter_np_x8, name='vgg_16/fc8/t_conv_x8')
        upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x8,
                                                output_shape=upsampled_logits_shape,
                                                strides=[1, upsample_factor, upsample_factor, 1],
                                                padding='SAME')

    lbl_onehot = tf.one_hot(annotation_tensor, number_of_classes)
    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=upsampled_logits,
                                                            labels=lbl_onehot)

    cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(cross_entropies, axis=-1))

    return upsampled_logits,cross_entropy_loss