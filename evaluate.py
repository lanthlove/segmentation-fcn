# python3
# -*- coding: utf-8 -*-
import argparse
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from dataset import inputs
import logging
from utils import grayscale_to_voc_impl

import fcn
import perform_crf as crf
import sys
from metrics import mean_IU
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)
slim = tf.contrib.slim

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset_val', type=str)
    parser.add_argument('--upsample_factor', type=int, default=8)
    parser.add_argument('--save_samples', type=int, default=20)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

FLAGS, unparsed = parse_args()
number_of_classes = 21

tf.reset_default_graph()
image_tensor, orig_img_tensor, annotation_tensor = inputs(FLAGS.dataset_val, train=False)

upsampled_logits,_ = fcn.fcn(image_tensor,FLAGS.upsample_factor,number_of_classes,annotation_tensor)

# probabilities we will have to apply softmax.
pred = tf.argmax(upsampled_logits, axis=3)
probabilities = tf.nn.softmax(upsampled_logits)

eval_folder = os.path.join(FLAGS.output_dir, 'eval_n')
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)
ckpt_folder = os.path.join(FLAGS.output_dir, 'train')
checkpoint_path = tf.train.latest_checkpoint(ckpt_folder)

if checkpoint_path:
    tf.logging.info(
        'ckpt exist in %s'% ckpt_folder)
    variables_to_restore = slim.get_model_variables()
else:
    tf.logging.error('No ckpt in %s,program exit'% ckpt_folder)
    sys.exit()


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

init_op = tf.global_variables_initializer()
init_local_op = tf.local_variables_initializer()
saver = tf.train.Saver()

with sess:
    sess.run(init_op)
    sess.run(init_local_op)
    saver.restore(sess, checkpoint_path)
    logging.debug('checkpoint restored from [{0}]'.format(checkpoint_path))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start = time.time()
    IoU_all = 0
    sample_cnt = 0
    try:
        while not coord.should_stop():
            val_pred, val_orig_image, val_annot, val_poss = sess.run([pred, orig_img_tensor, annotation_tensor, probabilities])
            IoU_all = mean_IU(np.squeeze(val_pred), np.squeeze(val_annot)) + IoU_all
            if sample_cnt % 100 == 0:
                logging.debug("validation generated at step [{0}]".format(sample_cnt))
            if sample_cnt < FLAGS.save_samples:
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_img.jpg'.format(sample_cnt)), cv2.cvtColor(np.squeeze(val_orig_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_annotation.jpg'.format(sample_cnt)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_annot)), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_prediction.jpg'.format(sample_cnt)),  cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(val_pred)), cv2.COLOR_RGB2BGR))

                crf_ed = crf.perform_crf(val_orig_image, val_poss, number_of_classes)
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_prediction_crfed.jpg'.format(sample_cnt)), cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR))

                overlay = cv2.addWeighted(cv2.cvtColor(np.squeeze(val_orig_image), cv2.COLOR_RGB2BGR), 1, cv2.cvtColor(grayscale_to_voc_impl(np.squeeze(crf_ed)), cv2.COLOR_RGB2BGR), 0.8, 0)
                cv2.imwrite(os.path.join(eval_folder, 'val_{0}_overlay.jpg'.format(sample_cnt)), overlay)
            sample_cnt = sample_cnt + 1
    except tf.errors.OutOfRangeError:
         logging.debug('Total sampels = %d' % sample_cnt) 
    
    finally:
        coord.request_stop()

    coord.join(threads)
    coord.request_stop()
    coord.join(threads)

IoU = IoU_all/sample_cnt
logging.debug("Validation done. MIoU = %f" %(IoU))