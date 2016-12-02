import tensorflow as tf
import nvidia_input, nvidia_model
import os
import time
from datetime import datetime
import numpy as np

def train(data_dirs, batch_size=32, num_classes=1, augment_data=True, checkpoint_dir='checkpoints',
          restore_checkpoint=True, checkpoint_file=None, restore_step=None, save_checkpoint_step=1000, save_summary_step=100, log_step=10,
          dropout=0.8, max_steps=100000, num_examples_per_epoch=1000, log_device_placement=False, cameras=None,
          min_angle=None, max_angle=None):

    with tf.Graph().as_default():

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if restore_checkpoint and ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            if not restore_step:
                restore_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

            print('Checkpoint step:', restore_step)
            global_step = tf.Variable(restore_step, trainable=False)

        else:
            global_step = tf.Variable(0, trainable=False)
            restore_step = 0
            print('No checkpoint file found')

        # Get images and labels.
        images, labels = nvidia_input.inputs(batch_size=batch_size, data_dirs=data_dirs, shuffle=True,
                                             num_classes=num_classes, augment_data=augment_data,
                                             num_examples_per_epoch=num_examples_per_epoch, cameras=cameras,
                                             min_angle=min_angle, max_angle=max_angle, raw_labels=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        output = nvidia_model.inference(images, dropout, num_classes=num_classes)

        # Calculate loss.
        _loss = nvidia_model.loss(output, labels, num_classes=num_classes)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = nvidia_model.train(_loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
        sess.run(init)

        if restore_checkpoint and ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, os.path.join(checkpoint_dir, 'model.ckpt-{}'.format(restore_step)))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(checkpoint_dir, sess.graph)

        for step in range(restore_step, max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, _loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % log_step == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step > 0 and step % save_summary_step == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step > 0 and (step % save_checkpoint_step == 0 or (step + 1) == max_steps):
                checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
