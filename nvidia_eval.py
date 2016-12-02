import tensorflow as tf
import nvidia_input, nvidia_model
import os
import time
import math
from datetime import datetime
import numpy as np
import pandas as pd

def makedirs(dirs):
    try:
        os.makedirs(dirs)
    except OSError:
        if not os.path.isdir(dirs):
            raise

def eval_once(saver, summary_writer, output, labels, summary_op, last_step, num_classes,
              batch_size, checkpoint_dir, restore_step, log_dir, num_examples, unlabeled,
              min_angle, max_angle):
    """Run Eval once.
    Args:
        saver: Saver.
        summary_writer: Summary writer.
        eval_op: Evaluation op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            if not restore_step:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            else:
                saver.restore(sess, os.path.join(checkpoint_dir, 'model.ckpt-{}'.format(restore_step)))
                global_step = restore_step

            #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('{}: Checkpoint step: {}'.format(datetime.now(), global_step))
        else:
            print('{}: No checkpoint found'.format(datetime.now()))
            return last_step

        if global_step == last_step:
            print('{}: Checkpoint already evaluated'.format(datetime.now()))
            return last_step
        else:
            last_step = global_step

        # Start the queue runners.
        coord = tf.train.Coordinator()

        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            result = np.array([])
            example_count = 0
            step = 0

            while example_count < num_examples and not coord.should_stop():

                start_time = time.time()

                #label_value, output_value, imgs = sess.run([labels, output, images])
                label_value, output_value = sess.run([labels, output])

                duration = time.time() - start_time

                if num_classes > 1:
                    # back from vectors to angles...
                    output_value = [nvidia_input.vec_to_angle(ov, num_classes, max_angle, min_angle) for ov in output_value]

                    if len(label_value.shape) > 1:
                        label_value = [nvidia_input.vec_to_angle(lv, num_classes, max_angle, min_angle) for lv in label_value]

                else:
                    output_value = [nvidia_input.angle_from_normalized(ov, max_angle, min_angle) for ov in output_value]

                    if not unlabeled:
                        label_value = [nvidia_input.angle_from_normalized(lv, max_angle, min_angle) for lv in label_value]

                output_value = np.reshape([output_value], [-1, 1])
                label_value = np.reshape([label_value], [-1, 1])
                rows = np.hstack((label_value, output_value))
                result = np.append(result, rows)
                result = np.reshape(result, [-1, 2])

                example_count += output_value.shape[0]

                if step % 10 == 0:
                    #print(label_value[:3], output_value[:3])
                    examples_per_sec = batch_size / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step,
                                         examples_per_sec, sec_per_batch))

                step += 1

            eval_dir = os.path.join(checkpoint_dir, log_dir)
            makedirs(eval_dir)

            if not unlabeled:

                result = pd.DataFrame(result, columns=['label', 'steering_angle'])
                rmse = np.sqrt(np.mean((output_value-label_value)**2))
                print('Root mean square error: {:.6f}'.format(rmse))
                result.iloc[:num_examples+1].to_csv(os.path.join(eval_dir, 'eval-{0:010d}.csv'.format(last_step)), index=False)
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Root mean square error', simple_value=float(rmse))
                summary_writer.add_summary(summary, last_step)

            else: # just generate submission file
                result = pd.DataFrame(result, columns=['frame_id', 'steering_angle'])
                result['frame_id'] = result.frame_id.apply(lambda x :  '%.f' % (x))
                result.iloc[:num_examples+1].to_csv(os.path.join(eval_dir, 'test-{0:010d}.csv'.format(last_step)), index=False)

        except Exception as e:
            coord.request_stop(e)
            print(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        return last_step


def evaluate(data_dirs, num_classes=1, batch_size=128, checkpoint_dir='checkpoints', restore_step=None,
             log_dir='eval', interval=300, run_once=False, num_examples=1000, device='/cpu:0', unlabeled=False,
             cameras=None, min_angle=None, max_angle=math.pi/8):
    """Eval for a number of steps."""

    with tf.Graph().as_default() as g:
        with tf.device(device):
            # Get images and labels.

            if not unlabeled:
                images, labels = nvidia_input.inputs(data_dirs, batch_size=batch_size, shuffle=False,
                                                     num_classes=num_classes, augment_data=False,
                                                     num_examples_per_epoch=1000, raw_labels=False,
                                                     cameras=cameras, min_angle=min_angle, max_angle=max_angle)
            else:
                images, labels = nvidia_input.unlabeled_inputs(data_dirs, batch_size=batch_size)

            # Build a Graph that computes the output predictions from the
            # inference model.
            output = nvidia_model.inference(images, 1., num_classes=num_classes)

            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(nvidia_model.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()

            eval_dir = os.path.join(checkpoint_dir, log_dir)
            makedirs(eval_dir)
            summary_writer = tf.train.SummaryWriter(eval_dir, g)

            last_step = -1

            while True:
                last_step = eval_once(saver, summary_writer, output, labels, summary_op, last_step,
                                      num_classes, batch_size, checkpoint_dir, restore_step, log_dir,
                                      num_examples, unlabeled, min_angle, max_angle)
                if run_once or unlabeled:
                    break
                time.sleep(interval)
