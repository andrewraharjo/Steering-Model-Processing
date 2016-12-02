'''

Model based on End-to-end Learning for Self-Driving Cars

https://arxiv.org/pdf/1604.07316v1.pdf

'''

import tensorflow as tf
import re

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
USE_FP16 = False

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
#BATCH_SIZE = 128

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """

    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer=tf.constant_initializer(0)):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """

    with tf.device('/cpu:0'):
        dtype = tf.float16 if USE_FP16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev=0.1, wd=4e-5):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if USE_FP16 else tf.float32
    #var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    var = _variable_on_cpu(name, shape, tf.contrib.layers.variance_scaling_initializer())

    if wd > 0:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def inference(images, keep_prob, num_classes=1):
    """Build the model.
    Args:
        images: Images returned from distorted_inputs() or inputs().
    Returns:
        Steering angles.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 24])
        conv = conv2d(images, kernel, 2)
        biases = _variable_on_cpu('biases', [24])
        #bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(conv + biases, name=scope.name)
        _activation_summary(conv1)

    # pool1
    #pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
    #                     padding='VALID', name='pool1')
    # norm1
    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 24, 36])
        conv = conv2d(conv1, kernel, 2)
        biases = _variable_on_cpu('biases', [36])
        conv2 = tf.nn.relu(conv + biases, name=scope.name)
        _activation_summary(conv2)

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 36, 48])
        conv = conv2d(conv2, kernel, 2)
        biases = _variable_on_cpu('biases', [48])
        conv3 = tf.nn.relu(conv + biases, name=scope.name)
        _activation_summary(conv3)

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 48, 64])
        conv = conv2d(conv3, kernel, 1)
        biases = _variable_on_cpu('biases', [64])
        conv4 = tf.nn.relu(conv + biases, name=scope.name)
        _activation_summary(conv4)

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64])
        conv = conv2d(conv4, kernel, 1)
        biases = _variable_on_cpu('biases', [64])
        conv5 = tf.nn.relu(conv + biases, name=scope.name)
        #conv5 = conv + biases
        #conv5 = tf.maximum(0.01*conv5, conv5)
        _activation_summary(conv5)

    if num_classes > 1:
        layers = [num_classes * 3, num_classes * 3, int(num_classes * 2)]
    else:
        layers = [100, 50, 10]

    # fully1
    with tf.variable_scope('fully1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        #reshape = tf.reshape(conv5, [conv5.get_shape().as_list()[0], -1])
        reshape = tf.reshape(conv5, [-1, 1152])
        #dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[1152, 1164])
        biases = _variable_on_cpu('biases', [1164])
        fully1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        fully1_drop = tf.nn.dropout(fully1, keep_prob)
        _activation_summary(fully1_drop)

    # fully2
    with tf.variable_scope('fully2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1164, layers[0]])
        biases = _variable_on_cpu('biases', [layers[0]])
        fully2 = tf.nn.relu(tf.matmul(fully1_drop, weights) + biases, name=scope.name)
        fully2_drop = tf.nn.dropout(fully2, keep_prob)
        _activation_summary(fully2_drop)

    # fully3
    with tf.variable_scope('fully3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[layers[0], layers[1]])
        biases = _variable_on_cpu('biases', [layers[1]])
        fully3 = tf.nn.relu(tf.matmul(fully2_drop, weights) + biases, name=scope.name)
        fully3_drop = tf.nn.dropout(fully3, keep_prob)
        _activation_summary(fully3_drop)

    # fully4
    with tf.variable_scope('fully4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[layers[1], layers[2]])
        biases = _variable_on_cpu('biases', [layers[2]])
        fully4 = tf.nn.relu(tf.matmul(fully3_drop, weights) + biases, name=scope.name)
        fully4_drop = tf.nn.dropout(fully4, keep_prob)
        _activation_summary(fully4_drop)

    # output
    with tf.variable_scope('output') as scope:
        weights = _variable_with_weight_decay('weights', [layers[2], num_classes])
        biases = _variable_on_cpu('biases', [num_classes])

        if num_classes > 1:
            output = tf.add(tf.matmul(fully4_drop, weights), biases, name=scope.name)
        else:
            output = tf.mul(tf.atan(tf.matmul(fully4_drop, weights) + biases), 2) #scale the atan output

        _activation_summary(output)

    return output

def loss(logits, labels, num_classes=1):

    if num_classes > 1:
        return loss_softmax(logits, labels)
    else:
        return loss_mse(logits, labels)

def loss_mse(output, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        output: Output from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """

    # The total loss is defined as the mean square error loss plus all of the weight
    # decay terms (L2 loss).
    mse = tf.reduce_mean(tf.square(tf.sub(output, labels)), name='mse')
    tf.add_to_collection('losses', mse)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss_softmax(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    '''num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)'''

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        #opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(1e-3)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
