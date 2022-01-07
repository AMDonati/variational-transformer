import tensorflow as tf
import os
import numpy as np


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def get_checkpoints(transformer, optimizer, checkpoint_path):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return ckpt_manager

def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=-1))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def write_to_tensorboard(writer, loss, ce_loss, kl_loss, accuracy, kl_weights, logvar, global_step, learned_q=None, ce_loss_posterior=None, logvar_posterior=None):
    with writer.as_default():
        tf.summary.scalar("loss", loss, step=global_step)
        tf.summary.scalar("ce_loss", ce_loss, step=global_step)
        tf.summary.scalar("kl_loss", kl_loss, step=global_step)
        tf.summary.scalar("accuracy", accuracy, step=global_step)
        tf.summary.scalar("var", logvar, step=global_step)
        if kl_weights is not None:
            tf.summary.scalar("kl_weight", kl_weights, step=global_step)
        if learned_q is not None:
            tf.summary.scalar("learnable_query_dim0", learned_q, step=global_step)
        if ce_loss_posterior is not None:
            tf.summary.scalar("ce_loss_posterior", ce_loss_posterior, step=global_step)
        if logvar_posterior is not None:
            tf.summary.scalar("var_posterior", logvar_posterior, step=global_step)

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.25):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else:
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L

def get_klweights(beta_schedule, n_cycle, n_iter, beta_stop=1.):
    if beta_schedule == "linear":
        range_klweights = frange_cycle_linear(n_iter=n_iter, n_cycle=n_cycle, stop=beta_stop)
    elif beta_schedule == "warmup":
        range_klweights = frange_cycle_zero_linear(n_iter=n_iter, n_cycle=n_cycle, stop=beta_stop)
    elif beta_schedule == "None":
        range_klweights = [0.] * n_iter
    return range_klweights



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d_model = 32
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    temp_learning_rate_schedule = CustomSchedule(d_model)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
