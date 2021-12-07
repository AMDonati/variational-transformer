import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model, dim=3):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    if dim == 3:
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq, num_particles):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    temp = seq[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
    return tf.tile(temp, multiples=[1, num_particles, 1, 1, 1])  # (batch_size, num_particles 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])



if __name__ == "__main__":

    # --- test of positional encoding -------------------------------------------------------------------------------------------------------------------
    b = 8
    S = 20
    pe_target = 10
    d_model = 64
    inputs = tf.random.uniform(shape=(b, S, d_model))
    pos_enc = positional_encoding(position=pe_target, d_model=d_model)