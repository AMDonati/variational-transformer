import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask, non_linearity=False, scale=True, temperature=0.5, subsize=10):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    q_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(q), tf.int32))
    if q_nan > 0:
        print("INITIAL queries are NAN")
    v_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(v), tf.int32))
    if v_nan > 0:
        print("INITIAL values are NAN")
    k_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(k), tf.int32))
    if k_nan > 0:
        print("INITIAL keys are NAN")


    scaled_attention_logits = compute_attention_logits(q=q, k=k, v=v, mask=mask, non_linearity=non_linearity, scale=scale)
   # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.

    # check if predictions have nan numbers
    pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(scaled_attention_logits), tf.int32))
    if pred_nan > 0:
        print("ATTENTION_LOGITS are NAN in NORMAL ATTENTION")

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def compute_attention_logits(q, k, v, mask,non_linearity=False, scale=True):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      attention_logits
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    if scale:
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        attention_logits = matmul_qk / tf.math.sqrt(dk)
    else:
        attention_logits = matmul_qk

    if non_linearity:
        attention_logits = tf.keras.activations.gelu(attention_logits)

    # add the mask to the scaled tensor.
    if mask is not None:
        attention_logits += (mask * -1e9) # shape (batch_size, 1, seq_len_k)

    return attention_logits

def stochastic_self_attention(q, k, v, mask, subsize=10, temperature=0.1, non_linearity=False, scale=True):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    # compute initial attention logits
    # check if predictions have nan numbers
    q_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(q), tf.int32))
    if q_nan > 0:
        print("queries are NAN")
    v_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(v), tf.int32))
    if v_nan > 0:
        print("values are NAN")
    k_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(k), tf.int32))
    if k_nan > 0:
        print("keys are NAN")
    attention_logits = compute_attention_logits(q,k,v,mask, non_linearity=non_linearity, scale=scale)

    # check if predictions have nan numbers
    pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(attention_logits), tf.int32))
    if pred_nan > 0:
        print("ATTENTION_LOGITS are NAN")

    # subsample keys based on the attention weights. # This outputs "relaxed one-hot vectors".
    relaxed_subk = compute_relaxed_subsampling(attention_logits, subsize=subsize, temperature=temperature)

    # check if predictions have nan numbers
    # pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(relaxed_subk), tf.int32))
    # if pred_nan > 0:
    #     print("RELAXED SUB KEYS are NAN")

    # get the "relaxed" subset of keys.
    sub_keys = k * relaxed_subk

    # check if predictions have nan numbers
    # pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(sub_keys), tf.int32))
    # if pred_nan > 0:
    #     print("SUB KEYS are NAN")

    # compute the new attention logits:
    stochastic_attention_logits = compute_attention_logits(q=q, k=sub_keys, v=v, mask=mask, non_linearity=non_linearity, scale=scale)

    # check if predictions have nan numbers
    # pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(stochastic_attention_logits), tf.int32))
    # if pred_nan > 0:
    #     print("stochastic_attention_logits are NAN")

    attention_weights = tf.nn.softmax(stochastic_attention_logits, axis=-1)  # (batch_size, 1, seq_len_k)
    # check if predictions have nan numbers
    # pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(attention_weights), tf.int32))
    # if pred_nan > 0:
    #     print("attention_weights are NAN")
    initial_attention_weights = tf.nn.softmax(attention_logits, axis=-1) # TO DEBUG

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights, initial_attention_weights

def compute_relaxed_subsampling(attention_logits, subsize=10, temperature=0.5):
    gumbels = - tf.math.log(-tf.math.log(tf.random.uniform(shape=attention_logits.shape, maxval=1)))
    r_keys = gumbels + attention_logits
    relaxed_topks = relaxed_topk(r_keys, temperature=temperature, subsize=subsize)
    return relaxed_topks

def relaxed_topk(r_keys, subsize=10, temperature=0.5):
    alpha = tf.squeeze(r_keys, axis=-2)
    relaxed_topks = [tf.nn.softmax(alpha/temperature)] # initialization
    #print("-" * 50)
    pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(relaxed_topks[0]), tf.int32))
    # if pred_nan > 0:
    #     print("INITIAL alpha are NAN")
    #     print("INIT ALPHA MIN", tf.reduce_min(alpha))
    #     print("INIT ALPHA MAX", tf.reduce_max(alpha))
    #print("-"*30)
    for j in range(subsize-1):
        alpha = alpha + tf.math.log(1-tf.nn.softmax(alpha/temperature))
        #print("MIN ALPHA:{}, MAX ALPHA:{}".format(tf.reduce_min(alpha), tf.reduce_max(alpha)))
        #pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(alpha), tf.int32))
        #if pred_nan > 0:
            #print("alpha are NAN")
        relaxed_topks.append(tf.nn.softmax(alpha/temperature))
        #print("index: {}, RELAXED TOP KEYS: {}".format(j, relaxed_topks))
    print("-" * 30)
    relaxed_topks = tf.stack(relaxed_topks, axis=0) # shape (k, batch_size, 1, num_timesteps)
    out = tf.reduce_sum(relaxed_topks, axis=0) # shape (batch_size, 1, num_timesteps)
    return out[:,:,:,tf.newaxis]

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, non_linearity=False, scale=True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        self.non_linearity = non_linearity # for average attention: use a gelu activation
        self.scale = scale # for average attention: no scaling.
        self.attention_fn = scaled_dot_product_attention

        self.subsize = None # dummy variable for d_VAE transformer.

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, z=None, temperature=0.5):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # check if predictions have nan numbers
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        q_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(q), tf.int32))
        if q_nan > 0:
            print("queries are NAN")
        v_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(v), tf.int32))
        if v_nan > 0:
            print("values are NAN")
        k_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(k), tf.int32))
        if k_nan > 0:
            print("keys are NAN")
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        out = self.attention_fn(
            q, k, v, mask, non_linearity=self.non_linearity, scale=self.scale, temperature=temperature, subsize=self.subsize)

        if len(out) == 2:
            # for Baselines
            (scaled_attention, attention_weights) = out
        elif len(out) == 3:
            # for d_VAE Transformer
            (scaled_attention, attention_weights, initial_attention_weights) = out
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        if len(out) == 2:
            return output, attention_weights
        elif len(out) == 3:
            return output, attention_weights, initial_attention_weights

class Stochastic_MHA(MultiHeadAttention):
    '''has same call function than MultiHeadAttention'''
    def __init__(self, d_model, num_heads, non_linearity=True, scale=False, subsize=10):
        super(Stochastic_MHA, self).__init__(d_model=d_model, num_heads=num_heads, non_linearity=non_linearity, scale=scale)
        self.subsize = subsize
        self.attention_fn = stochastic_self_attention

class PseudoSelfAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, non_linearity=False, scale=True):
        super(PseudoSelfAttention, self).__init__(d_model=d_model, num_heads=num_heads, non_linearity=non_linearity, scale=scale)

        self.wz = tf.keras.layers.Dense(2*d_model)

    def call(self, q, k, v, z, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # adding key_z and value_z from z:
        key_z, value_z = tf.split(self.wz(z), num_or_size_splits=2, axis=-1)
        key_z = self.split_heads(key_z, batch_size)
        value_z = self.split_heads(value_z, batch_size)

        k = tf.concat([key_z, k], axis=-2)
        v = tf.concat([value_z, v], axis=-2)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, non_linearity=self.non_linearity, scale=self.scale)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # TEST RELAXED_SUBSAMPLING:
    logits_weights = tf.math.log(tf.constant([0.1, 0.2, 0.3, 0.4]))
    subsets = []
    temperature = 10
    for i in range(500):
        relaxed_top2 = compute_relaxed_subsampling(logits_weights, subsize=2, temperature=temperature)
        val, subset = tf.math.top_k(relaxed_top2, k=2)
        subset = sorted(list(subset.numpy() + 1))
        subset_str = ';'.join([str(s) for s in subset])
        subsets.append(subset_str)
    plt.hist(subsets)
    plt.savefig("test_subset_sampling_temp{}.png".format(temperature))
    print("done")