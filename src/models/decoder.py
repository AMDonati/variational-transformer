import tensorflow as tf
from src.models.transformer_utils import positional_encoding, point_wise_feed_forward_network
from src.models.self_attention import MultiHeadAttention, PseudoSelfAttention


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(q=out1, k=enc_output, v=enc_output, mask=padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class VAEDecoderLayer(DecoderLayer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(VAEDecoderLayer, self).__init__(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        if training:
            look_ahead_mask_ = None
        else:
            look_ahead_mask_ = look_ahead_mask

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask_)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(q=out1, k=enc_output, v=enc_output, mask=padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2



class VAEDecoder(Decoder):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(VAEDecoder, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, target_vocab_size=target_vocab_size,
                 maximum_position_encoding=maximum_position_encoding, rate=0.1)

        self.dec_layers = [VAEDecoderLayer(d_model, num_heads, dff, rate)] + [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers - 1)]

        self.prior_net = tf.keras.layers.Dense(2 * d_model, name='prior_net')
        self.posterior_net = tf.keras.layers.Dense(2 * d_model, name='posterior_net')

    def encode_prior(self, x):
        mean, logvar = tf.split(self.prior_net(x), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def encode_posterior(self, x):
        mean, logvar = tf.split(self.posterior_net(x), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # variationnal decoder layer:
        x, block1, block2 = self.dec_layers[0](x, enc_output, training,
                                               look_ahead_mask, padding_mask)
        attention_weights['decoder_layer1_block1'] = block1
        attention_weights['decoder_layer1_block2'] = block2

        # deriving the latent variable z from x:
        # compute mean, logvar from prior and posterior
        recog_mean, recog_logvar = self.encode_posterior(x)
        prior_mean, prior_logvar = self.encode_prior(x)

        # derive latent z from mean and logvar
        mean = recog_mean if training else prior_mean
        logvar = recog_logvar if training else prior_logvar
        z = self.reparameterize(mean, logvar)

        for i in range(1, self.num_layers):
            z, block1, block2 = self.dec_layers[i](z, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return z, attention_weights, (recog_mean, recog_logvar), (prior_mean, prior_logvar)