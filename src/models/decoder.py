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


class VAEDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, latent="input"):
        super(VAEDecoderLayer, self).__init__()

        if latent == "attention":
            self.mha = PseudoSelfAttention(d_model, num_heads)
        else:
            self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, query, keys, values, training, mask, z=None):
        attn_output, attention_weights_block = self.mha(q=query, k=keys, v=values,  mask=mask, z=z)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(query + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2, attention_weights_block


class VAEDecoder(Decoder):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, latent="attention"):
        super(VAEDecoder, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, target_vocab_size=target_vocab_size,
                 maximum_position_encoding=maximum_position_encoding, rate=0.1)

        self.dec_layers = [VAEDecoderLayer(d_model, num_heads, dff, rate, latent=latent)
                           for _ in range(num_layers)]
        self.latent = latent
        if self.latent == "input":
            self.input_proj = tf.keras.layers.Dense(d_model, use_bias=False)
        if self.latent == "attention":
            self.attn_proj = tf.keras.layers.Dense(d_model*num_layers, use_bias=False)

    def pseudo_self_attention(self, x, z):
        # x: shape (batch_size, tar_seq_len, d_model)
        # z: shape (batch_size, d_model)
        out = tf.concat([z, x], axis=1)
        return out

    def call(self, x, z, training,
             look_ahead_mask):
        #TODO: careful the look_ahead mask should be adaptated to eventually add one timestep for the attention latent.
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        if self.latent == "input":
            x = x + self.input_proj(z)

        x = self.dropout(x, training=training)

        if self.latent == "attention":
            z = tf.split(self.attn_proj(z), num_or_size_splits=self.num_layers, axis=-1)

        for i in range(self.num_layers):
            if self.latent == "attention":
                z_i = z[i]
                x, block = self.dec_layers[i](query=x, keys=x, values=x, z=z_i, training=training, mask=look_ahead_mask)

            else:
                x, block = self.dec_layers[i](query=x, keys=x, values=x, training=training, mask=look_ahead_mask)

            attention_weights['decoder_layer{}'.format(i + 1)] = block

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights