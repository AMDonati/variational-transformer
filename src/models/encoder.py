import tensorflow as tf
from src.models.self_attention import MultiHeadAttention
from src.models.transformer_utils import point_wise_feed_forward_network, positional_encoding


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x,
                                  mask)  # (batch_size, input_seq_len, d_model) # In the encoder, this is a padding mask !

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class VAEEncoder(Encoder):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, simple_average=False):
        super(VAEEncoder, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                         input_vocab_size=input_vocab_size,
                                         maximum_position_encoding=maximum_position_encoding, rate=rate)
        self.simple_average = simple_average
        if not self.simple_average:
            self.average_attention = MultiHeadAttention(d_model, num_heads=1, non_linearity=True, scale=False) # in previous work, has a gelu non linearity.
            self.learnable_query = tf.Variable(initial_value=tf.random.normal(shape=(1, 1, d_model), stddev=0.02), name="learnable_query")
        #nn.init.normal_(w, std=0.02): in Transformer VAE: init with a random normal.

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        if self.simple_average:
            out = tf.reduce_mean(x, axis=1, keepdims=True)
            attn_weights = None
        else:
            average_query = tf.tile(self.learnable_query, multiples=[x.shape[0], 1, 1]) # shape (B, 1, d_model)
            out, attn_weights = self.average_attention(q=average_query, k=x, v=x, mask=mask)

        return out, attn_weights# (batch_size, 1, d_model)
