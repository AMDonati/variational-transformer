## Code Blocks for implementing the VAE Transformer
### Encoder
```python
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
```
### Decoder
```python
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

    def call(self, x, z, training,
             look_ahead_mask):
             
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
```
### VAE Transformer
```python
class VAETransformer(Transformer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, latent="input", simple_average=False):
        super(VAETransformer, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                             input_vocab_size=input_vocab_size,
                                             target_vocab_size=target_vocab_size, pe_input=pe_input,
                                             pe_target=pe_target, rate=rate, latent=latent)

        self.encoder = VAEEncoder(num_layers, d_model, num_heads, dff,
                                  input_vocab_size, pe_input, rate, simple_average=simple_average)

        self.decoder = VAEDecoder(num_layers, d_model, num_heads, dff,
                                  target_vocab_size, pe_target, rate, latent=latent)

        self.prior_net = tf.keras.layers.Dense(2 * d_model, name='prior_net')
        self.posterior_net = tf.keras.layers.Dense(2 * d_model, name='posterior_net')
        self.simple_average = simple_average
        # self.combination_layer = tf.keras.layers.Dense(d_model)

        if self.decoder.latent == "output":
            self.output_proj = tf.keras.Sequential([
                tf.keras.layers.Dense(dff),  # (batch_size, seq_len, dff)
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(target_vocab_size)  # (batch_size, seq_len, d_model)
            ])

    def encode_prior(self, x):
        mean, logvar = tf.split(self.prior_net(x), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def encode_posterior(self, x):
        mean, logvar = tf.split(self.posterior_net(x), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def compute_kl(self, prior_mean, prior_logvar, recog_mean, recog_logvar):
        kld = -0.5 * tf.math.reduce_sum(1 + (recog_logvar - prior_logvar)
                                        - tf.math.divide(tf.pow(prior_mean - recog_mean, 2), tf.exp(prior_logvar))
                                        - tf.math.divide(tf.exp(recog_logvar), tf.exp(prior_logvar)), axis=-1)
        return tf.reduce_mean(kld)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        if training:
            encoder_input = tf.concat([inp, tar], axis=1)
        else:
            encoder_input = inp

        if self.decoder.latent == "attention":
            # add a dummy timestep to tar to create masks with the right length for pseudo self-attention:
            tar_ = tf.concat([tf.ones_like(tf.expand_dims(tar[:, 0], axis=1)), tar], axis=1)
            enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(encoder_input,
                                                                                    tar_, size1=tar.shape[1],
                                                                                    size2=tar_.shape[1])
        else:
            tar_ = tar
            enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(encoder_input,
                                                                                    tar_)  # TODO: problem here for latent = "attention". The decoding padding mask should include one more timestep for the latent.

        enc_output, avg_attn_weights = self.encoder(encoder_input, training,
                                  enc_padding_mask)  # (batch_size, inp_seq_len, d_model) #TODO: do we need an enc_padding mask ?

        # compute mean, logvar from prior and posterior
        recog_mean, recog_logvar = self.encode_posterior(enc_output)
        prior_mean, prior_logvar = self.encode_prior(enc_output)
        # compute kl
        kl = self.compute_kl(prior_mean, prior_logvar, recog_mean, recog_logvar)
        # derive latent z from mean and logvar
        mean = recog_mean if training else prior_mean
        logvar = recog_logvar if training else prior_logvar
        z = self.reparameterize(mean, logvar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, z, training, look_ahead_mask)

        if self.decoder.latent == "output":
            z = self.output_proj(z)
            final_output = self.final_layer(dec_output) + z
        else:
            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, avg_attn_weights, kl, (mean, logvar)
 ```
 
 ### Pseudo Self-attention (used in the decoder to inject the latent variable z as an additionnal key-value pair in each self-attention block)
 ```python
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
 ```
