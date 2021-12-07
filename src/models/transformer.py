import tensorflow as tf
from src.models.encoder import Encoder, VAEEncoder
from src.models.decoder import Decoder, VAEDecoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


class VAETransformer(Transformer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, latent="attention"):
        super(VAETransformer, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, input_vocab_size=input_vocab_size,
                 target_vocab_size=target_vocab_size, pe_input=pe_input, pe_target=pe_target, rate=rate)

        self.encoder = VAEEncoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = VAEDecoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate, latent=latent)

        self.prior_net = tf.keras.layers.Dense(2*d_model)
        self.posterior_net = tf.keras.layers.Dense(2*d_model)
        #self.combination_layer = tf.keras.layers.Dense(d_model)

    def encode_prior(self, x):
        mean, logvar = tf.split(self.prior_net(x), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def encode_posterior(self, x):
        mean, logvar = tf.split(self.posterior_net(x), num_or_size_splits=2, axis=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def combination_layer(self, x, z):
        # x: shape of (batch_size, tar_seq_len, d_model)
        # z: shape of (batch size, d_model)
        out_ = self.final_layer(x)
        p_z = self.final_layer(z)
        p_z = tf.tile(p_z, multiples=[1,out_.shape[1], 1])
        return out_ + p_z

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask):

        if training:
            encoder_input = tf.concat([inp, tar], axis=1)
        else:
            encoder_input = inp

        enc_output = self.encoder(encoder_input, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model) #TODO: do we need an enc_padding mask ?

        mean, logvar = self.encode_posterior(enc_output) if training else self.encode_prior(enc_output)
        z = self.reparameterize(mean, logvar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, z, training, look_ahead_mask)

        if self.decoder.latent == "output":
            final_output = self.combination_layer(dec_output, z)
        else:
            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=64, num_heads=8, dff=256,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # -------------------------------------- VAE Transformer ----------------------------------------------------------------------------------

    sample_vae_transformer = VAETransformer(
        num_layers=2, d_model=64, num_heads=8, dff=256,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000, latent="input")

    vae_out, _ = sample_vae_transformer(temp_input, temp_target, training=True,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None)
    print(vae_out.shape)