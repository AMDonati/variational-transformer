import tensorflow as tf
from src.models.encoder import Encoder, VAEEncoder
from src.models.decoder import Decoder, VAEDecoder
from src.models.transformer_utils import create_look_ahead_mask, create_padding_mask
from src.models.vae_utils import gaussian_kld


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, latent="attention"):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights, 0.  # 0. is for KL loss.

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


class VAETransformer(Transformer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, latent="input"):
        super(VAETransformer, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                             input_vocab_size=input_vocab_size,
                                             target_vocab_size=target_vocab_size, pe_input=pe_input,
                                             pe_target=pe_target, rate=rate, latent=latent)

        self.encoder = VAEEncoder(num_layers, d_model, num_heads, dff,
                                  input_vocab_size, pe_input, rate)

        self.decoder = VAEDecoder(num_layers, d_model, num_heads, dff,
                                  target_vocab_size, pe_target, rate, latent=latent)

        self.prior_net = tf.keras.layers.Dense(2 * d_model, name='prior_net')
        self.posterior_net = tf.keras.layers.Dense(2 * d_model, name='posterior_net')
        # self.combination_layer = tf.keras.layers.Dense(d_model)

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
        return out_ + p_z

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
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(encoder_input,
                                                                                tar)  # TODO: problem here for latent = "attention". The decoding padding mask should include one more timestep for the latent.

        enc_output = self.encoder(encoder_input, training,
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
            final_output = self.combination_layer(dec_output, z)
        else:
            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights, kl


if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=64, num_heads=8, dff=256,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _, _ = sample_transformer((temp_input, temp_target), training=True)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # -------------------------------------- VAE Transformer ----------------------------------------------------------------------------------

    sample_vae_transformer = VAETransformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000, latent="output")

    vae_out, _, kl_loss = sample_vae_transformer((temp_input, temp_target), training=True)
    print(kl_loss)
    print(vae_out.shape)
