import tensorflow as tf
from src.models.encoder import Encoder, VAEEncoder
from src.models.decoder import Decoder, VAEDecoder
from src.models.transformer_utils import create_look_ahead_mask, create_padding_mask, point_wise_feed_forward_network


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
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

        return final_output, attention_weights, 0., None  # 0. is for KL loss., None for (mean, logvar)

    def create_masks(self, inp, tar, size1=None, size2=None):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        if size1 is None or size2 is None:
            look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1], tf.shape(tar)[1])
        else:
            look_ahead_mask = create_look_ahead_mask(size1, size2)

        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

class VAETransformer(Transformer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(VAETransformer, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                             input_vocab_size=input_vocab_size,
                                             target_vocab_size=target_vocab_size, pe_input=pe_input,
                                             pe_target=pe_target, rate=rate)

        self.decoder = VAEDecoder(num_layers=num_layers, d_model=d_model, nul_heads=num_heads, dff=dff,
                               target_vocab_size=target_vocab_size, pe_target=pe_target, rate=rate)

    def compute_kl(self, prior_mean, prior_logvar, recog_mean, recog_logvar):
        kld = -0.5 * tf.math.reduce_sum(1 + (recog_logvar - prior_logvar)
                                        - tf.math.divide(tf.pow(prior_mean - recog_mean, 2), tf.exp(prior_logvar))
                                        - tf.math.divide(tf.exp(recog_logvar), tf.exp(prior_logvar)), axis=-1)
        return kld



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
            tar = tf.cast(tar, dtype=tf.int64)
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
                                                                                    tar_)
        enc_output, avg_attn_weights = self.encoder(encoder_input, training,
                                  enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # compute mean, logvar from prior and posterior
        recog_mean, recog_logvar = self.encode_posterior(enc_output)
        prior_mean, prior_logvar = self.encode_prior(enc_output)
        # compute kl
        kl = self.compute_kl(recog_mean, recog_logvar, recog_mean, recog_logvar)
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


if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=64, num_heads=8, dff=256,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, attn_weights, _, _ = sample_transformer((temp_input, temp_target), training=True)

    # test masks.
    temp_input = tf.constant([[1,2,3,4,0,0],[2,4,6,0,0,0],[2,8,10,12,6,7],[3,5,0,0,0,0]], dtype=tf.int32)
    temp_target = tf.constant([[6, 2, 3, 0, 0, 0], [2, 4, 6, 8, 9, 0], [2, 8, 3, 12, 0, 0], [3, 1, 0, 0, 0, 0]],
                             dtype=tf.int32)

    enc_padding_mask, look_ahead_mask, dec_padding_mask = sample_transformer.create_masks(temp_input, temp_target)
    #print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # -------------------------------------- VAE Transformer ----------------------------------------------------------------------------------

    sample_vae_transformer = VAETransformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000, latent="output", rate=0.0)

    vae_out, attn_weights, kl_loss, (mean, logvar) = sample_vae_transformer((temp_input, temp_target), training=True)
    print(kl_loss)
    print(vae_out.shape)

    vae_out, _, kl_loss, (mean, logvar) = sample_vae_transformer((temp_input, temp_target), training=False)
    print(kl_loss)
    print(vae_out.shape)
