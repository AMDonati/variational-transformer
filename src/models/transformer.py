import tensorflow as tf
from src.models.encoder import Encoder, VAEEncoder, d_VAEEncoder
from src.models.decoder import Decoder, VAEDecoder
from src.models.transformer_utils import create_look_ahead_mask, create_padding_mask, point_wise_feed_forward_network
import tensorflow_probability as tfp


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, latent="attention", simple_average=False, **kwargs):
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
                 target_vocab_size, pe_input, pe_target, rate=0.1, latent="input", simple_average=False, **kwargs):
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

    def compute_vae_loss(self, recog_mean, recog_logvar, **kwargs):
        prior_mean = kwargs["prior_mean"]
        prior_logvar = kwargs["prior_logvar"]
        kld = -0.5 * tf.math.reduce_sum(1 + (recog_logvar - prior_logvar)
                                        - tf.math.divide(tf.pow(prior_mean - recog_mean, 2), tf.exp(prior_logvar))
                                        - tf.math.divide(tf.exp(recog_logvar), tf.exp(prior_logvar)), axis=-1)
        return tf.reduce_mean(kld)

    def create_masks(self, inp, tar, size1=None, size2=None):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        if size1 is None or size2 is None:
            look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1], tf.shape(tar)[1])
        else:
            look_ahead_mask = create_look_ahead_mask(size1, size2)

        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask

    def encode(self, encoder_input, targets, training):
        if self.decoder.latent == "attention":
            # add a dummy timestep to tar to create masks with the right length for pseudo self-attention:
            tar_ = tf.concat([tf.ones_like(tf.expand_dims(targets[:, 0], axis=1)), targets], axis=1)
            enc_padding_mask, look_ahead_mask = self.create_masks(encoder_input,
                                                                  tar_, size1=targets.shape[1],
                                                                  size2=tar_.shape[1])
        else:
            tar_ = targets
            enc_padding_mask, look_ahead_mask = self.create_masks(encoder_input,
                                                                  tar_)
        enc_output, avg_attn_weights = self.encoder(encoder_input, training,
                                                    enc_padding_mask)  #
        return enc_output, avg_attn_weights, look_ahead_mask

    def call(self, inputs, training, temperature=0.5):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs
        tar = tf.cast(tar, dtype=inp.dtype)
        posterior_input = tf.concat([inp, tar], axis=1)
        prior_input = inp

        posterior_output, post_attn_weights, look_ahead_mask = self.encode(encoder_input=posterior_input, targets=tar,
                                                                           training=training)
        # check if predictions have nan numbers
        pred_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(posterior_output), tf.int32))
        if pred_nan > 0:
            print("ENCODER outputs are NAN")
        prior_output, prior_attn_weights, look_ahead_mask2 = self.encode(
            encoder_input=prior_input, targets=tar, training=training)

        # compute mean, logvar from prior and posterior
        recog_mean, recog_logvar = self.encode_posterior(posterior_output)  # shape (batch_size, 1, d_model)
        prior_mean, prior_logvar = self.encode_prior(prior_output)

        # derive latent z from mean and logvar
        mean = recog_mean if training else prior_mean
        logvar = recog_logvar if training else prior_logvar
        z = self.reparameterize(mean, logvar)
        avg_attn_weights = post_attn_weights if training else prior_attn_weights

        # compute vae_loss (without the cross-entropy part)
        kl = self.compute_vae_loss(inputs=inputs, z=z, recog_mean=recog_mean, recog_logvar=recog_logvar,
                                   prior_mean=prior_mean, prior_logvar=prior_logvar, temperature=temperature)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, z, training, look_ahead_mask)

        if self.decoder.latent == "output":
            z = self.output_proj(z)
            final_output = self.final_layer(dec_output) + z
        else:
            final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, avg_attn_weights, kl, (mean, logvar)



class d_VAETransformer(VAETransformer):
    '''has same call function than the VAETransformer. Only the VAE loss and the Encoder model change.'''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, latent="attention", simple_average=False, **kwargs):
        super(d_VAETransformer, self).__init__(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                               input_vocab_size=input_vocab_size,
                                               target_vocab_size=target_vocab_size, pe_input=pe_input,
                                               pe_target=pe_target, rate=rate, latent=latent, simple_average=False)

        self.encoder = d_VAEEncoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                                    input_vocab_size=input_vocab_size, maximum_position_encoding=pe_input, rate=rate,
                                    subsize=tf.cast(kwargs["subsize"], tf.int32), simple_average=False)
        self.samples_loss = kwargs["samples_loss"]
        self.debug_loss = kwargs["debug_loss"]

    def compute_vae_loss_(self, recog_mean, recog_logvar, **kwargs):
        # get the M samples from the prior distrib.
        z = kwargs["z"]
        temperature = kwargs["temperature"]
        inp, tar = kwargs["inputs"]
        enc_padding_mask = create_padding_mask(inp)
        prior_probs = []
        for m in range(self.samples_loss):
            # draw \alphas using the prior path, and get encoder_output from alphas.
            enc_output, _ = self.encoder(inp, training=True,
                                         mask=enc_padding_mask, temperature=temperature)
            # get their mean_m and logvariance_m
            _, (mean_m, logvar_m) = self.get_z_from_encoder_output(enc_output,
                                                                   training=False)  # shape (batch_size, 1, d_model)
            # compute gaussian densities from z, mean_m, logvar_m
            std_m = tf.exp(logvar_m * 0.5)  # shape (batch_size, 1, d_model)
            gauss_distrib_m = tfp.distributions.MultivariateNormalDiag(loc=mean_m, scale_diag=std_m)
            prob_m = gauss_distrib_m.prob(z)  # shape (batch_size, 1)
            prior_probs.append(prob_m)
        sum_prior_density = tf.reduce_sum(tf.stack(prior_probs, axis=0), axis=0)  # shape (batch_size, 1)
        # sample from posterior one more time (here K=2)
        tar = tf.cast(tar, dtype=inp.dtype)
        encoder_input = tf.concat([inp, tar], axis=1)
        enc_padding_mask = create_padding_mask(encoder_input)
        enc_output_posterior, _ = self.encoder(encoder_input, training=True, mask=enc_padding_mask,
                                               temperature=temperature)
        # get mean and logvar
        _, (mean_posterior, logvar_posterior) = self.get_z_from_encoder_output(enc_output_posterior, training=True)
        print("MEAN POSTERIOR - MAX", tf.reduce_max(mean_posterior))
        print("-"*30)
        std_posterior = tf.exp(logvar_posterior * 0.5)
        gauss_distrib = tfp.distributions.MultivariateNormalDiag(loc=mean_posterior, scale_diag=std_posterior)
        prob_posterior = gauss_distrib.prob(z)  # shape (batch_size, 1)
        # compute initial gaussian density from z, recog_mean, recog_logvar:
        recog_std = tf.exp(recog_logvar * 0.5)
        gauss_distrib = tfp.distributions.MultivariateNormalDiag(loc=recog_mean, scale_diag=recog_std)
        print("RECOG MEAN - MAX", tf.reduce_max(recog_mean)) #TODO: RECOG_MEAN EXPLOSES... equal to 12 which leads to 10^36 in sum_posterior density !
        print("-" * 30)
        recog_prob = gauss_distrib.prob(z)  # shape (batch_size, 1)
        # sum-up all posterior densities:
        #sum_posterior_density = recog_prob + prob_posterior
        sum_posterior_density = recog_prob

        # return log [(K+1/M) sum_prior_densities] - log sum_posterior_densities
        #print("numerator KL LOSS -min ", tf.reduce_min(sum_prior_density))
        #print("numerator KL LOSS -max", tf.reduce_max(sum_prior_density))
        #print("denominateur KL LOSS - min", tf.reduce_min(sum_posterior_density))
        print("denominateur KL LOSS - max", tf.reduce_max(sum_posterior_density)) #TODO: BUG explose -> 10^31 -> KL infinity !
        print("-" * 30)
        log_loss = tf.math.log((1 / self.samples_loss) * sum_prior_density + 1e-9) - tf.math.log(sum_posterior_density +1e-9)
        return tf.reduce_mean(log_loss)


    def compute_vae_loss_debug(self, recog_mean, recog_logvar, **kwargs):
        prior_mean = kwargs["prior_mean"]
        prior_logvar = kwargs["prior_logvar"]
        # print("RECOG MEAN - MAX", tf.reduce_max(
        #     recog_mean)) #TODO: here ok. stays inferior to one...
        # print("-" * 30)
        # print("PRIOR MEAN - MAX", tf.reduce_max(
        #     prior_mean))
        # print("-" * 30)
        kld = -0.5 * tf.math.reduce_sum(1 + (recog_logvar - prior_logvar)
                                        - tf.math.divide(tf.pow(prior_mean - recog_mean, 2), tf.exp(prior_logvar))
                                        - tf.math.divide(tf.exp(recog_logvar), tf.exp(prior_logvar)), axis=-1)
        return tf.reduce_mean(kld)

    def compute_vae_loss(self, recog_mean, recog_logvar, **kwargs):
        if self.debug_loss:
            return self.compute_vae_loss_debug(recog_mean, recog_logvar, **kwargs)
        else:
            return self.compute_vae_loss_(recog_mean, recog_logvar, **kwargs)


    def get_z_from_encoder_output(self, enc_output, training):
        if training:
            mean, logvar = self.encode_posterior(enc_output)
        else:
            mean, logvar = self.encode_prior(enc_output)
        z = self.reparameterize(mean, logvar)
        return z, (mean, logvar)


if __name__ == '__main__':
    sample_transformer = Transformer(
        num_layers=2, d_model=64, num_heads=8, dff=256,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, attn_weights, _, _ = sample_transformer((temp_input, temp_target), training=True)

    # test masks.
    temp_input = tf.constant([[1, 2, 3, 4, 0, 0], [2, 4, 6, 0, 0, 0], [2, 8, 10, 12, 6, 7], [3, 5, 0, 0, 0, 0]],
                             dtype=tf.int32)
    temp_target = tf.constant([[6, 2, 3, 0, 0, 0], [2, 4, 6, 8, 9, 0], [2, 8, 3, 12, 0, 0], [3, 1, 0, 0, 0, 0]],
                              dtype=tf.int32)

    enc_padding_mask, look_ahead_mask, dec_padding_mask = sample_transformer.create_masks(temp_input, temp_target)
    # print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # -------------------------------------- VAE Transformer ----------------------------------------------------------------------------------
    print("-----------------------------------Testing VAE Transformer-------------------------------------------------")
    sample_vae_transformer = VAETransformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000, latent="output", rate=0.0)

    vae_out, attn_weights, kl_loss, (mean, logvar) = sample_vae_transformer((temp_input, temp_target), training=True)
    #print(kl_loss)
    print(vae_out.shape)

    vae_out, _, kl_loss, (mean, logvar) = sample_vae_transformer((temp_input, temp_target), training=False)
    #print(kl_loss)
    #print(vae_out.shape)

    # -------------------------------------------- test of d_VAE Transformer --------------------------------------#
    print("---------------------------------------------- Testing d_VAE Transformer ---------------------------------")
    d_vae_transformer = d_VAETransformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000, latent="attention", rate=0.0, debug_loss=1, subsize=10, samples_loss=1)

    vae_out, attn_weights, kl_loss, (mean, logvar) = d_vae_transformer((temp_input, temp_target), training=True)
    print("KL Loss", kl_loss)
    print(vae_out.shape)

    # debug_loss:
    d_vae_transformer_2 = d_VAETransformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000, latent="attention", rate=0.0, debug_loss=0, samples_loss=1, subsize=10)

    vae_out, _, kl_loss, (mean, logvar) = d_vae_transformer_2((temp_input, temp_target), training=True)
    print("KL LOSS", kl_loss)
    print(vae_out.shape)


    vae_out, _, kl_loss, (mean, logvar) = d_vae_transformer((temp_input, temp_target), training=False)
    print(kl_loss)
    print(vae_out.shape)

