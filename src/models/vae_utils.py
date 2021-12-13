import tensorflow as tf
import numpy as np


@tf.function
def sample(self, eps=None):
    if eps is None:
        eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)


def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar


def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.math.reduce_sum(1 + (recog_logvar - prior_logvar)
                                    - tf.math.divide(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                                    - tf.math.divide(tf.exp(recog_logvar), tf.exp(prior_logvar)), axis=1)
    return kld


# for latent = output:
# class LM_head_rep(nn.Module):
#     def __init__(self, in_dim=768, out_dim=50257):
#         super().__init__()
#
#         self.Nu_fc1 = nn.Linear(in_dim, 1024)
#         self.Nu_fc2 = nn.Linear(1024, out_dim)
#
#     def forward(self, z):
#         z = F.leaky_relu(self.Nu_fc1(z))
#         z = self.Nu_fc2(z)
#         return z
