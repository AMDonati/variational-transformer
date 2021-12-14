import tensorflow as tf
import time
from src.train.utils import CustomSchedule, get_checkpoints, loss_function, accuracy_function, write_to_tensorboard
from src.models.transformer import Transformer, VAETransformer

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.25):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else:
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L

def get_klweights(beta_schedule, n_cycle, n_iter):
    if beta_schedule == "linear":
        range_klweights = frange_cycle_linear(n_iter=n_iter, n_cycle=n_cycle)
    elif beta_schedule == "warmup":
        range_klweights = frange_cycle_zero_linear(n_iter=n_iter, n_cycle=n_cycle)
    return range_klweights

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]
#
#
# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar, transformer, optimizer, loss_object):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _, _ = transformer((inp, tar_inp),
                                        True)
        loss = loss_function(tar_real, predictions, loss_object)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    accuracy = accuracy_function(tar_real, predictions)

    return loss, accuracy


# kl_weights = tf.minimum(tf.to_float(self.global_step) / 20000, 1.0) # simple KL annealing schedule.
# global step = step for each update.
# if num_iters % args.cycle >= args.cycle - args.beta_warmup:
# beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)


def train_step_vae(inp, tar, transformer, optimizer, loss_object, kl_weights):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _, kl_loss = transformer((inp, tar_inp),
                                              True)
        ce_loss = loss_function(tar_real, predictions, loss_object)
        #kl_weights = tf.minimum(tf.cast(global_step, tf.float32) / 20000, 1.0)
        loss = ce_loss + kl_weights * kl_loss

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    accuracy = accuracy_function(tar_real, predictions)

    return (loss, ce_loss, kl_loss), accuracy, kl_weights


def eval_step(inp, tar, transformer, loss_object):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _, _ = transformer((inp, tar_inp),
                                    False)
    loss = loss_function(tar_real, predictions, loss_object)
    accuracy = accuracy_function(tar_real, predictions)
    return loss, accuracy


def eval_step_vae(inp, tar, transformer, loss_object, kl_weights):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _, kl_loss = transformer((inp, tar_inp),
                                          False)
    ce_loss = loss_function(tar_real, predictions, loss_object)
    #kl_weights = tf.minimum(tf.cast(global_step, tf.float32) / 20000, 1.0)
    loss = ce_loss + kl_weights * kl_loss
    accuracy = accuracy_function(tar_real, predictions)
    return (loss, ce_loss, kl_loss), accuracy


def train(EPOCHS, train_dataset, val_dataset, ckpt_manager, transformer, optimizer, loss_object, range_klweights=None, logger=None,
          train_writer=None, val_writer=None):
    metrics = dict.fromkeys(["train_loss", "val_loss", "train_accuracy", "val_accuracy"])
    for key in metrics.keys():
        metrics[key] = []
    for epoch in range(EPOCHS):
        start = time.time()

        loss_epoch, accuracy_epoch = 0., 0.
        val_loss_epoch, val_accuracy_epoch = 0., 0.

        for (batch, (inp, tar)) in enumerate(train_dataset):
            loss_batch, accuracy_batch = train_step(inp, tar, transformer, optimizer, loss_object)
            loss_epoch += loss_batch
            accuracy_epoch += accuracy_batch

        for key, val in zip(["train_loss", "train_accuracy"], [loss_epoch, accuracy_epoch]):
            metrics[key].append((val / (batch + 1)).numpy())

        for (batch_val, (inp, tar)) in enumerate(val_dataset):
            loss_batch, accuracy_batch = eval_step(inp, tar, transformer, loss_object)
            val_loss_epoch += loss_batch
            val_accuracy_epoch += accuracy_batch

        for key, val in zip(["val_loss", "val_accuracy"], [val_loss_epoch, val_accuracy_epoch]):
            metrics[key].append((val / (batch_val + 1)).numpy())

        ckpt_save_path = ckpt_manager.save()

        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        if logger is None:
            print('Epoch: {}'.format(epoch + 1))
            print('Train Loss: {:.4f}, Train Accuracy {:.4f}'.format(metrics["train_loss"][-1],
                                                                     metrics["train_accuracy"][-1]
                                                                     ))
            print('Val Loss: {:.4f}, Val Accuracy {:.4f}'.format(metrics["val_loss"][-1],
                                                                     metrics["val_accuracy"][-1]
                                                                     ))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        else:
            logger.info('-' * 40 + 'Epoch: {}'.format(epoch + 1) + '-' * 40)
            logger.info('Train Loss: {:.4f}, Train Accuracy {:.4f}'.format(metrics["train_loss"][-1],
                                                                     metrics["train_accuracy"][-1]
                                                                     ))
            logger.info('Val Loss: {:.4f}, Val Accuracy {:.4f}'.format(metrics["val_loss"][-1],
                                                                 metrics["val_accuracy"][-1]
                                                                 ))
            logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    return metrics


def train_VAE(EPOCHS, train_dataset, val_dataset, ckpt_manager, transformer, optimizer, loss_object, range_klweights, logger=None,
              train_writer=None, val_writer=None):
    metrics = dict.fromkeys(
        ["train_loss", "val_loss", "train_ce_loss", "val_ce_loss", "train_kl_loss", "val_kl_loss", "train_accuracy",
         "val_accuracy"])
    for key in metrics.keys():
        metrics[key] = []
    global_step = 0
    for epoch in range(EPOCHS):
        start = time.time()
        loss_epoch, ce_loss_epoch, kl_loss_epoch, accuracy_epoch = 0., 0., 0., 0.
        val_loss_epoch, val_ce_loss_epoch, val_kl_loss_epoch, val_accuracy_epoch = 0., 0., 0., 0.

        for (batch, (inp, tar)) in enumerate(train_dataset):
            kl_weights = tf.constant(range_klweights[global_step], dtype=tf.float32)
            (loss, ce_loss, kl_loss), accuracy_batch, kl_weights = train_step_vae(inp, tar, transformer, optimizer,
                                                                                  loss_object, kl_weights)
            # get learnable_query:
            learned_q = tf.squeeze(transformer.encoder.learnable_query[:, :, 0])
            if train_writer is not None:
                write_to_tensorboard(writer=train_writer, loss=loss, ce_loss=ce_loss, kl_loss=kl_loss,
                                     accuracy=accuracy_batch, kl_weights=kl_weights, global_step=global_step,
                                     learned_q=learned_q)
            loss_epoch += loss
            ce_loss_epoch += ce_loss
            kl_loss_epoch += kl_loss
            accuracy_epoch += accuracy_batch
            global_step += 1

        for key, val in zip(["train_loss", "train_ce_loss", "train_kl_loss", "train_accuracy"],
                            [loss_epoch, ce_loss_epoch, kl_loss_epoch,
                             accuracy_epoch]):
            metrics[key].append((val / (batch + 1)).numpy())

        for (batch_val, (inp, tar)) in enumerate(val_dataset):
            (val_loss, val_ce_loss, val_kl_loss), val_accuracy = eval_step_vae(inp, tar, transformer, loss_object,
                                                                               kl_weights)
            if val_writer is not None:
                write_to_tensorboard(writer=val_writer, loss=val_loss, ce_loss=val_ce_loss, kl_loss=val_kl_loss,
                                     accuracy=val_accuracy, kl_weights=None, global_step=global_step, learned_q=None)
            val_loss_epoch += val_loss
            val_ce_loss_epoch += val_ce_loss
            val_kl_loss_epoch += val_kl_loss
            val_accuracy_epoch += val_accuracy

        for key, val in zip(["val_loss", "val_ce_loss", "val_kl_loss", "val_accuracy"],
                            [val_loss_epoch, val_ce_loss_epoch, val_kl_loss_epoch,
                             val_accuracy_epoch]):
            metrics[key].append((val / (batch_val + 1)).numpy())

        ckpt_save_path = ckpt_manager.save()

        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        if logger is None:
            print('Epoch: {}'.format(epoch + 1))
            print('Train Loss: {:.4f}, Train CE loss: {:.4f}, Train Accuracy {:.4f}'.format(metrics["train_loss"][-1],
                                                                                            metrics["train_ce_loss"][
                                                                                                -1],
                                                                                            metrics["train_accuracy"][
                                                                                                -1]))
            print('Val Loss: {:.4f}, Val CE loss: {:.4f}, Val Accuracy {:.4f}'.format(metrics["val_loss"][-1],
                                                                                      metrics["val_ce_loss"][
                                                                                          -1],
                                                                                      metrics["val_accuracy"][
                                                                                          -1]))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        else:
            logger.info('-' * 40 + 'Epoch: {}'.format(epoch + 1) + '-' * 40)
            logger.info(
                'Train Loss: {:.4f}, Train CE loss: {:.4f}, Train Accuracy {:.4f}'.format(metrics["train_loss"][-1],
                                                                                          metrics["train_ce_loss"][
                                                                                              -1],
                                                                                          metrics["train_accuracy"][
                                                                                              -1]))
            logger.info('Val Loss: {:.4f}, Val CE loss: {:.4f}, Val Accuracy {:.4f}'.format(metrics["val_loss"][-1],
                                                                                            metrics["val_ce_loss"][
                                                                                                -1],
                                                                                            metrics["val_accuracy"][
                                                                                                -1]))
            logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    return metrics


if __name__ == '__main__':
    d_model = 32
    learning_rate = CustomSchedule(d_model)
    EPOCHS = 2
    batch_size = 16
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    transformer = Transformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=2500, target_vocab_size=2000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((200, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((200, 36), dtype=tf.int64, minval=0, maxval=200)

    tfdataset = tf.data.Dataset.from_tensor_slices((temp_input, temp_target))
    tfdataloader = tfdataset.batch(batch_size, drop_remainder=True)

    checkpoint_path = "output/temp/train/checkpoints"

    ckpt_manager = get_checkpoints(transformer, optimizer, checkpoint_path)

    results = train(EPOCHS=EPOCHS, train_dataset=tfdataloader, val_dataset=tfdataloader, ckpt_manager=ckpt_manager,
                    transformer=transformer, optimizer=optimizer, loss_object=loss_object)
    print("done for basic transformer")

    vae_transformer = VAETransformer(
        num_layers=2, d_model=32, num_heads=8, dff=128,
        input_vocab_size=2500, target_vocab_size=2000,
        pe_input=10000, pe_target=6000, latent="input")

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    checkpoint_path = "output/temp/train/checkpoints"

    ckpt_manager = get_checkpoints(vae_transformer, optimizer, checkpoint_path)

    vae_results = train_VAE(EPOCHS=EPOCHS, train_dataset=tfdataloader, val_dataset=tfdataloader,
                            ckpt_manager=ckpt_manager,
                            transformer=vae_transformer, optimizer=optimizer, loss_object=loss_object)

    print("done for VAE Transformer")
