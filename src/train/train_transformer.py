import tensorflow as tf
import time
from src.train.utils import CustomSchedule, get_checkpoints, loss_function, accuracy_function
from src.models.transformer import Transformer

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
        predictions, _ = transformer((inp, tar_inp),
                                     True)
        loss = loss_function(tar_real, predictions, loss_object)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    accuracy = accuracy_function(tar_real, predictions)

    return loss, accuracy

def eval_step(inp, tar, transformer, loss_object):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _ = transformer((inp, tar_inp),
                                     False)
    loss = loss_function(tar_real, predictions, loss_object)
    accuracy = accuracy_function(tar_real, predictions)
    return loss, accuracy


def train(EPOCHS, train_dataset, val_dataset, ckpt_manager, transformer, optimizer, loss_object, logger=None):
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

        for (batch, (inp, tar)) in enumerate(val_dataset):
            loss_batch, accuracy_batch = eval_step(inp, tar, transformer, loss_object)
            val_loss_epoch += loss_batch
            val_accuracy_epoch += accuracy_batch

        ckpt_save_path = ckpt_manager.save()

        for key, val in zip(metrics.keys(), [loss_epoch, val_loss_epoch, accuracy_epoch, val_accuracy_epoch]):
            metrics[key].append(val/(batch+1))

        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        if logger is None:
            print('Epoch: {}'.format(epoch + 1))
            print('Train Loss: {:.4f}, Train Accuracy {:.4f}'.format(loss_epoch / (batch + 1), accuracy_epoch/(batch+1)))
            print(
                'Val Loss: {:.4f}, Val Accuracy {:.4f}'.format(val_loss_epoch / (batch + 1), val_accuracy_epoch / (batch + 1)))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        else:
            logger.info('-' * 40 + 'Epoch: {}'.format(epoch + 1) + '-' * 40)
            logger.info('Train Loss: {:.4f}, Train Accuracy {:.4f}'.format(loss_epoch / (batch + 1),
                                                                     accuracy_epoch / (batch + 1)))
            logger.info(
                'Val Loss: {:.4f}, Val Accuracy {:.4f}'.format(val_loss_epoch / (batch + 1),
                                                               val_accuracy_epoch / (batch + 1)))
            logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    return metrics


if __name__ == '__main__':
    d_model = 32
    learning_rate = CustomSchedule(d_model)
    EPOCHS = 10
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

    results = train(EPOCHS=EPOCHS, train_dataset=tfdataloader, val_dataset=tfdataloader, ckpt_manager=ckpt_manager, transformer=transformer, optimizer=optimizer, loss_object=loss_object)
    print("done")