from schedules import get_warmup_schedule
from dataloader import prepare_dataset
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from models import *
from utils import *
from losses import *
from tqdm import tqdm
from summary import summary_logger



TFRecord_filenames = [
    "./tfrecords/tsdc_abus_0.tfrec",
    "./tfrecords/tsdc_abus_1.tfrec",
    "./tfrecords/tsdc_abus_2.tfrec",
    "./tfrecords/tsdc_abus_3.tfrec",
    "./tfrecords/tsdc_abus_4.tfrec",
]

train_args = {
    "learning_rate": 4e-4,
    "len_data": 80,
    "batch_size": 4,
    "epochs": 100,
    "warmup_epoch_percentage": 0.1,
}
swin_args = {
    "input_shape": (64, 64, 64, 1),
    "embed_dim": 48,
    "window_size": [7, 7, 7],
    "patch_size": [4, 4, 4],
    "mask_ratio": 0.01,
    "depths": [2, 2, 2, 2],
    "mlp_ratio": 4.0,
    "num_heads": [3, 6, 12, 24],
    "patch_norm": True,
    "qkv_bias": True,
    "drop_path": 0.5,
    "attn_drop": 0.0,
    "proj_drop": 0.0,
}

tr_ds = prepare_dataset(
    TFRecord_filenames[:-1],
    batch_size=train_args["batch_size"],
    patch_size=swin_args["input_shape"][:-1],
)
vl_ds = prepare_dataset(
    TFRecord_filenames[-1], batch_size=1, patch_size=swin_args["input_shape"][:-1]
)

logger = summary_logger()
swin_loss = DICE
conv_loss = WCE

model = get_XNet(**swin_args)
scheduled_lrs = get_warmup_schedule(**train_args)
optim = optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=1e-4)

@tf.function
def train_step(x,y):
    with tf.GradientTape() as tape:
        conv_out, swin_out = model(x)
        l1 = swin_loss(y, swin_out)
        l2 = conv_loss(y, conv_out)
        tr_loss = l1 + 3.0 * l2
    trainable_vars = model.trainable_variables
    grads = tape.gradient(tr_loss, trainable_vars)
    optim.apply_gradients(zip(grads, trainable_vars))
    return tr_loss


@tf.function
def val_step(x,y):
    conv_out, swin_out = model(x)
    l1 = swin_loss(y, swin_out)
    l2 = conv_loss(y, conv_out)
    val_loss = l1 + 3.0 * l2
    return val_loss

def train(
    model,
    train_dataset,
    val_dataset,
    optimizer,
    epochs,
    logger,
    model_checkpoint_dir="None",
):
    #     if os.path.exists(model_checkpoint_dir):
    #         model = models.load_model(model_checkpoint_dir)
    #         print("Loaded From %s"%model_checkpoint_dir)
    best_val = np.inf
    for i in range(epochs):
        with tqdm(
            bar_format="{desc}|{percentage:3.0f}%| {elapsed} |{n_fmt}/{total_fmt}|{postfix}"
        ) as bar:
            train_losses, val_losses = [], []
            for x, y in train_dataset:
                tr_loss = train_step(x,y)
                train_losses.append(tr_loss)
                bar.set_postfix_str("train_loss={:.3f}".format(tr_loss))
                bar.update()
            
            for x, y in val_dataset:
                val_loss = val_step(x,y)
                val_losses.append(val_loss)
            
            tr_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            logger.write(tr_loss, val_loss, i)

            epoch_end_str = (
                "Epoch({}): lr={:.5f}, tr_loss={:.5f}, val_loss={:.5f}".format(
                    str(i + 1).zfill(3),
                    optimizer.learning_rate.numpy(),
                    tr_loss,
                    val_loss,
                )
            )
            if val_loss < best_val:
                epoch_end_str = (
                    epoch_end_str + f" improved from {best_val:.5f}. Saving Model!"
                )
                best_val = val_loss
                model.save(model_checkpoint_dir)

            bar.set_postfix_str(epoch_end_str)


# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# with strategy.scope():

# print(model.summary())

train(
    model,
    tr_ds,
    vl_ds,
    optim,
    epochs=train_args["epochs"],
    logger=logger,
    model_checkpoint_dir="./fold1_xnet_w7_p4_e48.keras",
)
