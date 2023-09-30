from schedules import get_warmup_schedule
from dataloader import prepare_dataset
import tensorflow as tf
from tensorflow.keras import optimizers
from models import *
from utils import *
from losses import *
from tqdm import tqdm
from summary import summary_logger


def train(
    model,
    train_dataset,
    val_dataset,
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

            for x, y, c in train_dataset:
                model.train_step(x, y, c)
                tr_loss = model.metrics["tr_loss"].result().numpy()
                bar.set_postfix_str(
                    "train_loss={:.3f}".format(tr_loss))
                bar.update()

            for x, y, c in val_dataset:
                model.val_step(x, y, c)
            val_loss = model.metrics["vl_loss"].result().numpy()

            logger.write(tr_loss, val_loss, i)

            epoch_end_str = (
                "Epoch({}): lr={:.5f}, tr_iou={:.4f}, tr_rec={:.5f}, tr_pre={:.5f}, tr_loss={:.5f}, vl_iou={:.5f}, vl_rec={:.5f}, vl_pre={:.5f}, vl_loss={:.5f}".format(
                    str(i + 1).zfill(3),
                    model.optim.learning_rate.numpy(),
                    model.metrics["tr_iou"].result().numpy(),
                    model.metrics["tr_rec"].result().numpy(),
                    model.metrics["tr_pre"].result().numpy(),
                    tr_loss,
                    model.metrics["vl_iou"].result().numpy(),
                    model.metrics["vl_rec"].result().numpy(),
                    model.metrics["vl_pre"].result().numpy(),
                    val_loss
                )
            )
            if val_loss < best_val:
                epoch_end_str = (
                    epoch_end_str +
                    f" improved from {best_val:.5f}. Saving Model!"
                )
                best_val = val_loss
                model.model.save(model_checkpoint_dir)

            bar.set_postfix_str(epoch_end_str)
            for k in model.metrics.keys():
                model.metrics[k].reset_state()


def __main__():
    TFRecord_filenames = [
        "./tfrecords/1.tfrec",
        "./tfrecords/2.tfrec",
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
        buffer_size=4,
    )

    vl_ds = prepare_dataset(
        TFRecord_filenames[-1],
        batch_size=1,
        patch_size=swin_args["input_shape"][:-1],
        buffer_size=0,
    )
    logger = summary_logger()
    scheduled_lrs = get_warmup_schedule(**train_args)
    optim = optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=1e-4)

    # model = XNet(conv_loss=WCE, swin_loss=DICE, **swin_args)
    model = XNet_v2(optimizer=optim, cls_loss=BCE,
                    swin_loss=Dice_WCE, **swin_args)

    train(
        model,
        tr_ds,
        vl_ds,
        epochs=train_args["epochs"],
        logger=logger,
        model_checkpoint_dir="./model.keras",
    )


if __name__ == "__main__":
    __main__()
