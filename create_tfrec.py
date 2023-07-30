import numpy as np
import nrrd
import tensorflow as tf
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from volumentations import *
import pandas as pd


# Helper Functions
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def image_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def create_example(volume, mask, type, shape):
    feature = {
        "volume": image_feature(tf.io.serialize_tensor(volume)),
        "mask": image_feature(tf.io.serialize_tensor(mask)),
        "shape": _int64_feature(shape),
        "type": _int64_feature(type),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def normalize(volume, only_on_non_zero=True):
    v = np.copy(volume)
    if only_on_non_zero:
        v = volume[volume > 0]
    volume = (volume - np.mean(v)) / np.std(v)
    return np.float32(volume)


def scale(volume):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return np.float32(volume)


img_nrrd_root = "./data/images"
msk_nrrd_root = "./data/masks"
table = pd.read_csv("./data/labels.csv")
labels = np.where(table["label"] == "M", 1, 0)


num_tfrecords = 5
Normalize = True

margin_x, margin_y, margin_d = (
    256,
    128,
    128,
)  # margin around tumor to extract subvolumes
indeces = np.array_split(np.arange(100), num_tfrecords)  # we have 100 vols
options = tf.io.TFRecordOptions(compression_type="GZIP")

for i in range(num_tfrecords):
    tfrecord_filename = f"./tfrecords/tsdc_abus_{i}.tfrec"

    with tf.io.TFRecordWriter(tfrecord_filename, options=options) as writer:
        for j in tqdm(indeces[i]):
            img_path = "{}/DATA_{}.nrrd".format(img_nrrd_root, str(j).zfill(3))
            msk_path = "{}/MASK_{}.nrrd".format(msk_nrrd_root, str(j).zfill(3))

            vol, header = nrrd.read(img_path)
            vol = scale(vol)
            #             if Normalize:
            #                 vol = normalize(vol, only_on_non_zero=True)
            msk, _ = nrrd.read(msk_path)  # header same as vol header
            msk = scale(msk)
            #             print(np.max(vol),np.min(vol),np.max(msk),np.min(msk))
            shape = header["sizes"]
            idxs = np.argwhere(msk > 0)
            lH, lW, lD = np.min(idxs, 0)
            hH, hW, hD = np.max(idxs, 0)
            lH, lW, lD = (
                max(lH - margin_x, 0),
                max(lW - margin_y, 0),
                max(lD - margin_d, 0),
            )
            hH, hW, hD = (
                min(hH + margin_x, msk.shape[0]),
                min(hW + margin_y, msk.shape[1]),
                min(hD + margin_d, msk.shape[2]),
            )
            serialized_example = create_example(
                vol[lH:hH, lW:hW, lD:hD], msk[lH:hH, lW:hW, lD:hD], [labels[j]], shape
            )
            writer.write(serialized_example)
            del serialized_example
            gc.collect()
            # if j > 4:
            #     break
    # break
