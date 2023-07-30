import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_slices(vol, figsize=10, filename=None):
    num_slices = vol.shape[2]
    N = np.int32(np.sqrt(num_slices)) + 1
    fig = plt.figure(figsize=(figsize, figsize))
    for i in range(num_slices):
        plt.subplot(N, N, i + 1)
        # plt.title(i)
        plt.axis("off")
        plt.imshow(vol[:, :, i], "gray")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if filename is not None:
        plt.savefig(f"{filename}.jpg")
    plt.show()


@tf.function
def patch_extract_gather(x, indices, patch_dims):
    x_patches = tf.extract_volume_patches(
        x, ksizes=[1, *patch_dims, 1], strides=[1, *patch_dims, 1], padding="VALID"
    )  # B, nH, nW, nD, patch_length
    # B, nH*nW*nD, patch_length
    x_patches = tf.reshape(x_patches, (tf.shape(x)[0], -1, np.prod(patch_dims)))
    x_patches = tf.gather(x_patches, indices, axis=1)
    return x_patches
