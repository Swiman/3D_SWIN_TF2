import keras.layers as KL
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations  # pylint: disable
import tensorflow_addons as tfa
import tensorflow_probability as tfp

import logging
logger = logging.getLogger("tensorflow-addons")
logger.setLevel(logging.ERROR)

tf.keras.saving.get_custom_objects().clear()


@tf.function
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -
                   1)  # (H or W or D, C/6,2)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb


@tf.function
def SinPosEncoding3D(input_shape, embed_dim, dtype=tf.float32):
    channels = int(np.ceil(embed_dim / 6) * 2)
    if channels % 2:
        channels += 1
    inv_freq = np.float32(
        1 / np.power(10000, np.arange(0, channels, 2) / np.float32(channels))
    )

    b, x, y, z, org_channels = input_shape

    pos_x = tf.range(x, dtype=dtype)
    pos_y = tf.range(y, dtype=dtype)
    pos_z = tf.range(z, dtype=dtype)

    sin_inp_x = tf.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = tf.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = tf.einsum("i,j->ij", pos_z, inv_freq)

    emb_x = tf.expand_dims(tf.expand_dims(get_emb(sin_inp_x), 1), 1)
    emb_y = tf.expand_dims(tf.expand_dims(get_emb(sin_inp_y), 1), 0)
    emb_z = tf.expand_dims(tf.expand_dims(get_emb(sin_inp_z), 0), 0)

    emb_x = tf.tile(emb_x, (1, y, z, 1))
    emb_y = tf.tile(emb_y, (x, 1, z, 1))
    emb_z = tf.tile(emb_z, (x, y, 1, 1))

    emb = tf.concat((emb_x, emb_y, emb_z), -1)
    penc = tf.repeat(emb[None, :, :, :, :org_channels], b, axis=0)
    return penc


@tf.keras.saving.register_keras_serializable()
class DropPath(layers.Layer):
    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.keep_prob = 1 - drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        rank = x.shape.rank
        shape = (input_shape[0],) + (1,) * (rank - 1)

        # random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        # out = tf.floor(random_tensor)
        # scaled_out = tf.math.divide(x, 1 - self.drop_prob) * out

        out = tfp.distributions.Bernoulli(
            probs=(self.keep_prob), dtype=tf.float32
        ).sample(shape)
        scaled_out = tf.math.divide(x, self.keep_prob) * out
        return scaled_out


@tf.keras.saving.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    # An embedding layer without positional encoding (with conv)
    def __init__(self, patch_size, embed_dim, use_norm, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Conv3D(
            filters=embed_dim, kernel_size=patch_size, strides=patch_size
        )
        if use_norm:
            self.norm = layers.LayerNormalization()
        else:
            self.norm = layers.Identity()

    def call(self, x):
        p = self.norm(self.proj(x))
        return p


@tf.keras.saving.register_keras_serializable()
class Masked_PatchEmbedding(layers.Layer):
    # An embedding layer with positional encoding and masking capablity
    def __init__(self, patch_size, mask_ratio, embed_dim, use_norm, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        # later used for patch extraction with tf
        self.patch_dims = [1, *patch_size, 1]
        self.proj = layers.Dense(embed_dim)
        if use_norm:
            self.norm = layers.LayerNormalization()
        #             self.norm = tfa.layers.InstanceNormalization()
        else:
            self.norm = layers.Identity()

    def build(self, input_shape):
        _, H, W, D, C = input_shape
        self.nH, self.nW, self.nD = (
            H // self.patch_size[0],
            W // self.patch_size[1],
            D // self.patch_size[2],
        )
        self.token_num = self.nH * self.nW * self.nD
        self.mask_num = int(self.token_num * self.mask_ratio)

        self.pos_embeddings = SinPosEncoding3D(
            input_shape=(1, self.nH, self.nW, self.nD, self.embed_dim),
            embed_dim=self.embed_dim,
        )
        self.pos_embeddings = tf.reshape(
            self.pos_embeddings, (1, self.token_num, self.embed_dim)
        )

        self.mask_token = tf.Variable(
            tf.random.normal(
                shape=(1, self.mask_num, np.prod(self.patch_dims))),
            trainable=True,
            name="mask_token",
        )

    def get_random_indeces(self):
        indeces = tf.argsort(tf.random.normal(shape=(self.token_num,)))
        return indeces[: self.mask_num], indeces[self.mask_num:]

    def call(self, x):
        B, H, W, D, C = x.shape
        patches = tf.extract_volume_patches(
            x, ksizes=self.patch_dims, strides=self.patch_dims, padding="VALID"
        )  # B, nH, nW, nD, patch_length
        patches_flat = tf.reshape(
            patches, (-1, self.token_num, tf.shape(patches)[-1]))
        mask_indeces, unmask_indeces = self.get_random_indeces()

        unmask_tokens = tf.gather(patches_flat, unmask_indeces, axis=1)
        unmask_pencs = tf.gather(self.pos_embeddings, unmask_indeces, axis=1)
        mask_pencs = tf.gather(self.pos_embeddings, mask_indeces, axis=1)

        a = self.proj(unmask_tokens) + \
            tf.repeat(unmask_pencs, tf.shape(x)[0], axis=0)
        b = tf.repeat(self.mask_token, tf.shape(x)[0], axis=0)
        b = self.proj(b) + tf.repeat(mask_pencs, tf.shape(x)[0], axis=0)
        out = tf.concat([a, b], axis=1)
        out = tf.reshape(
            out, (tf.shape(x)[0], self.nH, self.nW, self.nD, self.embed_dim)
        )
        out = self.norm(out)
        return out, mask_indeces


@tf.keras.saving.register_keras_serializable()
class PatchMerging(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.proj = layers.Dense(embed_dim * 2)
        self.norm = layers.LayerNormalization()

    def build(self, input_shape):
        self.p0 = input_shape[1] % 2
        self.p1 = input_shape[2] % 2
        self.p2 = input_shape[3] % 2

    def call(self, x):
        # B H W D C(embed_dim)
        x = tf.pad(x, [[0, 0], [0, self.p0], [
                   0, self.p1], [0, self.p2], [0, 0]])
        c = x.shape[-1]
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8C
        out = self.norm(self.proj(x))  # B H/2 W/2 D/2 2C
        return out


@tf.keras.saving.register_keras_serializable()
class Double_CIR(layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, padding, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
        )
        self.norm_1 = tfa.layers.InstanceNormalization()
        self.act_1 = layers.LeakyReLU()
        self.conv_2 = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
        )
        self.norm_2 = tfa.layers.InstanceNormalization()
        self.act_2 = layers.LeakyReLU()

    def call(self, x):
        out = self.act_1(self.norm_1(self.conv_1(x)))
        out = self.act_2(self.norm_2(self.conv_2(out)))
        return out


# @tf.function


@tf.keras.saving.register_keras_serializable()
class CIR(layers.Layer):
    def __init__(self, filters, kernel_size, strides, use_bias, padding, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
        )
        self.norm_1 = tfa.layers.InstanceNormalization()
        self.act_1 = layers.LeakyReLU()

    def call(self, x):
        out = self.act_1(self.norm_1(self.conv_1(x)))
        return out


def window_partition(x, window_size):
    B, H, W, D, C = x.shape
    w0, w1, w2 = window_size
    windows = tf.transpose(
        tf.reshape(x, (-1, H // w0, w0, W // w1, w1, D // w2, w2, C)),
        (0, 1, 3, 5, 2, 4, 6, 7),
    )
    windows = tf.reshape(windows, (-1, w0 * w1 * w2, C))
    return windows


# @tf.function


def window_reverse(x, dims, window_size):
    B, H, W, D, C = dims
    w0, w1, w2 = window_size
    x = tf.reshape(x, (-1, H // w0, W // w1, D // w2, w0, w1, w2, C))
    x = tf.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))
    x = tf.reshape(x, (-1, H, W, D, C))
    return x


# @tf.function


def compute_mask(dims, window_size, shift_size):
    coutn = 0
    H, W, D = dims
    count = 0
    img_mask = np.zeros((1, H, W, D, 1), dtype=np.float32)
    for h in [
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ]:
        for w in [
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ]:
            for d in [
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ]:
                img_mask[:, h, w, d, :] = count
                count += 1
    window_mask = window_partition(
        img_mask, window_size)  # num_windows,wH*wW*wD,1
    # num_windows,wH*wW*wD,wH*wW*wD
    window_mask = window_mask[:, None, :, 0] - window_mask[:, :, None, 0]
    window_mask = tf.where(window_mask == 0, 0.0, window_mask)
    window_mask = tf.where(window_mask != 0, -100.0, window_mask)
    return tf.Variable(initial_value=window_mask, trainable=False, name="window_mask")


def get_window_shift_size(input_shape, window_size, shift_size):
    stage_w, stage_s = np.copy(window_size), np.copy(shift_size)
    for j in range(3):
        if input_shape[j] < window_size[j]:
            stage_w[j] = input_shape[j]
            stage_s[j] = 0
    return stage_w, stage_s


@tf.keras.saving.register_keras_serializable()
class window_attention(layers.Layer):
    def __init__(
        self,
        embed_dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = layers.Dense(3 * embed_dim, use_bias=qkv_bias)
        self.proj = layers.Dense(embed_dim)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj_drop = layers.Dropout(proj_drop)
        self.softmax = layers.Softmax(axis=-1)

    def build(self, input_shape):
        shape = (
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1)
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(shape, self.num_heads),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_d = np.arange(self.window_size[2])
        coords = np.stack(np.meshgrid(
            coords_h, coords_w, coords_d, indexing="ij"))
        flat_coords = coords.reshape(3, -1)
        relative_coords = flat_coords[:, :, None] - flat_coords[:, None, :]
        relative_coords = np.transpose(relative_coords, (1, 2, 0))
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
            2 * self.window_size[2] - 1
        )
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = tf.convert_to_tensor(relative_coords.sum(-1))
        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index, trainable=False, name="rpi"
        )

    def call(self, x, mask=None):
        B, N, C = x.shape  # (batchsize*numwindows,w0w1w2,embed_dim)
        qkv = self.qkv(x)  # (batchsize*numwindows,w0w1w2,3*embed_dim)
        qkv = tf.transpose(
            tf.reshape(qkv, (-1, N, 3, self.num_heads,
                       self.head_dim)), (2, 0, 3, 1, 4)
        )
        # each one:(batchsize*numwindows,num_heads,w0w1w2,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        # (batchsize*numwindows,num_heads,w0w1w2,w0w1w2)
        attn = q @ tf.transpose(k, (0, 1, 3, 2))
        flat_relative_position_index = tf.reshape(
            self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, flat_relative_position_index
        )

        relative_position_bias = tf.reshape(relative_position_bias, (N, N, -1))
        relative_position_bias = tf.expand_dims(
            tf.transpose(relative_position_bias, (2, 0, 1)), axis=0
        )

        attn += relative_position_bias
        if mask is not None:
            nw = mask.shape[0]
            # 1, num_windows,1,wH*wW*wD,wH*wW*wD
            mask = tf.expand_dims(mask[:, tf.newaxis, :, :], axis=0)
            # B, num_windows,num_heads,wH*wW*wD,wH*wW*wD
            attn = tf.reshape(attn, shape=(-1, nw, self.num_heads, N, N))
            attn = attn + tf.cast(mask, tf.float32)
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
        attn = self.attn_drop(self.softmax(attn))
        # -1, w0w1w2, num_head, head_dim
        qkv = tf.transpose((attn @ v), (0, 2, 1, 3))
        qkv = tf.reshape(qkv, (-1, N, C))  # -1, w0w1w2, num_head*head_dim
        out = self.proj_drop(self.proj(qkv))  # -1, w0w1w2, embed_dim
        return out


@tf.keras.saving.register_keras_serializable()
class MLP(layers.Layer):
    def __init__(self, embed_dim, mlp_ratio, proj_drop, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(mlp_ratio * embed_dim)
        self.a1 = layers.Activation(activations.gelu)
        self.d1 = layers.Dropout(proj_drop)
        self.fc2 = layers.Dense(embed_dim)
        self.d2 = layers.Dropout(proj_drop)

    def call(self, x):
        o = self.d1(self.a1(self.fc1(x)))
        o = self.d2(self.fc2(o))
        return o


@tf.keras.saving.register_keras_serializable()
class SwinTransformerBlock(layers.Layer):
    def __init__(
        self,
        embed_dim,
        window_size,
        shift_size,
        mlp_ratio,
        num_heads,
        qkv_bias=True,
        drop_path=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.w_attn = window_attention(
            embed_dim, window_size, num_heads, qkv_bias, attn_drop, proj_drop
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp_ratio = mlp_ratio
        self.mlp = MLP(embed_dim=embed_dim,
                       mlp_ratio=mlp_ratio, proj_drop=proj_drop)

    def build(self, input_shape):
        self.window_size, self.shift_size = get_window_shift_size(
            input_shape[1:], self.window_size, self.shift_size
        )
        self.p0 = (
            self.window_size[0] - input_shape[1] % self.window_size[0]
        ) % self.window_size[0]
        self.p1 = (
            self.window_size[1] - input_shape[2] % self.window_size[1]
        ) % self.window_size[1]
        self.p2 = (
            self.window_size[2] - input_shape[3] % self.window_size[2]
        ) % self.window_size[2]
        self.new_input_shape = (
            input_shape[0],
            input_shape[1] + self.p0,
            input_shape[2] + self.p1,
            input_shape[3] + self.p2,
            input_shape[4],
        )

    def call(self, x, mask):
        B, H, W, D, C = x.shape
        x_norm = self.norm1(x)

        if min(self.p0, self.p1, self.p2) > 0:
            x_norm = tf.pad(
                x_norm, [[0, 0], [0, self.p0], [
                    0, self.p1], [0, self.p2], [0, 0]]
            )

        x_norm_shifted = x_norm
        attn_mask = None

        if min(self.shift_size) > 0:
            x_norm_shifted = tf.roll(
                x_norm,
                shift=[-self.shift_size[0], -
                       self.shift_size[1], -self.shift_size[2]],
                axis=(1, 2, 3),
            )
            attn_mask = mask

        x_windows = window_partition(
            x_norm_shifted, self.window_size)  # *(-1,w0w1w2,C)

        attn_windows = self.w_attn(x_windows, attn_mask)

        attn_windows = tf.reshape(attn_windows, (-1, *self.window_size, C))
        shifted_x = window_reverse(
            attn_windows, self.new_input_shape, self.window_size)

        if min(self.shift_size) > 0:
            shifted_x = tf.roll(
                shifted_x,
                shift=[self.shift_size[0],
                       self.shift_size[1], self.shift_size[2]],
                axis=(1, 2, 3),
            )
        if min(self.p0, self.p1, self.p2) > 0:
            shifted_x = shifted_x[:, :H, :W, :D, :]
        out1 = x + self.drop_path(shifted_x)
        out2 = self.drop_path(self.mlp(self.norm2(out1)))
        out = out1 + out2
        return out


@tf.keras.saving.register_keras_serializable()
class SwinTransformerStage(layers.Layer):
    def __init__(
        self,
        depth,
        embed_dim,
        window_size,
        mlp_ratio,
        num_heads,
        qkv_bias=True,
        drop_path=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path = drop_path
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.down = PatchMerging(embed_dim)

    def build(self, input_shape):
        _, H, W, D, _ = input_shape
        self.window_size, self.shift_size = get_window_shift_size(
            input_shape[1:], self.window_size, self.shift_size
        )

        pH = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        pW = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        pD = int(np.ceil(D / self.window_size[2])) * self.window_size[2]

        self.attention_mask = compute_mask(
            [pH, pW, pD], self.window_size, self.shift_size
        )
        self.e1 = SwinTransformerBlock(
            embed_dim=self.embed_dim,
            window_size=self.window_size,
            shift_size=self.shift_size,
            mlp_ratio=self.mlp_ratio,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            drop_path=self.drop_path,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            name=f"{self.name}_{0}",
        )
        self.e2 = SwinTransformerBlock(
            embed_dim=self.embed_dim,
            window_size=self.window_size,
            shift_size=[0, 0, 0],
            mlp_ratio=self.mlp_ratio,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            drop_path=self.drop_path,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            name=f"{self.name}_{1}",
        )

    def call(self, x):
        x = self.e1(x, self.attention_mask)
        x = self.e2(x, self.attention_mask)
        out = self.down(x)
        return out


@tf.keras.saving.register_keras_serializable()
class T_CIR(layers.Layer):
    def __init__(self, num_f, ks, st, use_norm, time_dist, mode="deform_conv", **kwargs):
        super().__init__(**kwargs)
        use_bias = False if use_norm else True

        self.timedist_conv = DCNv2(num_f, ks) if mode == "deform_conv" else layers.Conv2D(
            num_f, kernel_size=ks, strides=st, padding='SAME', kernel_initializer="glorot_normal", use_bias=use_bias)

        if time_dist:
            self.timedist_conv = layers.TimeDistributed(self.timedist_conv)
        self.norm = tfa.layers.InstanceNormalization() if use_norm else layers.Identity()
        self.act = layers.ReLU()

    def call(self, x):
        o = self.norm(self.timedist_conv(x))
        return self.act(o)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:04:53 2020
@author: hu
"""


class DCN_v1(layers.Conv2D):
    """Only support "channel last" data format"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.

        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if num_deformable_group is None:
            num_deformable_group = filters
        if filters % num_deformable_group != 0:
            raise ValueError(
                '"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # kernel_shape = self.kernel_size + (input_dim, self.filters)
        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        # create offset conv layer
        offset_num = self.kernel_size[0] * \
            self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            # 2 means x and y axis
            shape=self.kernel_size + (input_dim, offset_num * 2),
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        # get offset, shape [batch_size, out_h, out_w, filter_h, * filter_w * channel_out * 2]
        offset = tf.nn.conv2d(inputs,
                              filters=self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        # some length
        batch_size = inputs.get_shape()[0]
        channel_in = int(inputs.get_shape()[-1])
        in_h, in_w = [int(i) for i in inputs.get_shape()[
            1: 3]]  # input feature map size
        out_h, out_w = [int(i) for i in offset.get_shape()[
            1: 3]]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, (-1, out_h, out_w, 9, 2))
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [-1, 1, 1, 1, self.num_deformable_group])
                for i in [y, x]]
        y, x = [tf.reshape(i, [-1, *i.shape[1: 3], -1]) for i in [y, x]]
        y, x = [tf.cast(i, tf.float32) for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, in_h - 1)
        x = tf.clip_by_value(x, 0, in_w - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), tf.int32) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DCN_v1._get_pixel_values_at_point(
            inputs, i) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.cast(i, tf.float32) for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [
                            -1, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [-1, out_h * filter_h,
                            out_w * filter_w, self.num_deformable_group, channel_in])

        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(
            pixels, [-1, out_h * filter_h, out_w * filter_w, -1])

        # depth-wise conv
        out = tf.nn.depthwise_conv2d(
            pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')
        # add the output feature maps in the same group
        out = tf.reshape(
            out, [-1, out_h, out_w, self.filters, channel_in])
        out = tf.reduce_sum(out, axis=-1)
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.

        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.get_shape().as_list()[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + \
                    (filter_size - 1) * (dilation - 1)
                same_output = (
                    in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (
                    in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           # top, bottom padding
                           [padding_list[0], padding_list[1]],
                           # left, right padding
                           [padding_list[2], padding_list[3]],
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map

        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [int(i) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1])
                for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(i,
                                         [1, *self.kernel_size, 1],
                                         [1, *self.strides, 1],
                                         [1, *self.dilation_rate, 1],
                                         'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values

        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.get_shape().as_list()[0: 4]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)


class DCNv2(KL.Layer):
    def __init__(self, filters,
                 kernel_size,
                 # stride,
                 # padding,
                 # dilation = 1,
                 # deformable_groups = 1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):

        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1, 1, 1)
        # self.padding = padding
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        super(DCNv2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_size + (int(input_shape[-1]), self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype='float32',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
            )

        # [kh, kw, ic, 3 * groups * kh, kw]--->3 * groups * kh * kw = oc [output channels]
        self.offset_kernel = self.add_weight(
            name='offset_kernel',
            shape=self.kernel_size +
            (input_shape[-1], 3 * self.deformable_groups *
             self.kernel_size[0] * self.kernel_size[1]),
            initializer='zeros',
            trainable=True,
            dtype='float32')

        self.offset_bias = self.add_weight(
            name='offset_bias',
            shape=(
                3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable=True,
            dtype='float32',
        )
        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (
            self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype='int32')
        self.patch_yx = tf.stack(tf.meshgrid(
            tf.range(-self.phw[1], self.phw[1] + 1), tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis=-1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])
        super(DCNv2, self).build(input_shape)

    def call(self, x):
        # x: [B, H, W, C]
        # offset: [B, H, W, ic] convx [kh, kw, ic, 3 * groups * kh * kw] ---> [B, H, W, 3 * groups * kh * kw]
        offset = tf.nn.conv2d(x, self.offset_kernel,
                              strides=self.stride, padding='SAME')
        offset += self.offset_bias
        bs, ih, iw, ic = x.shape.as_list()

        bs = tf.shape(x)[0]
        # [B, H, W, 18], [B, H, W, 9]
        oyox, mask = offset[..., :2*self.ks], offset[..., 2*self.ks:]
        mask = tf.nn.sigmoid(mask)
        # [H, W, 2]
        grid_yx = tf.stack(tf.meshgrid(
            tf.range(iw), tf.range(ih))[::-1], axis=-1)
        # [1, H, W, 9, 2]
        grid_yx = tf.reshape(
            grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        # [B, H, W, 9, 2]
        grid_yx = tf.cast(grid_yx, 'float32') + \
            tf.reshape(oyox, [bs, ih, iw, -1, 2])
        grid_iy0ix0 = tf.floor(grid_yx)
        grid_iy1ix1 = tf.clip_by_value(
            grid_iy0ix0 + 1, 0, tf.constant([ih+1, iw+1], dtype='float32'))
        # [B, H, W, 9, 1] * 2
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis=4)
        grid_iy0ix0 = tf.clip_by_value(
            grid_iy0ix0, 0, tf.constant([ih+1, iw+1], dtype='float32'))
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis=4)
        grid_yx = tf.clip_by_value(
            grid_yx, 0, tf.constant([ih+1, iw+1], dtype='float32'))
        # [B, H, W, 9, 4, 1]
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [
                              1, ih, iw, self.ks, 4, 1])
        # [B, H, W, 9, 4, 2]
        grid = tf.reshape(tf.concat([grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0,
                          grid_ix1, grid_iy0ix0], axis=-1), [bs, ih, iw, self.ks, 4, 2])
        # [B, H, W, 9, 4, 3]
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis=-1)
        # [B, H, W, 9, 2, 2]
        delta = tf.reshape(tf.concat(
            [grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis=-1), [bs, ih, iw, self.ks, 2, 2])
        # [B, H, W, 9, 2, 1] * [B, H, W, 9, 1, 2] = [B, H, W, 9, 2, 2]
        w = tf.expand_dims(delta[..., 0], axis=-1) * \
            tf.expand_dims(delta[..., 1], axis=-2)
        # [B, H+2, W+2, C]
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)],
                   [int(self.pw), int(self.pw)], [0, 0]])
        # [B, H, W, 9, 4, C]
        map_sample = tf.gather_nd(x, grid)
        # ([B, H, W, 9, 4, 1] * [B, H, W, 9, 4, C]).SUM(-2) * [B, H, W, 9, 1] = [B, H, W, 9, C]
        map_bilinear = tf.reduce_sum(tf.reshape(
            w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis=-2) * tf.expand_dims(mask, axis=-1)
        # [B, H, W, 9*C]
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        # [B, H, W, OC]
        output = tf.nn.conv2d(map_all, tf.reshape(
            self.kernel, [1, 1, -1, self.filters]), strides=self.stride, padding='SAME')
        if self.use_bias:
            output += self.bias
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)


class DCNv3(KL.Layer):
    def __init__(self, filters,
                 kernel_size,
                 # stride,
                 # padding,
                 # dilation = 1,
                 # deformable_groups = 1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):

        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1, 1, 1)
        # self.padding = padding
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        super(DCNv2, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_size + (int(input_shape[-1]), self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype='float32',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
            )

        # [kh, kw, ic, 3 * groups * kh, kw]--->3 * groups * kh * kw = oc [output channels]
        self.offset_kernel = self.add_weight(
            name='offset_kernel',
            shape=self.kernel_size +
            (input_shape[-1], 3 * self.deformable_groups *
             self.kernel_size[0] * self.kernel_size[1]),
            initializer='zeros',
            trainable=True,
            dtype='float32')

        self.offset_bias = self.add_weight(
            name='offset_bias',
            shape=(
                3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable=True,
            dtype='float32',
        )
        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (
            self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype='int32')
        self.patch_yx = tf.stack(tf.meshgrid(
            tf.range(-self.phw[1], self.phw[1] + 1), tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis=-1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])
        super(DCNv2, self).build(input_shape)

    def call(self, x):
        # x: [B, H, W, C]
        # offset: [B, H, W, ic] convx [kh, kw, ic, 3 * groups * kh * kw] ---> [B, H, W, 3 * groups * kh * kw]
        offset = tf.nn.conv2d(x, self.offset_kernel,
                              strides=self.stride, padding='SAME')
        offset += self.offset_bias
        bs, ih, iw, ic = x.shape.as_list()

        bs = tf.shape(x)[0]
        # [B, H, W, 18], [B, H, W, 9]
        oyox, mask = offset[..., :2*self.ks], offset[..., 2*self.ks:]
        mask = tf.nn.sigmoid(mask)
        # [H, W, 2]
        grid_yx = tf.stack(tf.meshgrid(
            tf.range(iw), tf.range(ih))[::-1], axis=-1)
        # [1, H, W, 9, 2]
        grid_yx = tf.reshape(
            grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        # [B, H, W, 9, 2]
        grid_yx = tf.cast(grid_yx, 'float32') + \
            tf.reshape(oyox, [bs, ih, iw, -1, 2])
        grid_iy0ix0 = tf.floor(grid_yx)
        grid_iy1ix1 = tf.clip_by_value(
            grid_iy0ix0 + 1, 0, tf.constant([ih+1, iw+1], dtype='float32'))
        # [B, H, W, 9, 1] * 2
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis=4)
        grid_iy0ix0 = tf.clip_by_value(
            grid_iy0ix0, 0, tf.constant([ih+1, iw+1], dtype='float32'))
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis=4)
        grid_yx = tf.clip_by_value(
            grid_yx, 0, tf.constant([ih+1, iw+1], dtype='float32'))
        # [B, H, W, 9, 4, 1]
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [
                              1, ih, iw, self.ks, 4, 1])
        # [B, H, W, 9, 4, 2]
        grid = tf.reshape(tf.concat([grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0,
                          grid_ix1, grid_iy0ix0], axis=-1), [bs, ih, iw, self.ks, 4, 2])
        # [B, H, W, 9, 4, 3]
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis=-1)
        # [B, H, W, 9, 2, 2]
        delta = tf.reshape(tf.concat(
            [grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis=-1), [bs, ih, iw, self.ks, 2, 2])
        # [B, H, W, 9, 2, 1] * [B, H, W, 9, 1, 2] = [B, H, W, 9, 2, 2]
        w = tf.expand_dims(delta[..., 0], axis=-1) * \
            tf.expand_dims(delta[..., 1], axis=-2)
        # [B, H+2, W+2, C]
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)],
                   [int(self.pw), int(self.pw)], [0, 0]])
        # [B, H, W, 9, 4, C]
        map_sample = tf.gather_nd(x, grid)
        # ([B, H, W, 9, 4, 1] * [B, H, W, 9, 4, C]).SUM(-2) * [B, H, W, 9, 1] = [B, H, W, 9, C]
        map_bilinear = tf.reduce_sum(tf.reshape(
            w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis=-2) * tf.expand_dims(mask, axis=-1)
        # [B, H, W, 9*C]
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        # [B, H, W, OC]
        output = tf.nn.conv2d(map_all, tf.reshape(
            self.kernel, [1, 1, -1, self.filters]), strides=self.stride, padding='SAME')
        if self.use_bias:
            output += self.bias
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)
