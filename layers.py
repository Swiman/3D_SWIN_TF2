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
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)  # (H or W or D, C/6,2)
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
            tf.random.normal(shape=(1, self.mask_num, np.prod(self.patch_dims))),
            trainable=True,
            name="mask_token",
        )

    def get_random_indeces(self):
        indeces = tf.argsort(tf.random.normal(shape=(self.token_num,)))
        return indeces[: self.mask_num], indeces[self.mask_num :]

    def call(self, x):
        B, H, W, D, C = x.shape
        patches = tf.extract_volume_patches(
            x, ksizes=self.patch_dims, strides=self.patch_dims, padding="VALID"
        )  # B, nH, nW, nD, patch_length
        patches_flat = tf.reshape(patches, (-1, self.token_num, tf.shape(patches)[-1]))
        mask_indeces, unmask_indeces = self.get_random_indeces()

        unmask_tokens = tf.gather(patches_flat, unmask_indeces, axis=1)
        unmask_pencs = tf.gather(self.pos_embeddings, unmask_indeces, axis=1)
        mask_pencs = tf.gather(self.pos_embeddings, mask_indeces, axis=1)

        a = self.proj(unmask_tokens) + tf.repeat(unmask_pencs, tf.shape(x)[0], axis=0)
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
        x = tf.pad(x, [[0, 0], [0, self.p0], [0, self.p1], [0, self.p2], [0, 0]])
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
    window_mask = window_partition(img_mask, window_size)  # num_windows,wH*wW*wD,1
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
        coords = np.stack(np.meshgrid(coords_h, coords_w, coords_d, indexing="ij"))
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
            tf.reshape(qkv, (-1, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4)
        )
        # each one:(batchsize*numwindows,num_heads,w0w1w2,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        # (batchsize*numwindows,num_heads,w0w1w2,w0w1w2)
        attn = q @ tf.transpose(k, (0, 1, 3, 2))
        flat_relative_position_index = tf.reshape(self.relative_position_index, (-1,))
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
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, proj_drop=proj_drop)

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
                x_norm, [[0, 0], [0, self.p0], [0, self.p1], [0, self.p2], [0, 0]]
            )

        x_norm_shifted = x_norm
        attn_mask = None

        if min(self.shift_size) > 0:
            x_norm_shifted = tf.roll(
                x_norm,
                shift=[-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]],
                axis=(1, 2, 3),
            )
            attn_mask = mask

        x_windows = window_partition(x_norm_shifted, self.window_size)  # *(-1,w0w1w2,C)

        attn_windows = self.w_attn(x_windows, attn_mask)

        attn_windows = tf.reshape(attn_windows, (-1, *self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.new_input_shape, self.window_size)

        if min(self.shift_size) > 0:
            shifted_x = tf.roll(
                shifted_x,
                shift=[self.shift_size[0], self.shift_size[1], self.shift_size[2]],
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
