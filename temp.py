import numpy as np
import tensorflow as tf
from models import get_XNet
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

# xnet = get_XNet(**swin_args)
model = tf.keras.models.load_model("/home/user/abus/fold1_xnet_w7_p4_e48.keras")
print(model.summary(line_length=128))
