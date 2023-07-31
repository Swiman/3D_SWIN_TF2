# import pandas as pd
# import numpy as np
# from dataloader import *
# from models import *

# TFRecord_filenames = [
#     "./tfrecords/tsdc_abus_0.tfrec",
#     "./tfrecords/tsdc_abus_1.tfrec",
#     "./tfrecords/tsdc_abus_2.tfrec",
#     "./tfrecords/tsdc_abus_3.tfrec",
#     "./tfrecords/tsdc_abus_4.tfrec",
# ]
# tr_ds = prepare_dataset(
#     TFRecord_filenames[:-1],
#     batch_size=4,
#     patch_size=(64, 64, 64),
# )
# for _ in range(3):
#     print("_______________88888888888888888")
#     for x, y, c in tr_ds:
#         print(x.shape, y.shape, c)
#         break

#     # print(y)
#     break

# swin_args = {
#     "input_shape": (64, 64, 64, 1),
#     "embed_dim": 48,
#     "window_size": [7, 7, 7],
#     "patch_size": [4, 4, 4],
#     "mask_ratio": 0.01,
#     "depths": [2, 2, 2, 2],
#     "mlp_ratio": 4.0,
#     "num_heads": [3, 6, 12, 24],
#     "patch_norm": True,
#     "qkv_bias": True,
#     "drop_path": 0.5,
#     "attn_drop": 0.0,
#     "proj_drop": 0.0,
# }


# # m = get_XNet_V2(**swin_args)
# # o = m(np.ones((1, 64, 64, 64, 1)))
# # print(m.summary(line_length=128))
def temp(a, *args):
    print(a, args[0])


temp(1, 2)
