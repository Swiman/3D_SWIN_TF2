import tensorflow as tf
import numpy as np
from models import *


m = get_DeformConv_Model((96, 96, 1), 3, 32, 3,
                         "glorot_normal", tf.nn.leaky_relu, conv_mode="deform_conv_v1")
m(np.ones((32, 96, 96, 1)))
print(m.summary(line_length=128))
