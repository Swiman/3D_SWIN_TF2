import tensorflow as tf


def DICE(y_true, logits):
    smooth = 1e-6
    y_pred = tf.nn.sigmoid(logits)  #  logits -> probs
    yp, yt = tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)
    num = 2 * tf.reduce_sum(yp * yt) + smooth
    denom = tf.reduce_sum(yp + yt) + smooth
    return 1 - num / denom


def WCE(y_true, logits):
    wce = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=logits, pos_weight=tf.constant(1.5)
    )
    return tf.reduce_mean(wce)


def Dice_WCE(y_true, logits):
    l1 = DICE(y_true, logits)
    l2 = WCE(y_true, logits)
    return l1 + 3 * l2


def BCE(y_true, logits):
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
    return tf.reduce_mean(bce)
