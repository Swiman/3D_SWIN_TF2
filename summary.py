import datetime
import tensorflow as tf


class summary_logger(object):
    def __init__(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def write(self,tr_loss, vl_loss, step):
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss',tr_loss,step=step)
        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss',vl_loss,step=step)