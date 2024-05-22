import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class TensorBoardCallback(Callback):
    """
    Custom TensorBoard callback to log training metrics to TensorBoard.

    :param log_dir: directory to save the logs
    """

    def __init__(self, log_dir):
        super(TensorBoardCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        """
        Log the metrics to TensorBoard at the end of each epoch.

        :param epoch: the current epoch
        :param logs: the logs dict to be logged
        """

        with self.writer.as_default():
            if logs is not None:
                for key, value in logs.items():
                    tf.summary.scalar(key, value, step=epoch)
            self.writer.flush()

    def on_train_end(self, logs=None):
        """
        Close the writer when training ends.
        """

        self.writer.close()