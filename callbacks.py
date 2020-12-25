import tensorflow as tf
from tensorflow import keras

def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

class prediction_history(keras.callbacks.Callback):
    def __init__(self):
        self.predhis_train = []
        self.predhis_valid = []
    def on_epoch_end(self, epoch, logs={}):
        self.predhis_train.append(model.predict(x_train))
        self.predhis_valid.append(model.predict(x_val))