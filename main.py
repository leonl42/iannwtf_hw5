import tensorflow as tf
from tensorflow.python.ops.nn_ops import dropout
from util import load_data, test
from model import MyModel
import numpy as np
from classify import classify
from rField import rField

tf.keras.backend.clear_session()
train_ds, test_ds = load_data()

layers = [
    tf.keras.layers.Conv2D(filters=12, kernel_size=(
        5, 5), strides=(1, 1), padding="same", activation=tf.nn.sigmoid, kernel_regularizer="l1_l2"),
    tf.keras.layers.AveragePooling2D(
        pool_size=2, strides=(2, 2), padding="same"),
    tf.keras.layers.Conv2D(filters=5, kernel_size=(
        3, 3), strides=(1, 1), padding="valid", activation=tf.nn.sigmoid, kernel_regularizer="l1_l2"),
    tf.keras.layers.AveragePooling2D(
        pool_size=2, strides=(2, 2), padding="same"),
    tf.keras.layers.Conv2D(filters=5, kernel_size=(
        3, 3), strides=(1, 1), padding="same", activation=tf.nn.sigmoid, kernel_regularizer="l1_l2"),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(
        120, kernel_regularizer="l1_l2", activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(
        rate=0.5
    ),
    tf.keras.layers.Dense(
        84, kernel_regularizer="l1_l2", activation=tf.nn.sigmoid),
    tf.keras.layers.Dropout(
        rate=0.5
    ),
    tf.keras.layers.Dense(
        10, kernel_regularizer="l1_l2", activation=tf.nn.softmax)]

model = MyModel(layers)

rfield = rField()
rfield.add_rfield([0, 0])
p = [[[0] for k in range(28)] for i in range(28)]
rfield.compute_rfield_from_layers(layers[0:5], [28, 28])
rfield.plot(p, (6, 6), (0, 0))

with tf.device('/device:gpu:0'):
    classify(model, tf.keras.optimizers.Adam(
        0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07), 100, train_ds, test_ds)
