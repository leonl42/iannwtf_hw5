import tensorflow as tf
from tensorflow.python.ops.nn_ops import dropout
from util import load_data, test, visualize
from model import MyModel
import numpy as np
from classify import classify
from rField import rField

tf.keras.backend.clear_session()
train_ds, valid_ds, test_ds = load_data()

layers = [
    # feature learning
    tf.keras.layers.Conv2D(filters = 12, kernel_size = 5, strides=1,padding="same",activation='relu'),
    tf.keras.layers.Conv2D(filters = 12, kernel_size = 3, strides=1,padding="same",activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding="same"),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = 9, strides=1,padding="same", kernel_regularizer="l1_l2",activation='relu'),
    #tf.keras.layers.Conv2D(filters = 24, kernel_size = 3, strides=1,padding="same", activation='relu'),
    tf.keras.layers.GlobalAvgPool2D(),
    # classification
    tf.keras.layers.Dense(10, kernel_regularizer="l1_l2", activation='softmax')]

model = MyModel(layers)

rfield = rField()
# rfield for positon[0,0] and [1,1]
rfield.add_rfield([0, 0])
rfield.add_rfield([1, 1])
p = [[[0] for k in range(28)] for i in range(28)]
rfield.compute_rfield_from_layers(layers, [28, 28])
rfield.plot(p, (6, 6), (0, 0))

with tf.device('/device:gpu:0'):
    results, trained_model = classify(model, tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), 10, train_ds, valid_ds)

    _, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy(),False)
    print("Accuracy:", test_accuracy)

    visualize(results[0],results[1],results[2])
