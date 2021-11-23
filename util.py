import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def load_data():
    """
    Loading and preprocessing the data.
        Results:
            train_ds,test_ds: the preprocessed datasets
    """

    train_ds, test_ds = tfds.load(name="fashion_mnist", split=[
                                  "train", "test"], as_supervised=True)
    train_ds = preprocess(train_ds)
    test_ds = preprocess(test_ds)
    return train_ds, test_ds


def preprocess(ds):
    """
    Preparing our data for our model.
      Args:
        ds: the dataset we want to preprocess
      Results:
        ds: preprocessed dataset
    """
    ds = ds.map(lambda feature, target: (feature, tf.cast(target, tf.int32)))
    ds = ds.map(lambda feature, target: (feature, tf.one_hot(target, 10)))
    ds = ds.map(lambda feature, target: (
        tf.cast(feature, tf.float32), tf.cast(target, tf.float32)))

    ds = ds.cache()
    # shuffle, batch, prefetch our dataset
    ds = ds.shuffle(5000)
    ds = ds.batch(64)
    ds = ds.prefetch(20)
    return ds


def train_step(model, input, target, loss_function, optimizer):
    """
    Performs a forward and backward pass for  one dataponit of our training set
      Args:
        model: our created MLP model (MyModel object)
        input: our input (tensor)
        target: our target (tensor)
        loss_funcion: function we used for calculating our loss (keras function)
        optimizer: our optimizer used for packpropagation (keras function)
      Results:
        loss: our calculated loss for the datapoint (float)
      """

    with tf.GradientTape() as tape:

        # forward step
        prediction = model(input)

        # calculating loss
        loss = loss_function(target, prediction)

        # calculaing the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

    # updating weights and biases
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test(model, test_data, loss_function):
    """
    Test our MLP, by going through our testing dataset,
    performing a forward pass and calculating loss and accuracy
      Args:
        model: our created MLP model (MyModel object)
        test_data: our preprocessed test dataset (set of tuples with tensors)
        loss_funcion: function we used for calculating our loss (keras function)
      Results:
          loss: our mean loss for this epoch (float)
          accuracy: our mean accuracy for this epoch (float)
    """

    # initializing lists for accuracys and loss
    accuracy_aggregator = []
    loss_aggregator = []

    for (input, target) in test_data:

        # forward step
        prediction = model(input)

        # calculating loss
        loss = loss_function(target, prediction)

        # add loss and accuracy to the lists
        loss_aggregator.append(loss.numpy())
        for t, p in zip(target, prediction):
            accuracy_aggregator.append(
                tf.cast(tf.math.argmax(t) == tf.math.argmax(p), tf.float32))

    # calculate the mean of the loss and accuracy (for this epoch)
    loss = tf.reduce_mean(loss_aggregator)
    accuracy = tf.reduce_mean(accuracy_aggregator)

    return loss, accuracy


def visualize(train_losses, valid_losses, valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
    Args:
      train_losses = mean training losses per epoch
      valid_losses = mean testing losses per epoch
      valid_accuracies = mean accuracies (testing dataset) per epoch
    """

    titles = ["SGD", "SGD_l1-l2", "SGD_drop-0.5", "SGD_l1-l2_drop-0.5",
              "Adam", "Adam_l1-l2", "Adam_drop-0.5", "Adam_l1-l2_drop-0.5", ]
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(13, 6)

    # making a grid with subplots
    for i in range(2):
        for j in range(4):
            axs[i, j].plot(train_losses[i*4+j])
            axs[i, j].plot(valid_losses[i*4+j])
            axs[i, j].plot(valid_accuracies[i*4+j])
            last_accuracy = valid_accuracies[i*4+j][-1].numpy()
            axs[i, j].sharex(axs[0, 0])
            axs[i, j].set_title(
                titles[i*4+j]+" \n Last Accuracy: "+str(round(last_accuracy, 4)))

    fig.legend([" Train_ds loss", " Valid_ds loss", " Valid_ds accuracy"])
    plt.xlabel("Training epoch")
    fig.tight_layout()
