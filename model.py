import tensorflow as tf


class MyModel(tf.keras.Model):
    """
    Our own custon MLP model, which inherits from the keras.Model class
      Functions:
        init: constructor of our model
        call: performs forward pass of our model
    """

    def __init__(self, layers):
        """
        Constructs our model.
        """

        super(MyModel, self).__init__()

        self._layers = layers

    def call(self, inputs):
        """
        Performs a forward step in our MLP
          Args:
            inputs: our preprocessed input data, we send through our model
          Results:
            output: the predicted output of our input data
        """

        output = inputs
        for layer in self._layers:
            output = layer(output)

        return output
