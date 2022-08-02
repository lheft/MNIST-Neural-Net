from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
import random
from random import randrange
class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        self.input_size = 784  # Size of image vectors because 28 x 28
        self.number_of_classes = 10  # Number of possible labels (0,1,2...9)
        self.learning_rate = 0.5  # As instructed in Google Doc
        self.batch_size = 100  # As instructed in Google Doc
        self.Weight = np.zeros((self.input_size, self.number_of_classes))
        self.bias = np.zeros(self.number_of_classes)

    def call(self, inputs):  # Return softmax probabilities for each digit for each image in batch

        logits = (inputs @ self.Weight) + self.bias
        softmax = np.exp(logits) / np.sum(np.exp(logits), keepdims=True, axis=1,)
        return softmax

    def loss(self, probabilities, labels):  # Implement cross-entropy loss functino
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step).
        NOTE: This function is not actually used for gradient descent
        in this assignment, but is a sanity check to make sure model
        is learning.
        """
        batch = 0
        average_loss = 0
        shape = probabilities.shape[0]
        for i in range(shape):
            batch = batch + 1
            average_loss = average_loss + probabilities[i][labels[i]]
        return -(np.log( average_loss / batch))

    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. The learning
        algorithm for updating weights and biases mentioned in
        class works for one image, but because we are looking at
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        """
        probabilities[np.arange(len(probabilities)), labels] -= 1
        probabilities = probabilities * -1
        gradB = (self.learning_rate * np.ones(self.batch_size) @ probabilities) / self.batch_size
        transposed_inputs = inputs.T @ probabilities
        gradW = (self.learning_rate * transposed_inputs) / self.batch_size

        return gradB, gradW

    def accuracy(self, probabilities, labels):

        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        """
        # TODO: calculate the batch accuracy
        count = 0
        shape = probabilities.shape[0]

        for i in range(shape):
            if np.argmax(probabilities[i]) == labels[i]:
                count = count + 1

        return count / len(probabilities)

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        self.Weight = self.Weight + gradW
        self.bias = self.bias + gradB


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    '''

    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains

    batches = int(len(train_inputs) / model.batch_size)

    for i in range(batches):
        inputs = train_inputs[i * model.batch_size: i * model.batch_size + model.batch_size]
        labels = train_labels[i * model.batch_size: i * model.batch_size + model.batch_size]

        val = model.call(inputs)
        gradB, gradW = model.back_propagation(inputs, val, labels)

        model.gradient_descent(gradW, gradB)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    """

    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    prob = model.call(test_inputs)
    accuracy = model.accuracy(prob, test_labels)
    return accuracy


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    '''
    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels

    # TODO: Create Model

    # TODO: Train model by calling train() ONCE on all data

    # TODO: Test the accuracy by calling test() after running train()

    # TODO: Visualize the data by using visualize_results()

    data_train, labels_train = get_data()
    data_test, labels_test = get_data('/Users/loganheft/Downloads/data/test-images.gz',
                                      '/Users/loganheft/Downloads/data/test-labels.gz',
                                      10000)

    model = Model()
    train(model, data_train, labels_train)
    visualize_results(data_test[42:52], model.call(data_test[42:52]), labels_test[42:52])
    print(test(model, data_test, labels_test))


if __name__ == '__main__':
    main()
