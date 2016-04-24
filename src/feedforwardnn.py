import math
from general_fns import dot, sigmoid


def feed_forward(neural_network, input_vector):
    """
    :param neural_network: list of lists of lists of weights
    :param input_vector: weights
    :return: output from forward propagating the input
    """

    outputs = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)

        input_vector = output

    return outputs


def neuron_output(weights, outputs):
    return sigmoid(dot(weights, outputs))


xor_network = [
                [[20,20, -30],
                 [20, 20, -10]],
                [[-60, 60, -30]]]


for x in [0, 1]:
    for y in [0,1]:
        print (x, y, feed_forward(xor_network, [x, y])[-1])


