from feedforwardnn import feed_forward
from general_fns import dot

def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)

    output_deltas = [output * (1-output) * (output-target)
                     for output, target in zip(outputs, targets)]


    #adjust weights for ouput layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            output_neuron[j] -= output_deltas[i] * hidden_output


    #back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * ( 1 - hidden_output)
                     * dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]


    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

