from random import seed
from random import random
from math import exp, sqrt
from random import randrange
import time


def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

#Create the list of weights and biases
def initialize_network(n_inputs, n_hidden, n_outputs, transfer_f, transfer_derfivative_f):
    network = list()
    weightspernode=n_inputs
    for nodesinhidden in n_hidden:
        layer = list()
        for i in range(nodesinhidden):
            layer.append({'weights': [random() * sqrt(1 / weightspernode) for j in range(weightspernode + 1)],
                 'transfer': transfer_f, 'transfer_derivative': transfer_derfivative_f})
        network.append(layer)
        weightspernode=nodesinhidden

    output_layer = list()
    for i in range(n_outputs):
        output_layer.append({'weights':[ random() * sqrt(1 / nodesinhidden) for j in range(nodesinhidden + 1) ],
            'transfer': transfer_sigmoid, 'transfer_derivative': transfer_sigmoid_derivative})
    network.append(output_layer)
    return network

def initialize_network_no_weights(n_inputs, n_hidden, n_outputs):
    network = list()
    weightspernode=n_inputs
    defaultweight = 1.0
    for nodesinhidden in n_hidden:
        network.append ( [{'weights': [defaultweight for i in range(weightspernode + 1)]} for i in range(nodesinhidden)] )
        weightspernode=nodesinhidden

    output_layer = [{'weights': [defaultweight for i in range(nodesinhidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer_rectified(output):
   return output * (output > 0)

def transfer_sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# Calculate the derivative of an neuron output
def transfer_rectified_derivative(output):
    return 1. * (output > 0)


def transfer_sigmoid_derivative(output):
    return output * (1.0 - output)

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = neuron['transfer'](activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * neuron['transfer_derivative'](neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, n_folds, l_rate, n_epoch, n_outputs, type, eventp, myapp):
    global timeleft
    step = 100.0/(n_folds*n_epoch*len(train))
    myapp.currentfold += 1
    myapp.currentcycle = 0
    for epoch in range(n_epoch):
        sum_error = 0
        myapp.valuemydata.networklist = network
        myapp.currentcycle += 1
        if myapp.stoptrainning:
            break
        i = 1
        for row in train:
            myapp.valuepb += step
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            myapp.mean_qerror = sum_error / i
            i += 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            if myapp.stoptrainning:
                break
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, myapp.l_rate, sum_error))

    if type == 'train':
        eventp.trainfinished(myapp)

#Works with btnGo
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))
#accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.0



    #MyApp.drawinglblblackkv(self, self.parentlayout.ids.idlayoutmiddle, valuemydata)
# Backpropagation Algorithm With Stochastic Gradient Descent

def back_propagation(train, test, eventp, myapp, n_folds, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    myapp.network = initialize_network(n_inputs, n_hidden, n_outputs, transfer_sigmoid, transfer_sigmoid_derivative)
    train_network(myapp.network, train, n_folds, l_rate, n_epoch, n_outputs, 'backp', eventp, myapp)
    predictions = list()
    for row in test:
        if myapp.stoptrainning:
            break
        prediction = predict(myapp.network, row)
        predictions.append(prediction)
    return(predictions)



def evaluate_algorithm(dataset, algorithm, eventp, myapp, n_folds, *args):
    global progressbar
    scores = list()
    folds = cross_validation_split(dataset, n_folds)
    for fold in folds:
        if myapp.stoptrainning:
            break
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            if myapp.stoptrainning:
                break
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, eventp, myapp, n_folds, *args)
        if myapp.stoptrainning:
            break
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    eventp.evevaluatealgorithm(myapp, scores)



def predictfunc(myapp,mydataset):
    print('len mydataset:{}' .format(len(mydataset)))
    myapp.valuepredict = 0
    myapp.totalpredict = 0
    myapp.correctpredict = 0
    step = 100.0/len(mydataset)
    predictions = list()
    myapp.mistakeslist = list()
    for row in mydataset:
        if myapp.stoptrainning:
            break
        predicted = predict(myapp.network, row)
        myapp.valuepredict += step
        predictions.append(predicted)
        myapp.totalpredict += 1
        if row[-1] == predicted:
            myapp.correctpredict += 1
            myapp.mistakeslist.append(True)
        else:
            myapp.mistakeslist.append(False)
    actual = [row[-1] for row in mydataset]
    accuracy = accuracy_metric(actual, predictions)
    myapp.myevent.predictfinished(myapp, accuracy)
